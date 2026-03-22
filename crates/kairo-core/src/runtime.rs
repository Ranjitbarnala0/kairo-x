//! Main runtime execution loop (§16).
//!
//! This module orchestrates the complete KAIRO-X workflow:
//! 1. **Init**: Fingerprint the project.
//! 2. **Plan**: Read spec → LLM plan → two-pass JSON → build graph.
//! 3. **Execute**: Controller step → action selection → execute → verify → update graph → checkpoint.
//! 4. **Complete**: All itch bits cleared → summary output.
//!
//! The runtime integrates all subsystems: arena, session manager, context engine,
//! verification engine, enforcement, and parallel scheduler.

use crate::arena::Arena;
use crate::arena::node::{NodeStatus, Priority};
use crate::arena::priority_queue::PendingEntry;
use crate::arena::query::StatusSummary;
use crate::classify::rules::classify_response;
use crate::controller::input_assembly::{
    self, encode_bool, encode_count, encode_float, encode_node_status, encode_priority, slots,
    InputPacket,
};
use crate::controller::Controller;
use crate::enforcement::compliance::ComplianceTracker;
use crate::enforcement::selector::{effective_intensity, select_template};
use crate::enforcement::templates::Template;
use crate::fingerprint::ProjectFingerprint;
use crate::parallel::ParallelScheduler;
use crate::persistence::CheckpointManager;
use crate::plan_parser::{self, PlanSummary};
use crate::session::manager::SessionManager;
use crate::session::token_tracker::{CostMode, TokenTracker};
use crate::tools::file_locks::{FileLockTable, TrackId};
use crate::tools::snapshots::SnapshotStore;
use crate::verification::deterministic::DeterministicVerifier;
use crate::verification::llm_audit::LlmAuditor;
use crate::verification::runner::VerificationRunner;
use kairo_llm::bridge::{BridgeCallResult, BridgeConfig, LLMBridge};
use kairo_llm::call::{self, LLMCallType, Message};
use kairo_llm::context_request;
use kairo_llm::response::{ContextRequestKind, ResponseClass};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Runtime errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("Project fingerprinting failed: {0}")]
    Fingerprint(String),

    #[error("Plan generation failed: {0}")]
    Planning(String),

    #[error("LLM bridge error: {0}")]
    Bridge(#[from] kairo_llm::bridge::BridgeError),

    #[error("Verification error: {0}")]
    Verification(String),

    #[error("Persistence error: {0}")]
    Persistence(#[from] crate::persistence::PersistenceError),

    #[error("All nodes exhausted retries. {failed} nodes failed.")]
    AllNodesFailed { failed: usize },

    #[error("Token budget exhausted")]
    BudgetExhausted,

    #[error("User cancelled")]
    Cancelled,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Edit error: {0}")]
    EditFailed(String),

    #[error("Node not found: {0}")]
    NodeNotFound(u32),

    #[error("Arena error: {0}")]
    Arena(#[from] crate::arena::ArenaError),

    #[error("LLM response contained no code implementation")]
    NoImplementation,

    #[error("File locked by another track, re-read needed: {0}")]
    FileLockedReRead(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

// ---------------------------------------------------------------------------
// Outcome enums
// ---------------------------------------------------------------------------

/// Outcome of executing a single node.
#[derive(Debug)]
pub enum NodeOutcome {
    /// Implementation was produced and edits applied.
    Implemented,
    /// Node was too large and was decomposed into sub-nodes.
    Decomposed { sub_node_count: usize },
    /// Context was provided (NeedsContext was handled).
    ContextProvided,
    /// Execution failed for the given reason.
    Failed(String),
    /// LLM response contained no code — only explanation text.
    NoCode,
}

/// Outcome of verifying a single node.
#[derive(Debug)]
pub enum VerifyOutcome {
    /// All verification layers passed.
    Pass,
    /// Deterministic (L1) verification failed.
    DeterministicFail { details: String },
    /// LLM audit (L2) verification failed.
    AuditFail { issues: Vec<String> },
    /// Verification was skipped (no deterministic checks available).
    Skipped,
}

// ---------------------------------------------------------------------------
// Runtime configuration
// ---------------------------------------------------------------------------

/// Configuration for the KAIRO-X runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Project root directory.
    pub project_root: PathBuf,
    /// Provider configuration (primary, fallback, audit).
    pub provider_config: kairo_llm::providers::ProviderConfig,
    /// Cost mode (thorough, balanced, efficient).
    pub cost_mode: CostMode,
    /// Maximum parallel tracks.
    pub max_parallel_tracks: u8,
    /// Checkpoint interval (every N steps).
    pub checkpoint_interval: u32,
    /// Maximum checkpoints to retain.
    pub max_checkpoints: usize,
    /// Token budget limit in tokens (None = unlimited).
    pub token_budget: Option<u64>,
    /// Cost budget limit in microdollars (None = unlimited).
    pub cost_limit_microdollars: Option<u64>,
    /// Bridge retry configuration.
    pub bridge_config: BridgeConfig,
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/// Complete runtime state for the KAIRO-X agent.
pub struct Runtime {
    /// The execution graph.
    pub(crate) arena: Arena,
    /// Project fingerprint.
    pub(crate) fingerprint: ProjectFingerprint,
    /// Session manager.
    pub(crate) session_manager: SessionManager,
    /// Token and cost tracker.
    pub(crate) token_tracker: TokenTracker,
    /// Compliance tracker (rolling LLM response quality).
    pub(crate) compliance: ComplianceTracker,
    /// Parallel execution scheduler.
    pub(crate) scheduler: ParallelScheduler,
    /// Per-file write locks for parallel safety.
    pub(crate) file_locks: FileLockTable,
    /// File snapshot store for rollback.
    pub(crate) snapshots: SnapshotStore,
    /// Checkpoint manager.
    pub(crate) checkpoint_manager: CheckpointManager,
    /// LLM bridge (provider communication).
    pub(crate) bridge: LLMBridge,
    /// Neural controller for enforcement intensity adjustment.
    pub(crate) controller: Controller,
    /// Runtime configuration.
    pub(crate) config: RuntimeConfig,
    /// Count of nodes that passed on first implementation attempt.
    pub(crate) first_pass_count: usize,
    /// Count of nodes that required fixes.
    pub(crate) fix_count: usize,
    /// Count of nodes that required decomposition.
    pub(crate) decompose_count: usize,
}

impl Runtime {
    /// Restore internal state from a checkpoint. Used by the CLI resume flow.
    pub fn restore_checkpoint_state(
        &mut self,
        arena: Arena,
        token_tracker: TokenTracker,
        compliance: ComplianceTracker,
        controller_state: &[f32],
        session_snapshot: Option<crate::session::manager::SessionManagerSnapshot>,
    ) {
        self.arena = arena;
        self.token_tracker = token_tracker;
        self.compliance = compliance;
        if !controller_state.is_empty() {
            let ok = self.controller.deserialize_state(controller_state);
            if ok {
                tracing::info!("Controller recurrent state restored");
            }
        }
        if let Some(snapshot) = session_snapshot {
            self.session_manager.restore_from_snapshot(snapshot);
            tracing::info!("Session summaries restored");
        }
    }

    /// Read-only access to the token tracker for display purposes.
    pub fn token_tracker(&self) -> &TokenTracker {
        &self.token_tracker
    }

    /// Read-only access to the arena for status queries.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Read-only access to the project fingerprint.
    pub fn fingerprint(&self) -> &ProjectFingerprint {
        &self.fingerprint
    }

    /// Read-only access to the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Task completion summary
// ---------------------------------------------------------------------------

/// Summary of a completed task.
#[derive(Debug)]
pub struct TaskCompletionSummary {
    /// Status summary from the arena.
    pub status: StatusSummary,
    /// Total nodes including decomposed sub-nodes.
    pub total_nodes_processed: usize,
    /// Nodes that passed on first implementation.
    pub first_pass_success: usize,
    /// Nodes that required fixes.
    pub required_fixes: usize,
    /// Nodes that required decomposition.
    pub required_decomposition: usize,
    /// Total LLM calls made.
    pub total_llm_calls: u32,
    /// Total input tokens.
    pub total_input_tokens: u64,
    /// Total output tokens.
    pub total_output_tokens: u64,
    /// Estimated cost in microdollars.
    pub estimated_cost_microdollars: u64,
}

// ---------------------------------------------------------------------------
// Runtime implementation — core helpers
// ---------------------------------------------------------------------------

impl Runtime {
    /// Initialize the runtime from configuration.
    pub fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        let fingerprint =
            crate::fingerprint::detector::fingerprint_project(&config.project_root);

        tracing::info!(
            language = %fingerprint.primary_language,
            "Project fingerprinted"
        );

        let arena = Arena::new();
        let session_manager = SessionManager::new();
        let cost_limit = config.cost_limit_microdollars.unwrap_or(0);
        let token_budget = config.token_budget.unwrap_or(0);
        // Use the actual provider's cost rates from the config, not hardcoded values.
        let primary = &config.provider_config.primary;
        let token_tracker = TokenTracker::new(
            primary.cost_per_input_mtok as u64,
            primary.cost_per_output_mtok as u64,
            cost_limit,
            token_budget,
            config.cost_mode,
        );
        let compliance = ComplianceTracker::new();
        let scheduler = ParallelScheduler::new(config.max_parallel_tracks);
        let file_locks = FileLockTable::new();
        let snapshots = SnapshotStore::new();

        let checkpoint_dir = config.project_root.join(".kairo/checkpoints");
        let checkpoint_manager = CheckpointManager::new(
            checkpoint_dir,
            config.checkpoint_interval,
            config.max_checkpoints,
        );

        let provider_manager = kairo_llm::providers::ProviderManager::new(
            config.provider_config.clone(),
            config.bridge_config.failover_threshold,
        );
        let bridge = LLMBridge::new(provider_manager, config.bridge_config.clone());
        let controller = Controller::zeros();

        Ok(Self {
            arena,
            fingerprint,
            session_manager,
            token_tracker,
            compliance,
            scheduler,
            file_locks,
            snapshots,
            checkpoint_manager,
            bridge,
            controller,
            config,
            first_pass_count: 0,
            fix_count: 0,
            decompose_count: 0,
        })
    }

    pub fn status(&self) -> StatusSummary {
        self.arena.status_summary()
    }

    pub fn can_terminate(&self) -> bool {
        self.arena.can_terminate()
    }

    pub fn is_budget_exhausted(&self) -> bool {
        self.token_tracker.is_budget_exhausted()
    }

    pub fn itch_stats(&self) -> (u32, usize) {
        self.arena.itch_stats()
    }

    pub fn completion_summary(&self) -> TaskCompletionSummary {
        let status = self.arena.status_summary();
        TaskCompletionSummary {
            status: status.clone(),
            total_nodes_processed: status.completed + status.failed,
            first_pass_success: self.first_pass_count,
            required_fixes: self.fix_count,
            required_decomposition: self.decompose_count,
            total_llm_calls: self.arena.total_llm_calls(),
            total_input_tokens: self.token_tracker.total_input,
            total_output_tokens: self.token_tracker.total_output,
            estimated_cost_microdollars: self.token_tracker.total_cost_microdollars,
        }
    }

    /// Select the enforcement template and intensity for a given action and priority.
    pub fn select_enforcement(
        &mut self,
        action: LLMCallType,
        priority: Priority,
    ) -> (Template, f32) {
        let selection = select_template(action, priority);

        let packet = InputPacket::new();
        let (_, itch_total) = self.arena.itch_stats();
        let output = self.controller.step(&packet, 0, itch_total > 0);
        let neural_adj = output.heads.enforcement_intensity_adj;

        let intensity = effective_intensity(priority, neural_adj, self.config.cost_mode);
        (selection.template, intensity)
    }

    pub fn record_compliance(&mut self, response_class: ResponseClass) {
        self.compliance.record(response_class.is_success());
    }

    pub fn maybe_checkpoint(&mut self) -> Result<(), RuntimeError> {
        if self
            .checkpoint_manager
            .should_checkpoint(self.arena.current_step)
        {
            self.checkpoint_manager.create_checkpoint(
                self.arena.current_step,
                &self.arena,
                &self.token_tracker,
                &self.compliance,
                &self.controller,
                &self.session_manager,
                &self.snapshots,
            )?;
        }
        Ok(())
    }

    /// Model name from the active provider.
    fn active_model(&self) -> String {
        self.bridge.active_model().to_string()
    }

    /// Maximum output tokens from the active provider.
    fn active_max_output(&self) -> u32 {
        self.bridge.active_max_output_tokens()
    }
}

// ---------------------------------------------------------------------------
// 1. generate_plan
// ---------------------------------------------------------------------------

impl Runtime {
    /// Generate an execution plan from a task specification.
    ///
    /// Two-pass: natural language plan → JSON conversion → parse → build graph.
    /// On parse failure: retry once, then fall back to numbered-list parsing.
    /// Prepends infrastructure prerequisite nodes when needed.
    pub async fn generate_plan(
        &mut self,
        task_spec: &str,
    ) -> Result<PlanSummary, RuntimeError> {
        let model = self.active_model();
        let max_out = self.active_max_output();

        let (tpl, intensity) = self.select_enforcement(LLMCallType::Plan, Priority::Critical);
        let enforcement = tpl.render(intensity);

        // --- Pass 1: natural language plan -----------------------------------
        tracing::info!("Plan pass 1: natural language plan");
        let plan_req = call::assemble_call(
            LLMCallType::Plan, "Task Plan", task_spec,
            "", "", &enforcement, true, &model, max_out,
        );
        let plan_text = self.call_and_extract(&plan_req, 0).await?;

        // --- Pass 2: convert to JSON -----------------------------------------
        tracing::info!("Plan pass 2: JSON conversion");
        let json_req = call::LLMRequest {
            messages: vec![
                Message::system(call::SYSTEM_PROMPT.to_string()),
                Message::user(format!("{enforcement}\n\nTask: {task_spec}")),
                Message::assistant(plan_text.clone()),
                Message::user(call::PLAN_TO_JSON_PROMPT.to_string()),
            ],
            model: model.clone(),
            temperature: 0.0,
            max_output_tokens: max_out,
            call_type: LLMCallType::Plan,
            stop_sequences: Vec::new(),
        };
        let json_text = self.call_and_extract(&json_req, 0).await?;

        // --- Parse JSON (with retry + numbered-list fallback) ----------------
        let components = match plan_parser::parse_plan_json(&json_text) {
            Ok(c) => c,
            Err(first_err) => {
                tracing::warn!(error = %first_err, "JSON parse failed, asking LLM to fix");
                let fix_prompt = call::fix_json_prompt(&first_err.to_string());
                let fix_req = call::LLMRequest {
                    messages: vec![
                        Message::system(call::SYSTEM_PROMPT.to_string()),
                        Message::user(format!("{enforcement}\n\nTask: {task_spec}")),
                        Message::assistant(plan_text.clone()),
                        Message::user(call::PLAN_TO_JSON_PROMPT.to_string()),
                        Message::assistant(json_text.clone()),
                        Message::user(fix_prompt),
                    ],
                    model: model.clone(),
                    temperature: 0.0,
                    max_output_tokens: max_out,
                    call_type: LLMCallType::Plan,
                    stop_sequences: Vec::new(),
                };
                let fixed = self.call_and_extract(&fix_req, 0).await?;
                match plan_parser::parse_plan_json(&fixed) {
                    Ok(c) => c,
                    Err(second_err) => {
                        tracing::warn!(error = %second_err, "Fix also failed, numbered-list fallback");
                        plan_parser::parse_plan_numbered_list(&plan_text).map_err(|e| {
                            RuntimeError::Planning(format!(
                                "All parse methods failed: json={first_err}, fix={second_err}, list={e}"
                            ))
                        })?
                    }
                }
            }
        };

        // --- Build graph -----------------------------------------------------
        let mut summary =
            plan_parser::build_graph_from_plan(&mut self.arena, &components, 0)
                .map_err(|e| RuntimeError::Planning(e.to_string()))?;

        // --- Infrastructure prerequisites ------------------------------------
        let prereqs = crate::verification::infrastructure::plan_prerequisites(
            &self.fingerprint,
            &self.config.project_root,
        );
        if !prereqs.is_empty() {
            tracing::info!(count = prereqs.len(), "Prepending infrastructure prereqs");
            let mut prereq_indices = Vec::new();
            for p in &prereqs {
                let node = crate::arena::node::Node::new(p.title.clone(), p.priority);
                let idx = self.arena.alloc(node);
                self.arena.set_spec(idx, p.spec.clone());
                self.arena.pending_queue.push(PendingEntry {
                    node_idx: idx,
                    priority: p.priority,
                });
                prereq_indices.push(idx);
            }
            for &plan_idx in &summary.node_indices {
                for &pre_idx in &prereq_indices {
                    self.arena.add_dependency(plan_idx, pre_idx)?;
                }
            }
            summary.total_components += prereqs.len();
        }

        tracing::info!(
            total = summary.total_components,
            critical = summary.critical_count,
            standard = summary.standard_count,
            mechanical = summary.mechanical_count,
            "Plan built into execution graph"
        );
        Ok(summary)
    }
}

// ---------------------------------------------------------------------------
// 2. execute_node
// ---------------------------------------------------------------------------

impl Runtime {
    /// Execute a single node: call LLM for implementation, apply edits.
    ///
    /// `track_id` identifies the parallel track (0 in sequential mode).
    pub async fn execute_node(
        &mut self,
        node_idx: u32,
        track_id: TrackId,
    ) -> Result<NodeOutcome, RuntimeError> {
        if node_idx == 0 || node_idx as usize >= self.arena.nodes.len() {
            return Err(RuntimeError::NodeNotFound(node_idx));
        }

        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::Implementing;
        self.arena.advance_step()?;

        let title = self.arena.get(node_idx).expect("node must exist").title.to_string();
        let priority = self.arena.get(node_idx).expect("node must exist").priority;
        let spec = self.arena.get_spec(node_idx).unwrap_or("").to_string();

        tracing::info!(node = node_idx, title = %title, "Executing node");

        let (tpl, intensity) = self.select_enforcement(LLMCallType::Implement, priority);
        let enforcement = tpl.render(intensity);

        let model = self.active_model();
        let max_out = self.active_max_output();
        let ctx = self.build_node_context(node_idx);
        let context_fp = xxhash_rust::xxh3::xxh3_64(ctx.as_bytes());

        let (session_id, is_new) = self.session_manager.get_or_create_session(
            node_idx, context_fp, LLMCallType::Implement,
        );

        let mut request = call::assemble_call(
            LLMCallType::Implement, &title, &spec,
            &ctx, "", &enforcement, is_new, &model, max_out,
        );

        // SESSION REPLAY: When continuing a session, prepend stored message
        // history so the LLM has full context from prior turns.
        if !is_new {
            if let Some(session) = self.session_manager.get_session(session_id) {
                if !session.messages.is_empty() {
                    let new_msgs = std::mem::take(&mut request.messages);
                    request.messages = session.messages.clone();
                    request.messages.extend(new_msgs);
                }
            }
        }

        // Call bridge with NeedsContext retry loop (up to 3 rounds).
        // SAFETY: Track previously requested paths to detect hallucination loops
        // where the LLM repeatedly asks for non-existent files.
        let mut req = request;
        let mut ctx_rounds = 0u32;
        let mut previously_requested: Vec<String> = Vec::new();
        let response = loop {
            let result = match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(req.clone())).await {
                Ok(r) => r?,
                Err(_) => return Err(RuntimeError::Timeout("LLM call timed out after 5 minutes".into())),
            };
            match result {
                BridgeCallResult::Success(resp) => {
                    self.record_call(node_idx, &resp, LLMCallType::Implement);
                    break resp;
                }
                BridgeCallResult::NeedsContext { requests, raw_response } => {
                    ctx_rounds += 1;
                    self.record_call(node_idx, &raw_response, LLMCallType::Implement);

                    // Hard circuit breaker: max 3 rounds
                    if ctx_rounds > 3 {
                        tracing::warn!(node = node_idx, "Context request limit exceeded — forcing LLM to proceed");
                        let content_clone = raw_response.content.clone();
                        req.messages.push(Message::assistant(content_clone));
                        req.messages.push(Message::user(
                            "This is all the context available. Write the code with what you have. Do not request more context.".to_string()
                        ));
                        // One final attempt with no more context requests honored
                        if let Ok(Ok(BridgeCallResult::Success(final_resp))) =
                            tokio::time::timeout(Duration::from_secs(300), self.bridge.call(req.clone())).await
                        {
                            self.record_call(node_idx, &final_resp, LLMCallType::Implement);
                            break final_resp;
                        }
                        break raw_response;
                    }

                    // Detect hallucination loop: if the LLM is re-requesting
                    // the same paths it already asked for, break the cycle
                    let current_paths: Vec<String> = requests.iter().map(|r| {
                        match &r.kind {
                            kairo_llm::response::ContextRequestKind::File { path, .. } => path.clone(),
                            kairo_llm::response::ContextRequestKind::Symbol { name } => name.clone(),
                        }
                    }).collect();

                    let repeated = current_paths.iter()
                        .filter(|p| previously_requested.contains(p))
                        .count();

                    if repeated > 0 && repeated == current_paths.len() {
                        tracing::warn!(
                            node = node_idx,
                            repeated_count = repeated,
                            "LLM re-requesting same context — breaking hallucination loop"
                        );
                        let content_clone2 = raw_response.content.clone();
                        req.messages.push(Message::assistant(content_clone2));
                        req.messages.push(Message::user(
                            "The files you requested do not exist. Proceed with the code using the context already provided. Do not request more context.".to_string()
                        ));
                        if let Ok(Ok(BridgeCallResult::Success(final_resp))) =
                            tokio::time::timeout(Duration::from_secs(300), self.bridge.call(req.clone())).await
                        {
                            self.record_call(node_idx, &final_resp, LLMCallType::Implement);
                            break final_resp;
                        }
                        break raw_response;
                    }

                    previously_requested.extend(current_paths);
                    let injection = self.fulfill_context_requests(&requests);
                    req.messages.push(Message::assistant(raw_response.content));
                    req.messages.push(Message::user(injection));
                }
                BridgeCallResult::NeedsDecomposition(resp) => {
                    self.record_call(node_idx, &resp, LLMCallType::Implement);
                    tracing::info!(node = node_idx, "Node needs decomposition");
                    return self.decompose_node(node_idx).await;
                }
            }
        };

        let class = classify_response(&response.content, LLMCallType::Implement, None);
        self.record_compliance(class.class);

        match class.class {
            ResponseClass::Implementation => {
                match self.apply_implementation(node_idx, &response.content, track_id).await {
                    Ok(()) => {
                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::AwaitingVerification;
                        self.session_manager.record_good_response(node_idx);
                        Ok(NodeOutcome::Implemented)
                    }
                    Err(RuntimeError::NoImplementation) => {
                        // FIX #2: LLM wrote an explanation instead of code.
                        // Re-call with stronger enforcement demanding actual code.
                        tracing::warn!(
                            node = node_idx,
                            "LLM response classified as Implementation but contained no code — retrying with enforcement"
                        );
                        self.session_manager.record_bad_response(node_idx);

                        let strong = tpl.render(1.0);
                        let no_code_prompt = format!(
                            "{strong}\n\nYour previous response contained no code. \
                             You MUST provide a code implementation using SEARCH/REPLACE blocks \
                             or complete file content with `// File:` markers. \
                             Do NOT explain — write the actual code."
                        );
                        let retry_req = call::assemble_call(
                            LLMCallType::Implement, &title, &spec,
                            &ctx, "", &no_code_prompt, true, &model, max_out,
                        );
                        match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(retry_req)).await {
                            Ok(Ok(BridgeCallResult::Success(retry_resp))) => {
                                self.record_call(node_idx, &retry_resp, LLMCallType::Implement);
                                match self.apply_implementation(node_idx, &retry_resp.content, track_id).await {
                                    Ok(()) => {
                                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::AwaitingVerification;
                                        Ok(NodeOutcome::Implemented)
                                    }
                                    Err(RuntimeError::NoImplementation) => {
                                        // Second attempt also had no code — mark as failed.
                                        tracing::error!(
                                            node = node_idx,
                                            "Second attempt also contained no code — marking node failed"
                                        );
                                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                                        Ok(NodeOutcome::NoCode)
                                    }
                                    Err(e) => {
                                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                                        Ok(NodeOutcome::Failed(format!("Edit after no-code retry: {e}")))
                                    }
                                }
                            }
                            _ => {
                                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                                Ok(NodeOutcome::NoCode)
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(node = node_idx, error = %e, "Edit application failed");
                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                        self.session_manager.record_bad_response(node_idx);
                        Ok(NodeOutcome::Failed(format!("Edit failed: {e}")))
                    }
                }
            }
            ResponseClass::PlaceholderDetected => {
                tracing::warn!(node = node_idx, "Placeholder detected, escalating enforcement");
                self.session_manager.record_bad_response(node_idx);

                // Retry with max enforcement
                let strong = tpl.render(1.0);
                let retry_req = call::assemble_call(
                    LLMCallType::Implement, &title, &spec,
                    &ctx, "", &strong, true, &model, max_out,
                );
                match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(retry_req)).await {
                    Ok(Ok(BridgeCallResult::Success(resp))) => {
                        self.record_call(node_idx, &resp, LLMCallType::Implement);
                        let rc = classify_response(&resp.content, LLMCallType::Implement, None);
                        self.record_compliance(rc.class);
                        if rc.class == ResponseClass::Implementation {
                            match self.apply_implementation(node_idx, &resp.content, track_id).await {
                                Ok(()) => {
                                    self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::AwaitingVerification;
                                    return Ok(NodeOutcome::Implemented);
                                }
                                Err(e) => {
                                    self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                                    return Ok(NodeOutcome::Failed(format!("Edit after retry: {e}")));
                                }
                            }
                        }
                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                        Ok(NodeOutcome::Failed(format!("Still {:?} after escalation", rc.class)))
                    }
                    Ok(Err(e)) => return Err(RuntimeError::Bridge(e)),
                    Err(_) => return Err(RuntimeError::Timeout("LLM call timed out after 5 minutes".into())),
                    _ => {
                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                        Ok(NodeOutcome::Failed("Retry produced non-success".to_string()))
                    }
                }
            }
            other => {
                self.session_manager.record_bad_response(node_idx);
                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                Ok(NodeOutcome::Failed(format!("Unexpected response class: {other:?}")))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. verify_node
// ---------------------------------------------------------------------------

impl Runtime {
    /// Verify a node via L1 (deterministic) and optionally L2 (LLM audit).
    pub async fn verify_node(
        &mut self,
        node_idx: u32,
    ) -> Result<VerifyOutcome, RuntimeError> {
        let node = self.arena.get(node_idx)
            .ok_or(RuntimeError::NodeNotFound(node_idx))?;
        let priority = node.priority;
        let title = node.title.to_string();
        let spec = self.arena.get_spec(node_idx).unwrap_or("").to_string();

        tracing::info!(node = node_idx, title = %title, "Verifying node");

        // --- L1: Deterministic -----------------------------------------------
        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::VerifyingDeterministic;

        if self.fingerprint.can_verify_deterministically() {
            let verifier = DeterministicVerifier::default();
            let det = verifier
                .run(&self.fingerprint, &self.config.project_root, true)
                .await
                .map_err(|e| RuntimeError::Verification(e.to_string()))?;

            if !det.passed() {
                let details: Vec<String> = det
                    .failed_steps()
                    .map(|s| format!("{}: {}", s.step_name, s.stderr_summary()))
                    .collect();
                let details_str = details.join("\n");
                tracing::warn!(node = node_idx, "L1 failed");
                self.arena.get_mut(node_idx).expect("node must exist").det_verdict =
                    crate::arena::node::DeterministicVerdict::Fail;
                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                return Ok(VerifyOutcome::DeterministicFail { details: details_str });
            }
            self.arena.get_mut(node_idx).expect("node must exist").det_verdict =
                crate::arena::node::DeterministicVerdict::Pass;
        } else {
            self.arena.get_mut(node_idx).expect("node must exist").det_verdict =
                crate::arena::node::DeterministicVerdict::Unavailable;
        }

        // --- L2: LLM Audit (if required) ------------------------------------
        let runner = VerificationRunner::new(self.config.cost_mode);
        if runner.should_run_l2(priority) {
            self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::VerifyingAudit;
            tracing::info!(node = node_idx, "Running L2 audit");

            let impl_content = self.build_node_context(node_idx);
            let (atpl, aint) = self.select_enforcement(LLMCallType::Audit, priority);
            let aenf = atpl.render(aint);
            let model = self.active_model();
            let max_out = self.active_max_output();

            let audit_req = call::assemble_call(
                LLMCallType::Audit, &title, &spec,
                &impl_content, "", &aenf, true, &model, max_out,
            );

            match tokio::time::timeout(Duration::from_secs(300), self.bridge.call_audit(audit_req)).await {
                Ok(Ok(BridgeCallResult::Success(resp))) => {
                    self.record_call(node_idx, &resp, LLMCallType::Verify);
                    let ac = classify_response(&resp.content, LLMCallType::Audit, None);
                    self.record_compliance(ac.class);

                    if ac.class == ResponseClass::VerificationFail {
                        let issues = LlmAuditor::parse_issues(&resp.content);
                        self.arena.get_mut(node_idx).expect("node must exist").llm_verdict =
                            crate::arena::node::LLMVerdict::Fail;
                        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                        return Ok(VerifyOutcome::AuditFail { issues });
                    }
                    self.arena.get_mut(node_idx).expect("node must exist").llm_verdict =
                        crate::arena::node::LLMVerdict::Pass;
                }
                Ok(Err(e)) => return Err(RuntimeError::Bridge(e)),
                Err(_) => return Err(RuntimeError::Timeout("LLM audit timed out after 5 minutes".into())),
                _ => {
                    self.arena.get_mut(node_idx).expect("node must exist").llm_verdict =
                        crate::arena::node::LLMVerdict::Skipped;
                }
            }
        } else {
            self.arena.get_mut(node_idx).expect("node must exist").llm_verdict =
                crate::arena::node::LLMVerdict::Skipped;
        }

        // --- All passed: mark complete ----------------------------------------
        self.arena.mark_complete(node_idx);

        // Discard file snapshots (no longer needed for rollback)
        let file_hashes: Vec<u64> = self.arena.get(node_idx).expect("node must exist").impl_files.iter().copied().collect();
        for fh in file_hashes {
            if let Some(ps) = self.arena.resolve_file_path(fh) {
                self.snapshots.discard(Path::new(ps));
            }
        }
        Ok(VerifyOutcome::Pass)
    }
}

// ---------------------------------------------------------------------------
// 4. fix_node
// ---------------------------------------------------------------------------

impl Runtime {
    /// Fix issues in a node. Continues the existing LLM session and calls Fix.
    ///
    /// `track_id` identifies the parallel track (0 in sequential mode).
    pub async fn fix_node(
        &mut self,
        node_idx: u32,
        issues: &str,
        track_id: TrackId,
    ) -> Result<(), RuntimeError> {
        let node = self.arena.get(node_idx)
            .ok_or(RuntimeError::NodeNotFound(node_idx))?;
        let title = node.title.to_string();
        let priority = node.priority;
        let spec = self.arena.get_spec(node_idx).unwrap_or("").to_string();

        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::Fixing;
        self.arena.advance_step()?;
        tracing::info!(node = node_idx, title = %title, "Fixing node");

        let (tpl, intensity) = self.select_enforcement(LLMCallType::Fix, priority);
        let enforcement = tpl.render(intensity);

        let model = self.active_model();
        let max_out = self.active_max_output();
        let fix_spec = format!(
            "{spec}\n\nVerification found these issues:\n{issues}\n\nFix all of them."
        );
        let fix_ctx = self.build_node_context(node_idx);
        let context_fp = xxhash_rust::xxh3::xxh3_64(fix_ctx.as_bytes());

        let (sid, is_new) = self.session_manager.get_or_create_session(
            node_idx, context_fp, LLMCallType::Fix,
        );

        let mut req = call::assemble_call(
            LLMCallType::Fix, &title, &fix_spec,
            "", "", &enforcement, is_new, &model, max_out,
        );

        // SESSION REPLAY for fix: continue with prior conversation so LLM
        // has the implementation context + error details.
        if !is_new {
            if let Some(session) = self.session_manager.get_session(sid) {
                if !session.messages.is_empty() {
                    let new_msgs = std::mem::take(&mut req.messages);
                    req.messages = session.messages.clone();
                    req.messages.extend(new_msgs);
                }
            }
        }

        match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(req)).await {
            Ok(Ok(BridgeCallResult::Success(resp))) => {
                self.record_call(node_idx, &resp, LLMCallType::Fix);
                let fc = classify_response(&resp.content, LLMCallType::Fix, None);
                self.record_compliance(fc.class);
                if fc.class == ResponseClass::Implementation || fc.class == ResponseClass::Plan {
                    // In fix context, NoImplementation is tolerable — treat as
                    // still needing fix rather than hard error.
                    match self.apply_implementation(node_idx, &resp.content, track_id).await {
                        Ok(()) => {}
                        Err(RuntimeError::NoImplementation) => {
                            tracing::warn!(node = node_idx, "Fix response had no code");
                            self.session_manager.record_bad_response(node_idx);
                            self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                            self.arena.get_mut(node_idx).expect("node must exist").retry_count += 1;
                            return Ok(());
                        }
                        Err(e) => {
                            return Err(RuntimeError::EditFailed(format!("fix edit: {e}")));
                        }
                    }
                    self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::AwaitingVerification;
                    self.session_manager.record_good_response(node_idx);
                } else {
                    self.session_manager.record_bad_response(node_idx);
                    self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                    self.arena.get_mut(node_idx).expect("node must exist").retry_count += 1;
                }
            }
            Ok(Err(e)) => return Err(RuntimeError::Bridge(e)),
            Err(_) => return Err(RuntimeError::Timeout("LLM fix call timed out after 5 minutes".into())),
            _ => {
                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                self.arena.get_mut(node_idx).expect("node must exist").retry_count += 1;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 5. run_loop — dispatch to sequential or parallel
// ---------------------------------------------------------------------------

impl Runtime {
    /// The main execution loop. Dispatches to sequential (single-track) or
    /// parallel (multi-track) based on `config.max_parallel_tracks`.
    pub async fn run_loop(&mut self) -> Result<TaskCompletionSummary, RuntimeError> {
        if self.config.max_parallel_tracks <= 1 {
            return self.run_loop_sequential().await;
        }
        self.run_loop_parallel().await
    }

    /// Sequential single-track loop — the original behavior with zero overhead.
    async fn run_loop_sequential(&mut self) -> Result<TaskCompletionSummary, RuntimeError> {
        tracing::info!("Starting execution loop (sequential, 1 track)");

        loop {
            if self.can_terminate() {
                tracing::info!("All itch bits cleared -- task complete");
                break;
            }
            if self.is_budget_exhausted() {
                tracing::warn!("Token budget exhausted");
                return Err(RuntimeError::BudgetExhausted);
            }

            let node_idx = match self.pick_next_node() {
                Some(idx) => idx,
                None => {
                    let st = self.status();
                    if st.failed > 0 && st.pending == 0 && st.active == 0 && st.fix_needed == 0 {
                        return Err(RuntimeError::AllNodesFailed { failed: st.failed });
                    }
                    tracing::warn!("No actionable nodes found");
                    break;
                }
            };

            let node_status = self.arena.get(node_idx).expect("node must exist").status;
            match node_status {
                NodeStatus::Pending | NodeStatus::Ready => {
                    let outcome = self.execute_node(node_idx, 0).await?;
                    match outcome {
                        NodeOutcome::Implemented => {
                            let vr = self.verify_node(node_idx).await?;
                            match vr {
                                VerifyOutcome::Pass => { self.first_pass_count += 1; }
                                VerifyOutcome::Skipped => {
                                    self.arena.mark_complete(node_idx);
                                    self.first_pass_count += 1;
                                }
                                VerifyOutcome::DeterministicFail { details } => {
                                    self.handle_verify_fail(node_idx, &details).await?;
                                }
                                VerifyOutcome::AuditFail { issues } => {
                                    self.handle_verify_fail(node_idx, &issues.join("\n")).await?;
                                }
                            }
                        }
                        NodeOutcome::Decomposed { sub_node_count: _ } => {
                            self.decompose_count += 1;
                        }
                        NodeOutcome::NoCode | NodeOutcome::ContextProvided | NodeOutcome::Failed(_) => {
                            self.handle_exec_fail(node_idx).await?;
                        }
                    }
                }
                NodeStatus::FixNeeded => {
                    let issues = self.get_node_issues(node_idx);
                    self.fix_node(node_idx, &issues, 0).await?;
                    self.fix_count += 1;

                    if self.arena.get(node_idx).expect("node must exist").status == NodeStatus::AwaitingVerification {
                        let vr = self.verify_node(node_idx).await?;
                        match vr {
                            VerifyOutcome::Pass | VerifyOutcome::Skipped => {}
                            VerifyOutcome::DeterministicFail { details } => {
                                self.handle_verify_fail(node_idx, &details).await?;
                            }
                            VerifyOutcome::AuditFail { issues } => {
                                self.handle_verify_fail(node_idx, &issues.join("\n")).await?;
                            }
                        }
                    }
                }
                NodeStatus::AwaitingVerification => {
                    let vr = self.verify_node(node_idx).await?;
                    if let VerifyOutcome::DeterministicFail { details } = vr {
                        self.handle_verify_fail(node_idx, &details).await?;
                    } else if let VerifyOutcome::AuditFail { issues } = vr {
                        self.handle_verify_fail(node_idx, &issues.join("\n")).await?;
                    }
                }
                _ => {
                    tracing::warn!(node = node_idx, status = ?node_status, "Unexpected status, skipping");
                }
            }

            self.maybe_checkpoint()?;
        }

        Ok(self.completion_summary())
    }

    /// Parallel multi-track loop — round-robin processing of all active tracks.
    ///
    /// Each iteration:
    /// 1. Fill idle tracks with pending nodes (via `scheduler.assign_next_pending`).
    /// 2. Step each active track one phase forward (Implementing → Verifying → done).
    /// 3. Advance the verification queue (serialized deterministic checks).
    /// 4. Check termination conditions.
    ///
    /// This avoids spawning tokio tasks (which would require `&mut self` to be
    /// shared across tasks). Instead, each track gets one async step per iteration
    /// of the outer loop. LLM calls within each step are `await`ed inline; the
    /// parallelism comes from interleaving *independent* node work across tracks
    /// rather than running LLM calls concurrently (which is limited by the API
    /// rate limit anyway).
    async fn run_loop_parallel(&mut self) -> Result<TaskCompletionSummary, RuntimeError> {
        let max_tracks = self.scheduler.max_tracks;
        tracing::info!(tracks = max_tracks, "Starting execution loop (parallel)");
        let mut consecutive_no_progress: u32 = 0;

        loop {
            // ----- Termination checks ------------------------------------------
            if self.can_terminate() {
                tracing::info!("All itch bits cleared -- task complete");
                break;
            }
            if self.is_budget_exhausted() {
                tracing::warn!("Token budget exhausted");
                // Release all track locks before returning
                for tid in 0..max_tracks {
                    if !self.scheduler.track(tid).is_idle() {
                        self.scheduler.release_track(tid, &self.file_locks);
                    }
                }
                return Err(RuntimeError::BudgetExhausted);
            }

            // ----- Phase 1: Fill idle tracks with pending nodes ----------------
            let mut assigned_any = false;
            loop {
                if self.scheduler.find_idle_track().is_none() {
                    break;
                }
                if let Some((track_id, node_idx)) =
                    self.scheduler.assign_next_pending(&mut self.arena, &self.file_locks)
                {
                    tracing::info!(
                        track = track_id,
                        node = node_idx,
                        title = %self.arena.get(node_idx).expect("node must exist").title,
                        "Assigned node to track"
                    );
                    // Acquire file locks for this track's node
                    self.acquire_node_file_locks(node_idx, track_id);
                    assigned_any = true;
                } else {
                    // No more assignable nodes — also try fix-needed / awaiting nodes
                    break;
                }
            }

            // Also assign fix-needed and awaiting-verification nodes to idle tracks
            if self.scheduler.find_idle_track().is_some() {
                self.assign_actionable_to_idle_tracks();
            }

            // ----- Phase 2: Step each active track one phase forward -----------
            let mut any_progress = false;
            for tid in 0..max_tracks {
                let track_state = self.scheduler.track(tid).state.clone();
                if self.scheduler.track(tid).is_idle() {
                    continue;
                }

                let node_idx = self.scheduler.track(tid).node_id;

                match track_state {
                    crate::parallel::TrackState::Implementing => {
                        any_progress = true;
                        let outcome = self.execute_node(node_idx, tid).await?;
                        match outcome {
                            NodeOutcome::Implemented => {
                                // Enqueue for verification (serialized deterministic checks)
                                self.scheduler.enqueue_verification(tid);
                                tracing::debug!(
                                    track = tid, node = node_idx,
                                    "Implementation done, enqueued for verification"
                                );
                            }
                            NodeOutcome::Decomposed { sub_node_count } => {
                                self.decompose_count += 1;
                                tracing::info!(
                                    track = tid, node = node_idx,
                                    sub_nodes = sub_node_count,
                                    "Node decomposed"
                                );
                                self.scheduler.release_track(tid, &self.file_locks);
                            }
                            NodeOutcome::NoCode | NodeOutcome::ContextProvided | NodeOutcome::Failed(_) => {
                                self.handle_exec_fail(node_idx).await?;
                                if self.arena.get(node_idx).expect("node must exist").status == NodeStatus::Failed {
                                    self.scheduler.release_track(tid, &self.file_locks);
                                } else {
                                    // Node can retry — transition to Fixing
                                    self.scheduler.track_mut(tid).state =
                                        crate::parallel::TrackState::Fixing;
                                }
                            }
                        }
                    }

                    crate::parallel::TrackState::Verifying => {
                        // Only the active verifier runs deterministic checks
                        if self.scheduler.verification_queue.is_active_verifier(tid) {
                            any_progress = true;
                            let vr = self.verify_node(node_idx).await?;
                            self.scheduler.verification_queue.finish_current();

                            match vr {
                                VerifyOutcome::Pass => {
                                    self.first_pass_count += 1;
                                    tracing::info!(
                                        track = tid, node = node_idx,
                                        "Verification passed"
                                    );
                                    self.scheduler.release_track(tid, &self.file_locks);
                                }
                                VerifyOutcome::Skipped => {
                                    self.arena.mark_complete(node_idx);
                                    self.first_pass_count += 1;
                                    self.scheduler.release_track(tid, &self.file_locks);
                                }
                                VerifyOutcome::DeterministicFail { details } => {
                                    tracing::warn!(
                                        track = tid, node = node_idx,
                                        "Deterministic verification failed"
                                    );
                                    self.handle_verify_fail(node_idx, &details).await?;
                                    if self.arena.get(node_idx).expect("node must exist").status == NodeStatus::Failed {
                                        self.scheduler.release_track(tid, &self.file_locks);
                                    } else {
                                        // Transition to Fixing
                                        self.scheduler.track_mut(tid).state =
                                            crate::parallel::TrackState::Fixing;
                                    }
                                }
                                VerifyOutcome::AuditFail { issues } => {
                                    tracing::warn!(
                                        track = tid, node = node_idx,
                                        "Audit verification failed"
                                    );
                                    let joined = issues.join("\n");
                                    self.handle_verify_fail(node_idx, &joined).await?;
                                    if self.arena.get(node_idx).expect("node must exist").status == NodeStatus::Failed {
                                        self.scheduler.release_track(tid, &self.file_locks);
                                    } else {
                                        self.scheduler.track_mut(tid).state =
                                            crate::parallel::TrackState::Fixing;
                                    }
                                }
                            }
                        }
                        // If not the active verifier, the track waits (no progress this iteration)
                    }

                    crate::parallel::TrackState::Fixing => {
                        any_progress = true;
                        let issues = self.get_node_issues(node_idx);
                        self.fix_node(node_idx, &issues, tid).await?;
                        self.fix_count += 1;

                        if self.arena.get(node_idx).expect("node must exist").status == NodeStatus::AwaitingVerification {
                            // Re-enqueue for verification
                            self.scheduler.enqueue_verification(tid);
                        } else if !self.arena.get(node_idx).expect("node must exist").can_retry() {
                            // Max retries exhausted
                            self.arena.mark_failed(node_idx);
                            self.scheduler.release_track(tid, &self.file_locks);
                        } else {
                            // Still FixNeeded but can retry — stay in Fixing state
                        }
                    }

                    crate::parallel::TrackState::Blocked(ref _path) => {
                        // Check if the blocking lock has been released
                        let node_files = self.get_node_file_paths(node_idx);
                        let all_available = node_files.iter().all(|p| {
                            match self.file_locks.held_by(p) {
                                None => true,
                                Some(holder) => holder == tid,
                            }
                        });

                        if all_available {
                            // Unblock: re-acquire locks and resume implementing
                            self.acquire_node_file_locks(node_idx, tid);
                            self.scheduler.track_mut(tid).state =
                                crate::parallel::TrackState::Implementing;
                            any_progress = true;
                        }
                        // Otherwise remain blocked — no progress this iteration
                    }

                    crate::parallel::TrackState::Idle => {
                        // Already handled above, but be defensive
                    }
                }
            }

            // ----- Phase 3: Advance verification queue -------------------------
            // Try to promote the next waiter to active verifier
            if let Some((next_tid, next_node)) =
                self.scheduler.verification_queue.try_start_next()
            {
                tracing::debug!(
                    track = next_tid, node = next_node,
                    "Started verification for track"
                );
            }

            // ----- Phase 4: Check if we're stuck or done -----------------------
            // Track consecutive iterations with no progress for deadlock detection.
            if any_progress || assigned_any {
                consecutive_no_progress = 0;
            } else {
                consecutive_no_progress += 1;
                if consecutive_no_progress >= 100 {
                    self.file_locks.clear();
                    let st = self.status();
                    tracing::error!(
                        iterations = consecutive_no_progress,
                        pending = st.pending,
                        active = st.active,
                        fix_needed = st.fix_needed,
                        "Deadlock detected: 100 consecutive iterations with no progress"
                    );
                    return Err(RuntimeError::AllNodesFailed { failed: st.failed + st.pending });
                }
            }

            let active_tracks = self.scheduler.active_track_count();
            let st = self.status();

            if active_tracks == 0 && !assigned_any {
                // No tracks active and we couldn't assign anything
                if st.pending == 0 && st.fix_needed == 0 {
                    if st.failed > 0 && st.completed == 0 {
                        // Release any remaining locks
                        self.file_locks.clear();
                        return Err(RuntimeError::AllNodesFailed { failed: st.failed });
                    }
                    // All done (some completed, some failed, none pending)
                    break;
                }
                // There are pending nodes but none could be assigned (dependency-blocked)
                // or fix-needed nodes that are file-conflicting. Check for true deadlock.
                if !any_progress {
                    // If we have pending nodes whose dependencies are all resolved
                    // but they conflict with active tracks, we must wait. But if
                    // there are NO active tracks, it's a genuine stall.
                    if active_tracks == 0 {
                        tracing::warn!(
                            pending = st.pending, fix_needed = st.fix_needed,
                            "No actionable nodes and no active tracks -- stall detected"
                        );
                        break;
                    }
                }
            }

            self.maybe_checkpoint()?;

            // Yield to the executor between iterations to avoid starving
            // other tasks (e.g., signal handlers, timeouts).
            tokio::task::yield_now().await;
        }

        // Cleanup: release all remaining file locks
        self.file_locks.clear();
        Ok(self.completion_summary())
    }

    /// Acquire file locks for all files associated with a node.
    fn acquire_node_file_locks(&self, node_idx: u32, track_id: u8) {
        for &fh in &self.arena.get(node_idx).expect("node must exist").impl_files {
            if let Some(path_str) = self.arena.resolve_file_path(fh) {
                let path = Path::new(path_str);
                let _ = self.file_locks.acquire(path, track_id);
            }
        }
    }

    /// Get resolved file paths for a node.
    fn get_node_file_paths(&self, node_idx: u32) -> Vec<PathBuf> {
        match self.arena.get(node_idx) {
            Some(node) => node
                .impl_files
                .iter()
                .filter_map(|&fh| {
                    self.arena
                        .resolve_file_path(fh)
                        .map(PathBuf::from)
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Assign fix-needed and awaiting-verification nodes to idle tracks.
    ///
    /// Called after `assign_next_pending` has tried (and possibly exhausted)
    /// pending queue nodes. This handles the two other actionable states
    /// that the sequential loop's `pick_next_node` covers.
    fn assign_actionable_to_idle_tracks(&mut self) {
        // Collect fix-needed nodes that can retry and are not already on a track
        let fix_nodes: Vec<u32> = self
            .arena
            .nodes_needing_fix()
            .into_iter()
            .filter(|&idx| {
                self.arena.get(idx).map_or(false, |n| n.can_retry())
                    && !self.is_node_on_any_track(idx)
            })
            .collect();

        for node_idx in fix_nodes {
            if let Some(tid) = self.scheduler.find_idle_track() {
                // Check for file conflicts with active tracks
                let conflicts = self.node_conflicts_with_active_tracks(node_idx, tid);
                if !conflicts {
                    self.scheduler.track_mut(tid).assign(node_idx, 0);
                    self.scheduler.track_mut(tid).state =
                        crate::parallel::TrackState::Fixing;
                    self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::Fixing;
                    self.acquire_node_file_locks(node_idx, tid);
                    tracing::info!(
                        track = tid, node = node_idx,
                        "Assigned fix-needed node to track"
                    );
                }
            } else {
                break; // No more idle tracks
            }
        }

        // Collect awaiting-verification nodes not already on a track
        let verify_nodes: Vec<u32> = self
            .arena
            .nodes_by_status(NodeStatus::AwaitingVerification)
            .into_iter()
            .filter(|&idx| !self.is_node_on_any_track(idx))
            .collect();

        for node_idx in verify_nodes {
            if let Some(tid) = self.scheduler.find_idle_track() {
                self.scheduler.track_mut(tid).assign(node_idx, 0);
                self.scheduler.enqueue_verification(tid);
                self.acquire_node_file_locks(node_idx, tid);
                tracing::info!(
                    track = tid, node = node_idx,
                    "Assigned awaiting-verification node to track"
                );
            } else {
                break;
            }
        }
    }

    /// Check whether a node is already assigned to any active track.
    fn is_node_on_any_track(&self, node_idx: u32) -> bool {
        self.scheduler
            .tracks
            .iter()
            .any(|t| !t.is_idle() && t.node_id == node_idx)
    }

    /// Check whether a node's files conflict with any other active track
    /// (excluding the specified track_id which is about to be assigned).
    fn node_conflicts_with_active_tracks(&self, node_idx: u32, exclude_track: u8) -> bool {
        for track in &self.scheduler.tracks {
            if track.is_idle() || track.id == exclude_track {
                continue;
            }
            if self.arena.nodes_share_files(node_idx, track.node_id) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// 6. assemble_input_packet
// ---------------------------------------------------------------------------

impl Runtime {
    /// Build the 32-slot InputPacket from live runtime state.
    pub fn assemble_input_packet(&self, active_node: u32) -> InputPacket {
        let mut packet = InputPacket::new();
        let status = self.arena.status_summary();

        // Slots 0-5: Execution graph summary
        {
            let s = packet.slot_mut(slots::EXECUTION_GRAPH.start);
            encode_count(s, status.total as u32, 256);
            encode_count(&mut s[4..], status.completed as u32, status.total.max(1) as u32);
            encode_count(&mut s[8..], status.pending as u32, status.total.max(1) as u32);
            encode_count(&mut s[12..], status.active as u32, 16);
            encode_count(&mut s[16..], status.failed as u32, status.total.max(1) as u32);
            encode_float(s, status.progress() as f32, 20);
        }

        // Slots 6-9: Active node context
        if active_node > 0 && (active_node as usize) < self.arena.nodes.len() {
            if let Some(node) = self.arena.get(active_node) {
                let s = packet.slot_mut(slots::ACTIVE_NODE.start);
                encode_priority(s, node.priority, 0);
                encode_node_status(s, node.status, 3);
                encode_count(&mut s[15..], node.retry_count as u32, node.max_retries as u32);
                encode_count(&mut s[19..], node.llm_calls_spent as u32, 20);
                encode_count(&mut s[23..], node.dependencies.len() as u32, 16);
                let resolved = node.dependencies.iter()
                    .filter(|&&d| self.arena.get(d).map_or(false, |n| n.status == NodeStatus::Verified))
                    .count();
                encode_count(&mut s[27..], resolved as u32, node.dependencies.len().max(1) as u32);
            }
        }

        // Slots 25-26: Itch state
        {
            let (active_count, total_nodes) = self.arena.itch_stats();
            let s = packet.slot_mut(slots::ITCH_STATE.start);
            encode_count(s, active_count, total_nodes.max(1) as u32);
            encode_count(&mut s[4..], self.arena.failed_nodes().len() as u32, total_nodes.max(1) as u32);
        }

        // Slots 27-28: Session state
        {
            let s = packet.slot_mut(slots::SESSION_STATE.start);
            encode_count(s, self.session_manager.active_count() as u32, 32);
            encode_count(&mut s[4..], (self.token_tracker.total_calls.min(u32::MAX as u64)) as u32, 1000);
        }

        // Slot 29: Cost state
        {
            let s = packet.slot_mut(slots::COST_STATE.start);
            let bl = self.token_tracker.cost_limit_microdollars;
            if bl > 0 {
                encode_float(s, self.token_tracker.cost_budget_remaining().unwrap_or(0) as f32 / bl as f32, 0);
            }
            let cm = match self.config.cost_mode {
                CostMode::Thorough => 0, CostMode::Balanced => 1, CostMode::Efficient => 2,
            };
            input_assembly::encode_onehot(s, cm, 3, 1);
            encode_bool(s, self.token_tracker.is_budget_exhausted(), 4);
        }

        packet
    }
}

// ---------------------------------------------------------------------------
// Lock-on-write helpers (Critical Fix #1)
// ---------------------------------------------------------------------------

impl Runtime {
    /// Whether the runtime is in parallel mode (more than one track).
    fn is_parallel(&self) -> bool {
        self.config.max_parallel_tracks > 1
    }

    /// Write a file with lock acquisition for the given track.
    ///
    /// In sequential mode (`max_parallel_tracks <= 1`), writes directly without
    /// locking overhead. In parallel mode, acquires the file lock via the
    /// `FileLockTable`'s acquire, retrying every 500ms for up to
    /// 5 seconds. If still blocked, returns `RuntimeError::FileLockedReRead`
    /// so the caller can re-read and retry.
    ///
    /// On success the lock is *kept* (not released) so the track retains
    /// exclusive access until `release_track` or `release_all` is called.
    /// The file path is also registered in the node's `impl_files`.
    async fn write_file_locked(
        &mut self,
        path: &Path,
        content: &str,
        node_idx: u32,
        track_id: TrackId,
    ) -> Result<(), RuntimeError> {
        if !self.is_parallel() {
            // Sequential mode — write directly.
            crate::tools::filesystem::write_file(path, content, &self.config.project_root)
                .map_err(|e| RuntimeError::EditFailed(format!("write {}: {e}", path.display())))?;
        } else {
            // Parallel mode — acquire lock with async retry.
            let start = std::time::Instant::now();
            let max_wait = Duration::from_secs(5);
            let retry_interval = Duration::from_millis(500);

            loop {
                match self.file_locks.acquire(path, track_id) {
                    crate::tools::file_locks::LockResult::Acquired
                    | crate::tools::file_locks::LockResult::AlreadyOwned => {
                        // We hold the lock — write the file.
                        crate::tools::filesystem::write_file(path, content, &self.config.project_root)
                            .map_err(|e| RuntimeError::EditFailed(
                                format!("write {}: {e}", path.display())
                            ))?;
                        // Keep the lock (don't release) so the track retains access.
                        break;
                    }
                    crate::tools::file_locks::LockResult::Blocked(blocker) => {
                        let elapsed = start.elapsed();
                        if elapsed >= max_wait {
                            return Err(RuntimeError::FileLockedReRead(format!(
                                "{} blocked by track {} after {}ms",
                                path.display(),
                                blocker,
                                elapsed.as_millis()
                            )));
                        }
                        tokio::time::sleep(retry_interval.min(max_wait.saturating_sub(elapsed))).await;
                    }
                }
            }
        }

        // Register the file path in the node's impl_files.
        let path_str = path.to_string_lossy().to_string();
        let fh = self.arena.register_file_path(&path_str);
        if !self.arena.get(node_idx).expect("node must exist").impl_files.contains(&fh) {
            self.arena.get_mut(node_idx).expect("node must exist").impl_files.push(fh);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

impl Runtime {
    /// Call bridge and extract the content string, recording token usage.
    async fn call_and_extract(
        &mut self,
        request: &call::LLMRequest,
        node_idx: u32,
    ) -> Result<String, RuntimeError> {
        let result = match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(request.clone())).await {
            Ok(r) => r?,
            Err(_) => return Err(RuntimeError::Timeout("LLM call timed out after 5 minutes".into())),
        };
        match result {
            BridgeCallResult::Success(r) => {
                self.token_tracker.record_usage(node_idx, r.input_tokens, r.output_tokens);
                Ok(r.content)
            }
            BridgeCallResult::NeedsContext { raw_response: r, .. }
            | BridgeCallResult::NeedsDecomposition(r) => {
                self.token_tracker.record_usage(node_idx, r.input_tokens, r.output_tokens);
                Ok(r.content)
            }
        }
    }

    /// Record an LLM call against a node and the token tracker.
    fn record_call(
        &mut self,
        node_idx: u32,
        resp: &kairo_llm::response::LLMRawResponse,
        call_type: LLMCallType,
    ) {
        self.arena.get_mut(node_idx).expect("node must exist").record_llm_call();
        self.token_tracker.record_usage(node_idx, resp.input_tokens, resp.output_tokens);
        if let Some(session) = self.session_manager.get_session_for_node_mut(node_idx) {
            session.record_usage(resp.input_tokens, resp.output_tokens, call_type);
            // Store the assistant response so future session continuations
            // replay the full conversation history.
            session.messages.push(Message::assistant(resp.content.clone()));
        }
    }

    /// Pick the next node to work on (sequential mode only).
    ///
    /// In parallel mode, node assignment is handled by
    /// `scheduler.assign_next_pending()` and `assign_actionable_to_idle_tracks()`.
    fn pick_next_node(&mut self) -> Option<u32> {
        // 1. Fix-needed nodes that can still retry
        for idx in self.arena.nodes_needing_fix() {
            if self.arena.get(idx).map_or(false, |n| n.can_retry()) {
                return Some(idx);
            }
        }
        // 2. Nodes awaiting verification
        let awaiting = self.arena.nodes_by_status(NodeStatus::AwaitingVerification);
        if let Some(&idx) = awaiting.first() {
            return Some(idx);
        }
        // 3. Pop from pending queue (skip stale entries)
        while let Some(entry) = self.arena.pending_queue.pop() {
            let is_ready = self.arena.get(entry.node_idx)
                .map_or(false, |n| n.status == NodeStatus::Pending)
                && self.arena.are_dependencies_resolved(entry.node_idx);
            if is_ready {
                return Some(entry.node_idx);
            }
        }
        // 4. Safety net: scan for ready nodes
        self.arena.nodes_ready().into_iter().next()
    }

    /// Handle verification failure: retry or mark failed.
    async fn handle_verify_fail(
        &mut self,
        node_idx: u32,
        _issues: &str,
    ) -> Result<(), RuntimeError> {
        let node = self.arena.get(node_idx)
            .ok_or(RuntimeError::NodeNotFound(node_idx))?;
        if node.can_retry() {
            self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
        } else {
            // Restore file snapshots
            let file_hashes: Vec<u64> = self.arena.get(node_idx).expect("node must exist").impl_files.iter().copied().collect();
            for fh in file_hashes {
                if let Some(ps) = self.arena.resolve_file_path(fh) {
                    let _ = self.snapshots.restore_file(Path::new(ps));
                }
            }
            self.arena.mark_failed(node_idx);
        }
        Ok(())
    }

    /// Handle an execution failure.
    async fn handle_exec_fail(
        &mut self,
        node_idx: u32,
    ) -> Result<(), RuntimeError> {
        let can_retry = self.arena.get(node_idx)
            .ok_or(RuntimeError::NodeNotFound(node_idx))?
            .can_retry();
        if !can_retry {
            self.arena.mark_failed(node_idx);
        }
        Ok(())
    }

    /// Apply an implementation response: parse SEARCH/REPLACE blocks and apply edits.
    ///
    /// Returns `Ok(())` only if at least one file was actually written or edited.
    /// Returns `Err(RuntimeError::NoImplementation)` if the response contained no
    /// SEARCH/REPLACE blocks and no file creation markers (explanation-only).
    ///
    /// `track_id` is the parallel track executing this node (0 in sequential mode).
    /// All filesystem writes go through `write_file_locked` to ensure proper
    /// lock acquisition in parallel mode.
    async fn apply_implementation(
        &mut self,
        node_idx: u32,
        content: &str,
        track_id: TrackId,
    ) -> Result<(), RuntimeError> {
        let project_root = self.config.project_root.clone();
        let blocks = crate::tools::search_replace::parse_search_replace_blocks(content);

        if blocks.is_empty() {
            // Check for new-file content
            if let Some(files) = self.extract_new_file_content(content) {
                for (path, file_content) in files {
                    let full = project_root.join(&path);
                    let _ = self.snapshots.snapshot_file(&full);
                    self.write_file_locked(&full, &file_content, node_idx, track_id).await?;
                    tracing::info!(path = %path, "Created new file");
                }
                return Ok(());
            }
            // No files written, no edits — this is an explanation-only response.
            return Err(RuntimeError::NoImplementation);
        }

        // Track whether any edits were actually applied.
        let mut files_written = false;

        // Determine target files
        let impl_files: Vec<String> = self.arena.get(node_idx).expect("node must exist").impl_files.iter()
            .filter_map(|&h| self.arena.resolve_file_path(h).map(|s| s.to_string()))
            .collect();

        if impl_files.len() == 1 {
            let fpath = project_root.join(&impl_files[0]);
            self.apply_edits_to_file(&fpath, &blocks, node_idx, track_id).await?;
            return Ok(());
        }

        // Try each known file
        for f in &impl_files {
            let fpath = project_root.join(f);
            if fpath.exists()
                && self.apply_edits_to_file(&fpath, &blocks, node_idx, track_id).await.is_ok()
            {
                return Ok(());
            }
        }

        // Broad search fallback — check file lock ownership before writing.
        // If the file is locked by another track, skip it to avoid cross-track conflicts.
        for block in &blocks {
            let needle = &block.search[..block.search.len().min(80)];
            if let Ok(matches) = crate::tools::filesystem::search_text(&project_root, needle, 3, &self.config.project_root) {
                if let Some(m) = matches.first() {
                    // Check file lock ownership: skip if locked by another track
                    if self.is_parallel() {
                        if let Some(holder) = self.file_locks.held_by(&m.path) {
                            if holder != track_id {
                                tracing::warn!(
                                    path = %m.path.display(),
                                    holder = holder,
                                    track = track_id,
                                    "Broad search fallback: file locked by another track, skipping"
                                );
                                continue;
                            }
                        }
                    }
                    let _ = self.snapshots.snapshot_file(&m.path);
                    if let Ok(c) = crate::tools::filesystem::read_file(&m.path, &self.config.project_root) {
                        if let Ok((nc, _)) = crate::tools::search_replace::apply_edits(&c, std::slice::from_ref(block)) {
                            self.write_file_locked(&m.path, &nc, node_idx, track_id).await?;
                            files_written = true;
                        }
                    }
                }
            }
        }

        if !files_written && impl_files.is_empty() {
            return Err(RuntimeError::NoImplementation);
        }
        Ok(())
    }

    /// Apply search/replace blocks to a single file.
    ///
    /// All writes go through `write_file_locked` for parallel safety.
    async fn apply_edits_to_file(
        &mut self,
        file_path: &Path,
        blocks: &[crate::tools::search_replace::SearchReplaceBlock],
        node_idx: u32,
        track_id: TrackId,
    ) -> Result<(), RuntimeError> {
        let _ = self.snapshots.snapshot_file(file_path);
        let content = if file_path.exists() {
            crate::tools::filesystem::read_file(file_path, &self.config.project_root)
                .map_err(|e| RuntimeError::EditFailed(format!("read {}: {e}", file_path.display())))?
        } else {
            String::new()
        };
        let (new_content, apps) = crate::tools::search_replace::apply_edits(&content, blocks)
            .map_err(|e| RuntimeError::EditFailed(format!("edit {}: {e}", file_path.display())))?;
        self.write_file_locked(file_path, &new_content, node_idx, track_id).await?;
        tracing::info!(path = %file_path.display(), edits = apps.len(), "Applied edits");
        Ok(())
    }

    /// Build context text by reading files associated with a node.
    fn build_node_context(&self, node_idx: u32) -> String {
        let root = &self.config.project_root;
        let mut parts = Vec::new();
        for &fh in &self.arena.get(node_idx).expect("node must exist").impl_files {
            if let Some(ps) = self.arena.resolve_file_path(fh) {
                let full = root.join(ps);
                if let Ok(c) = crate::tools::filesystem::read_file(&full, &self.config.project_root) {
                    parts.push(format!("--- {ps} ---\n{c}"));
                }
            }
        }
        parts.join("\n\n")
    }

    /// Get issues description for a node that needs fixing.
    fn get_node_issues(&self, node_idx: u32) -> String {
        let node = match self.arena.get(node_idx) {
            Some(n) => n,
            None => return "Node not found.".to_string(),
        };
        let mut parts = Vec::new();
        if node.det_verdict == crate::arena::node::DeterministicVerdict::Fail {
            parts.push("Deterministic verification (build/test/lint) failed.".to_string());
        }
        if node.llm_verdict == crate::arena::node::LLMVerdict::Fail {
            parts.push("LLM audit found issues.".to_string());
        }
        if parts.is_empty() {
            "Previous attempt failed. Review and fix.".to_string()
        } else {
            parts.join("\n")
        }
    }

    /// Decompose a node into sub-nodes via LLM.
    async fn decompose_node(
        &mut self,
        node_idx: u32,
    ) -> Result<NodeOutcome, RuntimeError> {
        let node = self.arena.get(node_idx)
            .ok_or(RuntimeError::NodeNotFound(node_idx))?;
        let title = node.title.to_string();
        let spec = self.arena.get_spec(node_idx).unwrap_or("").to_string();
        let priority = node.priority;

        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::Decomposing;

        let (tpl, int) = self.select_enforcement(LLMCallType::Decompose, priority);
        let enf = tpl.render(int);
        let model = self.active_model();
        let max_out = self.active_max_output();

        let req = call::assemble_call(
            LLMCallType::Decompose, &title, &spec,
            "", "", &enf, true, &model, max_out,
        );
        let decompose_text = match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(req)).await {
            Ok(Ok(BridgeCallResult::Success(r))) => { self.record_call(node_idx, &r, LLMCallType::Decompose); r.content }
            Ok(Err(e)) => return Err(RuntimeError::Bridge(e)),
            Err(_) => return Err(RuntimeError::Timeout("LLM decompose call timed out after 5 minutes".into())),
            _ => {
                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                return Ok(NodeOutcome::Failed("Decomposition failed".into()));
            }
        };

        // JSON pass
        let json_req = call::LLMRequest {
            messages: vec![
                Message::system(call::SYSTEM_PROMPT.to_string()),
                Message::user(format!("Decompose \"{title}\".\n\nSpec: {spec}")),
                Message::assistant(decompose_text.clone()),
                Message::user(call::PLAN_TO_JSON_PROMPT.to_string()),
            ],
            model, temperature: 0.0, max_output_tokens: max_out,
            call_type: LLMCallType::Decompose, stop_sequences: Vec::new(),
        };
        let json_text = match tokio::time::timeout(Duration::from_secs(300), self.bridge.call(json_req)).await {
            Ok(Ok(BridgeCallResult::Success(r))) => { self.record_call(node_idx, &r, LLMCallType::Decompose); r.content }
            Ok(Err(e)) => return Err(RuntimeError::Bridge(e)),
            Err(_) => return Err(RuntimeError::Timeout("LLM decompose JSON timed out after 5 minutes".into())),
            _ => {
                self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::FixNeeded;
                return Ok(NodeOutcome::Failed("Decompose JSON failed".into()));
            }
        };

        let components = plan_parser::parse_plan_json(&json_text)
            .or_else(|_| plan_parser::parse_plan_numbered_list(&decompose_text))
            .map_err(|e| RuntimeError::Planning(format!("Decompose parse: {e}")))?;

        let sub = plan_parser::build_graph_from_plan(&mut self.arena, &components, node_idx)
            .map_err(|e| RuntimeError::Planning(e.to_string()))?;
        // Don't mark the parent as Verified — it was decomposed, not completed.
        // Leave it in Decomposing status and clear its itch bit so the system
        // can terminate once all sub-nodes are done. The sub-nodes carry their
        // own itch bits and will be processed independently.
        self.arena.get_mut(node_idx).expect("node must exist").status = NodeStatus::Decomposing;
        self.arena.itch.clear(node_idx as usize);
        Ok(NodeOutcome::Decomposed { sub_node_count: sub.total_components })
    }

    /// Fulfill LLM context requests by reading files/symbols from the project.
    fn fulfill_context_requests(
        &self,
        requests: &[kairo_llm::response::ContextRequest],
    ) -> String {
        let root = &self.config.project_root;
        let mut fulfilled = Vec::new();
        let mut not_found = Vec::new();

        for req in requests {
            match &req.kind {
                ContextRequestKind::File { path, line_start, line_end } => {
                    let full = root.join(path);
                    match crate::tools::filesystem::read_file(&full, root) {
                        Ok(c) => {
                            let c = match (line_start, line_end) {
                                (Some(s), Some(e)) => c.lines()
                                    .skip((*s as usize).saturating_sub(1))
                                    .take(*e as usize - (*s as usize).saturating_sub(1))
                                    .collect::<Vec<_>>().join("\n"),
                                _ => c,
                            };
                            fulfilled.push((req.clone(), c));
                        }
                        Err(_) => {
                            let dir = full.parent().unwrap_or(root);
                            let alts: Vec<String> = crate::tools::filesystem::list_directory(dir, root)
                                .unwrap_or_default().iter().take(10).map(|e| e.name.clone()).collect();
                            not_found.push((req.clone(), alts));
                        }
                    }
                }
                ContextRequestKind::Symbol { name } => {
                    match crate::tools::filesystem::search_text(root, name, 5, root) {
                        Ok(ms) if !ms.is_empty() => {
                            let c = ms.iter().map(|m| format!("{}:{}: {}", m.path.display(), m.line_number, m.line))
                                .collect::<Vec<_>>().join("\n");
                            fulfilled.push((req.clone(), c));
                        }
                        _ => { not_found.push((req.clone(), Vec::new())); }
                    }
                }
            }
        }
        context_request::format_context_injection(&fulfilled, &not_found)
    }

    /// Extract new file content from a response (file-path + code-block patterns).
    fn extract_new_file_content(&self, response: &str) -> Option<Vec<(String, String)>> {
        let mut files = Vec::new();
        let mut cur_path: Option<String> = None;
        let mut cur_content = String::new();
        let mut in_block = false;

        for line in response.lines() {
            let t = line.trim();

            // File-path markers
            if let Some(p) = t.strip_prefix("// File: ")
                .or_else(|| t.strip_prefix("# File: "))
                .or_else(|| t.strip_prefix("// file: "))
            {
                if let Some(prev) = cur_path.take() {
                    if !cur_content.trim().is_empty() {
                        files.push((prev, cur_content.trim().to_string()));
                    }
                }
                cur_path = Some(p.trim().to_string());
                cur_content.clear();
                continue;
            }

            if t.starts_with("```") {
                if in_block {
                    in_block = false;
                    if let Some(ref p) = cur_path {
                        if !cur_content.trim().is_empty() {
                            files.push((p.clone(), cur_content.trim().to_string()));
                            cur_path = None;
                            cur_content.clear();
                        }
                    }
                } else {
                    in_block = true;
                    // Check for file path in fence
                    let rest = t.trim_start_matches('`');
                    let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                    if parts.len() == 2 && parts[1].contains('/') {
                        if let Some(prev) = cur_path.take() {
                            if !cur_content.trim().is_empty() {
                                files.push((prev, cur_content.trim().to_string()));
                            }
                        }
                        cur_path = Some(parts[1].trim().to_string());
                        cur_content.clear();
                    }
                }
                continue;
            }

            if cur_path.is_some() {
                if !cur_content.is_empty() { cur_content.push('\n'); }
                cur_content.push_str(line);
            }
        }

        if let Some(p) = cur_path {
            if !cur_content.trim().is_empty() {
                files.push((p, cur_content.trim().to_string()));
            }
        }

        if files.is_empty() { None } else { Some(files) }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Display for TaskCompletionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Task complete. {}", self.status)?;
        writeln!(f, "  LLM calls: {}", self.total_llm_calls)?;
        writeln!(f, "  Tokens: {} in / {} out", self.total_input_tokens, self.total_output_tokens)?;
        let cost = self.estimated_cost_microdollars as f64 / 1_000_000.0;
        writeln!(f, "  Estimated cost: ${cost:.2}")?;
        if self.first_pass_success > 0 {
            writeln!(f, "  First-pass success: {}", self.first_pass_success)?;
        }
        if self.required_fixes > 0 {
            writeln!(f, "  Required fixes: {}", self.required_fixes)?;
        }
        if self.required_decomposition > 0 {
            writeln!(f, "  Required decomposition: {}", self.required_decomposition)?;
        }
        Ok(())
    }
}
