//! Command handlers for the KAIRO-X CLI.
//!
//! Each handler corresponds to a top-level subcommand (`init`, `run`,
//! `status`, `resume`). They integrate with kairo-core's [`Runtime`] and
//! subsystems.

use crate::config::{self, CliOverrides};
use crate::display;
use anyhow::{bail, Context, Result};
use kairo_core::fingerprint::detector::fingerprint_project;
use kairo_core::persistence::CheckpointManager;
use kairo_core::runtime::Runtime;
use kairo_core::session::token_tracker::CostMode;
use kairo_llm::providers::ProviderKind;
use std::path::Path;

// ---------------------------------------------------------------------------
// kairo init
// ---------------------------------------------------------------------------

/// Fingerprint the project and create the `.kairo/` directory.
pub async fn handle_init(project_root: &Path) -> Result<()> {
    tracing::info!(path = %project_root.display(), "Initializing KAIRO-X project");

    // 1. Create .kairo/ directory structure.
    let kairo_dir = project_root.join(".kairo");
    let checkpoints_dir = kairo_dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoints_dir).with_context(|| {
        format!(
            "failed to create .kairo directory at {}",
            kairo_dir.display()
        )
    })?;

    // Copy the default config into .kairo/ if not already present.
    let local_config = kairo_dir.join("config.toml");
    if !local_config.exists() {
        let default_toml = include_str!("../../../config/default_config.toml");
        std::fs::write(&local_config, default_toml)
            .context("failed to write .kairo/config.toml")?;
        tracing::info!("Created .kairo/config.toml with defaults");
    }

    // 2. Fingerprint the project.
    let fingerprint = fingerprint_project(project_root);

    // 3. Save fingerprint to .kairo/ for later use.
    let fp_json = serde_json::to_string_pretty(&fingerprint)
        .context("failed to serialize fingerprint")?;
    std::fs::write(kairo_dir.join("fingerprint.json"), fp_json)
        .context("failed to write fingerprint.json")?;

    // 4. Display results.
    display::print_fingerprint(&fingerprint);

    println!(
        "  .kairo/ directory created at {}",
        kairo_dir.display()
    );
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// kairo run
// ---------------------------------------------------------------------------

/// Run a task described by a string or spec file.
#[allow(clippy::too_many_arguments)]
pub async fn handle_run(
    project_root: &Path,
    task: &str,
    cost_mode: Option<CostMode>,
    parallel: Option<u8>,
    budget: Option<u64>,
    resume: bool,
    provider_kind: ProviderKind,
    model: &str,
) -> Result<()> {
    // Resolve the task: either a literal description or a path to a spec file.
    let task_spec = resolve_task(project_root, task)?;
    tracing::info!(task = %task_spec, "Starting task");

    // Ensure .kairo/ exists.
    let kairo_dir = project_root.join(".kairo");
    if !kairo_dir.exists() {
        tracing::info!(".kairo/ not found; running init first");
        handle_init(project_root).await?;
    }

    // Load merged config. Only mark CLI values as explicit if the user
    // actually passed them (Option::Some), so config file values aren't
    // silently overridden by clap defaults.
    let overrides = CliOverrides {
        cost_mode: cost_mode.unwrap_or(CostMode::Balanced),
        cost_mode_explicit: cost_mode.is_some(),
        parallel: parallel.unwrap_or(3),
        parallel_explicit: parallel.is_some(),
        budget,
        provider_kind,
        model: model.to_string(),
    };
    let config = config::load_config(project_root, &overrides)
        .context("failed to load configuration")?;

    // Build the runtime.
    let mut runtime = Runtime::new(config).map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(
        language = %runtime.fingerprint.primary_language,
        cost_mode = %runtime.config.cost_mode,
        parallel = runtime.config.max_parallel_tracks,
        "Runtime initialized"
    );

    // If --resume flag is set, restore from checkpoint.
    if resume {
        restore_from_checkpoint(&mut runtime, project_root)?;
    }

    // Display the fingerprint summary.
    display::print_fingerprint(&runtime.fingerprint);

    println!("  Task: {}", task_spec);
    println!("  Cost mode: {}", runtime.config.cost_mode);
    println!("  Parallel tracks: {}", runtime.config.max_parallel_tracks);
    if let Some(b) = budget {
        println!("  Token budget: {b}");
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 1: Plan generation
    // -----------------------------------------------------------------------
    let plan = runtime
        .generate_plan(&task_spec)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!(
        "  Planned: {} components ({} critical, {} standard, {} mechanical)",
        plan.total_components,
        plan.critical_count,
        plan.standard_count,
        plan.mechanical_count,
    );
    println!();

    // Show initial progress
    let status = runtime.status();
    display::print_progress(&status, &runtime.token_tracker);

    // -----------------------------------------------------------------------
    // Step 2: Execute the plan
    // -----------------------------------------------------------------------
    let summary = runtime
        .run_loop()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    display::print_completion_summary(&summary);

    Ok(())
}

// ---------------------------------------------------------------------------
// kairo status
// ---------------------------------------------------------------------------

/// Show status of the latest in-progress or completed task.
pub async fn handle_status(project_root: &Path) -> Result<()> {
    let kairo_dir = project_root.join(".kairo");
    if !kairo_dir.exists() {
        bail!(
            "No .kairo/ directory found at {}. Run `kairo init` first.",
            project_root.display()
        );
    }

    let checkpoint_dir = kairo_dir.join("checkpoints");
    let mgr = CheckpointManager::new(checkpoint_dir.clone(), 10, 20);

    // List checkpoints.
    let checkpoints = mgr
        .list_checkpoints()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if checkpoints.is_empty() {
        println!();
        println!("  No task in progress. No checkpoints found.");
        println!("  Run `kairo run <task>` to start a task.");
        println!();
        return Ok(());
    }

    // Restore the latest checkpoint to read the full status.
    match mgr.restore_latest() {
        Ok(restored) => {
            let status = restored.arena.status_summary();
            display::print_status_summary(&status, &restored.token_tracker, &checkpoints);
        }
        Err(e) => {
            tracing::warn!(error = %e, "Could not restore latest checkpoint for full status");
            // Fall back to just showing checkpoint metadata.
            println!();
            println!("  Checkpoint metadata (could not restore full state):");
            println!();
            for cp in checkpoints.iter().rev().take(5) {
                println!(
                    "    step {:>5}  |  {} nodes  |  {} completed  |  {}",
                    cp.step,
                    cp.node_count,
                    cp.completed_count,
                    cp.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                );
            }
            println!();
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// kairo resume
// ---------------------------------------------------------------------------

/// Resume execution from the latest checkpoint.
pub async fn handle_resume(project_root: &Path) -> Result<()> {
    let kairo_dir = project_root.join(".kairo");
    if !kairo_dir.exists() {
        bail!(
            "No .kairo/ directory found at {}. Run `kairo init` first.",
            project_root.display()
        );
    }

    tracing::info!("Resuming from latest checkpoint");

    // Build a default config to initialize the runtime (the resume will
    // restore the arena and tracker from the checkpoint).
    let overrides = CliOverrides {
        cost_mode: CostMode::Balanced,
        cost_mode_explicit: false, // Resume uses config file defaults
        parallel: 3,
        parallel_explicit: false,
        budget: None,
        provider_kind: ProviderKind::Anthropic,
        model: "claude-sonnet-4-20250514".to_string(),
    };
    let config = config::load_config(project_root, &overrides)
        .context("failed to load configuration")?;

    let mut runtime = Runtime::new(config).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Restore from checkpoint.
    restore_from_checkpoint(&mut runtime, project_root)?;

    let status = runtime.status();
    display::print_progress(&status, &runtime.token_tracker);

    // Resume execution — continue the loop from where we left off.
    let summary = runtime
        .run_loop()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    display::print_completion_summary(&summary);

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a task argument: if it is a path to an existing file, read the
/// file contents as the spec. Otherwise treat it as a literal description.
fn resolve_task(project_root: &Path, task: &str) -> Result<String> {
    let candidate = project_root.join(task);
    if candidate.is_file() {
        let content = std::fs::read_to_string(&candidate)
            .with_context(|| format!("failed to read spec file {}", candidate.display()))?;
        tracing::info!(path = %candidate.display(), "Loaded task spec from file");
        Ok(content)
    } else {
        Ok(task.to_string())
    }
}

/// Restore checkpoint state into a live [`Runtime`], running change detection.
fn restore_from_checkpoint(runtime: &mut Runtime, project_root: &Path) -> Result<()> {
    let checkpoint_dir = project_root.join(".kairo/checkpoints");
    let mgr = CheckpointManager::new(checkpoint_dir, 10, 20);

    let mut restored = mgr
        .restore_latest()
        .map_err(|e| anyhow::anyhow!("Failed to restore checkpoint: {e}"))?;

    // Run change detection.
    let affected = kairo_core::persistence::resume_with_change_detection(&mut restored, project_root)
        .map_err(|e| anyhow::anyhow!("Change detection failed: {e}"))?;

    if !affected.is_empty() {
        tracing::warn!(
            count = affected.len(),
            "Re-verification needed for nodes modified since last checkpoint"
        );
    }

    // Transplant restored state into the live runtime.
    runtime.arena = restored.arena;
    runtime.token_tracker = restored.token_tracker;
    runtime.compliance = restored.compliance;

    // Restore controller recurrent state (if available).
    if !restored.controller_state.is_empty() {
        let ok = runtime.controller.deserialize_state(&restored.controller_state);
        if ok {
            tracing::info!("Controller recurrent state restored");
        }
    }

    // Restore session summaries (if available).
    if let Some(session_snapshot) = restored.session_snapshot {
        runtime.session_manager.restore_from_snapshot(session_snapshot);
        tracing::info!("Session summaries restored");
    }

    tracing::info!(
        step = restored.meta.step,
        nodes = restored.meta.node_count,
        completed = restored.meta.completed_count,
        "Checkpoint restored successfully"
    );

    Ok(())
}
