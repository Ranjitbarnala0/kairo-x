//! Session manager (§5): LLM session lifecycle and continuation logic.
//!
//! Each session represents a conversation with an LLM, bound to a specific
//! graph node. Sessions persist across multiple calls (implement, verify, fix)
//! to maintain context continuity.

use kairo_llm::call::{LLMCallType, Message};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// An active LLM conversation session bound to a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session ID.
    pub id: u32,
    /// Arena node ID this session is bound to.
    pub bound_node_id: u32,
    /// Messages exchanged in this session.
    pub messages: Vec<Message>,
    /// Cumulative input tokens consumed.
    pub total_input_tokens: u32,
    /// Cumulative output tokens generated.
    pub total_output_tokens: u32,
    /// XXH3 fingerprint of the context package used when this session was created.
    /// Used to detect context staleness.
    pub context_fingerprint: u64,
    /// Number of turns (user/assistant exchanges) in this session.
    pub turn_count: u32,
    /// Number of consecutive bad responses in this session.
    pub bad_response_streak: u32,
    /// Whether the session is still active.
    pub active: bool,
    /// The LLM call type of the most recent call in this session.
    pub last_call_type: Option<LLMCallType>,
    /// Timestamp of session creation (epoch millis).
    pub created_at_ms: i64,
    /// Timestamp of last activity (epoch millis).
    pub last_active_ms: i64,
}

impl Session {
    /// Create a new session bound to the given node.
    fn new(id: u32, node_id: u32, context_fingerprint: u64) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id,
            bound_node_id: node_id,
            messages: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            context_fingerprint,
            turn_count: 0,
            bad_response_streak: 0,
            active: true,
            last_call_type: None,
            created_at_ms: now,
            last_active_ms: now,
        }
    }

    /// Add a message to the session history.
    pub fn push_message(&mut self, msg: Message) {
        self.last_active_ms = chrono::Utc::now().timestamp_millis();
        self.messages.push(msg);
    }

    /// Record token usage from a completed LLM call.
    pub fn record_usage(&mut self, input_tokens: u32, output_tokens: u32, call_type: LLMCallType) {
        self.total_input_tokens = self.total_input_tokens.saturating_add(input_tokens);
        self.total_output_tokens = self.total_output_tokens.saturating_add(output_tokens);
        self.turn_count = self.turn_count.saturating_add(1);
        self.last_call_type = Some(call_type);
        self.last_active_ms = chrono::Utc::now().timestamp_millis();
    }

    /// Record a bad response (placeholder, refusal, error, etc.).
    pub fn record_bad_response(&mut self) {
        self.bad_response_streak = self.bad_response_streak.saturating_add(1);
    }

    /// Record a good response (resets the bad streak).
    pub fn record_good_response(&mut self) {
        self.bad_response_streak = 0;
    }

    /// Total tokens used by this session (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens.saturating_add(self.total_output_tokens)
    }

    /// Estimated context window usage based on messages.
    pub fn estimated_context_tokens(&self) -> u32 {
        self.messages.iter().fold(0u32, |acc, m| acc.saturating_add(m.estimated_tokens()))
    }
}

// ---------------------------------------------------------------------------
// Session continuation decision (§5.2)
// ---------------------------------------------------------------------------

/// Reason for the session continuation decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationDecision {
    /// Continue using the existing session.
    Continue,
    /// Close the current session and start a new one.
    NewSession,
}

/// Reason why a new session was recommended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewSessionReason {
    /// No existing session for this node.
    NoExistingSession,
    /// The context has changed (fingerprint mismatch).
    ContextChanged,
    /// The session has accumulated too many bad responses.
    TooManyBadResponses,
    /// The session has too many turns (context window filling up).
    TooManyTurns,
    /// The call type doesn't prefer continuation.
    CallTypePrefersFresh,
    /// Session has been inactive for too long.
    SessionStale,
}

/// Maximum turns before forcing a new session to avoid context window overflow.
const MAX_SESSION_TURNS: u32 = 12;

/// Maximum consecutive bad responses before closing a session.
const MAX_BAD_STREAK: u32 = 3;

/// Staleness threshold in milliseconds (5 minutes).
const STALENESS_THRESHOLD_MS: i64 = 5 * 60 * 1000;

/// Decide whether to continue an existing session or create a new one.
///
/// Decision table from §5.2:
///
/// | Condition               | Decision    |
/// |-------------------------|-------------|
/// | No existing session     | New session |
/// | Context fingerprint changed | New session |
/// | bad_response_streak >= 3 | New session |
/// | turn_count >= 12        | New session |
/// | Call type !prefers_continue | New session |
/// | Session stale (>5 min)  | New session |
/// | Otherwise               | Continue    |
pub fn should_continue_session(
    session: &Session,
    new_context_fingerprint: u64,
    next_call_type: LLMCallType,
) -> (ContinuationDecision, Option<NewSessionReason>) {
    // Context changed
    if session.context_fingerprint != new_context_fingerprint {
        return (
            ContinuationDecision::NewSession,
            Some(NewSessionReason::ContextChanged),
        );
    }

    // Too many bad responses in a row
    if session.bad_response_streak >= MAX_BAD_STREAK {
        return (
            ContinuationDecision::NewSession,
            Some(NewSessionReason::TooManyBadResponses),
        );
    }

    // Too many turns
    if session.turn_count >= MAX_SESSION_TURNS {
        return (
            ContinuationDecision::NewSession,
            Some(NewSessionReason::TooManyTurns),
        );
    }

    // Call type doesn't prefer continuation
    if !next_call_type.prefers_continue() {
        return (
            ContinuationDecision::NewSession,
            Some(NewSessionReason::CallTypePrefersFresh),
        );
    }

    // Session stale
    let now = chrono::Utc::now().timestamp_millis();
    if now - session.last_active_ms > STALENESS_THRESHOLD_MS {
        return (
            ContinuationDecision::NewSession,
            Some(NewSessionReason::SessionStale),
        );
    }

    (ContinuationDecision::Continue, None)
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// Lightweight session metadata for checkpoint serialization.
///
/// Contains everything needed to reconstruct a session's bookkeeping
/// without the full message history (which can be very large and is
/// rebuilt on resume).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: u32,
    pub bound_node_id: u32,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub turn_count: u32,
    pub bad_response_streak: u32,
    pub active: bool,
    pub context_fingerprint: u64,
    pub created_at_ms: i64,
    pub last_active_ms: i64,
}

/// Serializable snapshot of session manager state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagerSnapshot {
    pub summaries: Vec<SessionSummary>,
    pub node_to_session: HashMap<u32, u32>,
    pub next_id: u32,
    pub total_created: u64,
    pub total_closed: u64,
}

/// Manages all active LLM sessions.
///
/// Sessions are indexed by ID and also queryable by bound node ID.
/// The manager handles session creation, lookup, and cleanup.
#[derive(Debug)]
pub struct SessionManager {
    /// All sessions indexed by session ID.
    sessions: HashMap<u32, Session>,
    /// Mapping from arena node ID to active session ID.
    node_to_session: HashMap<u32, u32>,
    /// Next session ID to assign.
    next_id: u32,
    /// Total sessions ever created (for metrics).
    total_created: u64,
    /// Total sessions closed (for metrics).
    total_closed: u64,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            node_to_session: HashMap::new(),
            next_id: 1,
            total_created: 0,
            total_closed: 0,
        }
    }

    /// Create a new session bound to the given node.
    ///
    /// If the node already has an active session, it is closed first.
    pub fn create_session(&mut self, node_id: u32, context_fingerprint: u64) -> u32 {
        // Close any existing session for this node
        if let Some(&existing_id) = self.node_to_session.get(&node_id) {
            self.close_session(existing_id);
        }

        let session_id = self.next_id;
        self.next_id = self.next_id.checked_add(1)
            .expect("session ID space exhausted after 4 billion sessions");

        let session = Session::new(session_id, node_id, context_fingerprint);
        self.sessions.insert(session_id, session);
        self.node_to_session.insert(node_id, session_id);
        self.total_created += 1;

        tracing::debug!(
            session_id,
            node_id,
            "Created new LLM session"
        );

        session_id
    }

    /// Get an immutable reference to a session by ID.
    pub fn get_session(&self, session_id: u32) -> Option<&Session> {
        self.sessions.get(&session_id).filter(|s| s.active)
    }

    /// Get a mutable reference to a session by ID.
    pub fn get_session_mut(&mut self, session_id: u32) -> Option<&mut Session> {
        self.sessions.get_mut(&session_id).filter(|s| s.active)
    }

    /// Get the active session for a given node, if any.
    pub fn get_session_for_node(&self, node_id: u32) -> Option<&Session> {
        self.node_to_session
            .get(&node_id)
            .and_then(|&sid| self.get_session(sid))
    }

    /// Get a mutable reference to the active session for a node.
    pub fn get_session_for_node_mut(&mut self, node_id: u32) -> Option<&mut Session> {
        if let Some(&sid) = self.node_to_session.get(&node_id) {
            self.sessions.get_mut(&sid).filter(|s| s.active)
        } else {
            None
        }
    }

    /// Close a session by ID.
    pub fn close_session(&mut self, session_id: u32) {
        if let Some(session) = self.sessions.get_mut(&session_id) {
            if session.active {
                session.active = false;
                self.node_to_session.remove(&session.bound_node_id);
                self.total_closed += 1;

                tracing::debug!(
                    session_id,
                    node_id = session.bound_node_id,
                    turns = session.turn_count,
                    total_tokens = session.total_tokens(),
                    "Closed LLM session"
                );

                // Free message memory immediately rather than waiting for GC.
                session.messages.clear();
                session.messages.shrink_to_fit();
            }
        }
    }

    /// Close the session for a given node, if any.
    pub fn close_session_for_node(&mut self, node_id: u32) {
        if let Some(&session_id) = self.node_to_session.get(&node_id) {
            self.close_session(session_id);
        }
    }

    /// Remove all closed sessions from the map, reclaiming memory.
    ///
    /// Call periodically (e.g., after each planning cycle) to prevent the
    /// sessions map from growing without bound.
    pub fn gc_closed_sessions(&mut self) {
        self.sessions.retain(|_, session| session.active);
    }

    /// Record a bad response on the session for a given node.
    pub fn record_bad_response(&mut self, node_id: u32) {
        if let Some(session) = self.get_session_for_node_mut(node_id) {
            session.record_bad_response();
        }
    }

    /// Record a good response on the session for a given node.
    pub fn record_good_response(&mut self, node_id: u32) {
        if let Some(session) = self.get_session_for_node_mut(node_id) {
            session.record_good_response();
        }
    }

    /// Get or create a session for a node, based on the continuation decision.
    ///
    /// This is the main entry point for the runtime loop: given a node and the
    /// next call type, either continue the existing session or create a new one.
    pub fn get_or_create_session(
        &mut self,
        node_id: u32,
        context_fingerprint: u64,
        next_call_type: LLMCallType,
    ) -> (u32, bool) {
        if let Some(session) = self.get_session_for_node(node_id) {
            let (decision, reason) =
                should_continue_session(session, context_fingerprint, next_call_type);

            match decision {
                ContinuationDecision::Continue => {
                    let sid = session.id;
                    (sid, false) // existing session, not new
                }
                ContinuationDecision::NewSession => {
                    tracing::debug!(
                        node_id,
                        reason = ?reason,
                        "Session continuation denied, creating new session"
                    );
                    let sid = self.create_session(node_id, context_fingerprint);
                    (sid, true)
                }
            }
        } else {
            let sid = self.create_session(node_id, context_fingerprint);
            (sid, true)
        }
    }

    /// Number of currently active sessions.
    pub fn active_count(&self) -> usize {
        self.node_to_session.len()
    }

    /// Total sessions ever created.
    pub fn total_created(&self) -> u64 {
        self.total_created
    }

    /// Total sessions closed.
    pub fn total_closed(&self) -> u64 {
        self.total_closed
    }

    /// Clean up all sessions for nodes that are in terminal states.
    ///
    /// `terminal_nodes` should be a list of node IDs that are Verified or Failed.
    pub fn cleanup_terminal_nodes(&mut self, terminal_nodes: &[u32]) {
        for &node_id in terminal_nodes {
            self.close_session_for_node(node_id);
        }
    }

    /// Create a serializable snapshot of session metadata (without message
    /// history) for checkpointing.
    pub fn snapshot(&self) -> SessionManagerSnapshot {
        let summaries = self
            .sessions
            .values()
            .map(|s| SessionSummary {
                id: s.id,
                bound_node_id: s.bound_node_id,
                total_input_tokens: s.total_input_tokens,
                total_output_tokens: s.total_output_tokens,
                turn_count: s.turn_count,
                bad_response_streak: s.bad_response_streak,
                active: s.active,
                context_fingerprint: s.context_fingerprint,
                created_at_ms: s.created_at_ms,
                last_active_ms: s.last_active_ms,
            })
            .collect();

        SessionManagerSnapshot {
            summaries,
            node_to_session: self.node_to_session.clone(),
            next_id: self.next_id,
            total_created: self.total_created,
            total_closed: self.total_closed,
        }
    }

    /// Restore session metadata from a checkpoint snapshot.
    ///
    /// Sessions are recreated with empty message histories (since messages
    /// are too large to checkpoint). Active sessions will have their
    /// bookkeeping restored so the controller can make informed decisions.
    pub fn restore_from_snapshot(&mut self, snapshot: SessionManagerSnapshot) {
        self.sessions.clear();
        self.node_to_session = snapshot.node_to_session;
        self.next_id = snapshot.next_id;
        self.total_created = snapshot.total_created;
        self.total_closed = snapshot.total_closed;

        for summary in snapshot.summaries {
            let session = Session {
                id: summary.id,
                bound_node_id: summary.bound_node_id,
                messages: Vec::new(), // Messages not persisted
                total_input_tokens: summary.total_input_tokens,
                total_output_tokens: summary.total_output_tokens,
                context_fingerprint: summary.context_fingerprint,
                turn_count: summary.turn_count,
                bad_response_streak: summary.bad_response_streak,
                active: summary.active,
                last_call_type: None,
                created_at_ms: summary.created_at_ms,
                last_active_ms: summary.last_active_ms,
            };
            self.sessions.insert(summary.id, session);
        }
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_session() {
        let mut mgr = SessionManager::new();
        let sid = mgr.create_session(1, 0xABCD);

        let session = mgr.get_session(sid).unwrap();
        assert_eq!(session.bound_node_id, 1);
        assert_eq!(session.context_fingerprint, 0xABCD);
        assert!(session.active);
        assert_eq!(session.turn_count, 0);
    }

    #[test]
    fn test_get_session_for_node() {
        let mut mgr = SessionManager::new();
        mgr.create_session(42, 0x1234);

        let session = mgr.get_session_for_node(42).unwrap();
        assert_eq!(session.bound_node_id, 42);
    }

    #[test]
    fn test_close_session() {
        let mut mgr = SessionManager::new();
        let sid = mgr.create_session(1, 0);
        mgr.close_session(sid);

        assert!(mgr.get_session(sid).is_none());
        assert!(mgr.get_session_for_node(1).is_none());
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_create_replaces_existing() {
        let mut mgr = SessionManager::new();
        let sid1 = mgr.create_session(1, 0xAAAA);
        let sid2 = mgr.create_session(1, 0xBBBB);

        assert_ne!(sid1, sid2);
        assert!(mgr.get_session(sid1).is_none()); // old session closed
        let session = mgr.get_session(sid2).unwrap();
        assert_eq!(session.context_fingerprint, 0xBBBB);
    }

    #[test]
    fn test_session_token_tracking() {
        let mut mgr = SessionManager::new();
        let sid = mgr.create_session(1, 0);

        let session = mgr.get_session_mut(sid).unwrap();
        session.record_usage(1000, 500, LLMCallType::Implement);
        session.record_usage(800, 300, LLMCallType::Verify);

        assert_eq!(session.total_input_tokens, 1800);
        assert_eq!(session.total_output_tokens, 800);
        assert_eq!(session.total_tokens(), 2600);
        assert_eq!(session.turn_count, 2);
    }

    #[test]
    fn test_bad_response_streak() {
        let mut mgr = SessionManager::new();
        mgr.create_session(1, 0);

        mgr.record_bad_response(1);
        mgr.record_bad_response(1);
        assert_eq!(mgr.get_session_for_node(1).unwrap().bad_response_streak, 2);

        mgr.record_good_response(1);
        assert_eq!(mgr.get_session_for_node(1).unwrap().bad_response_streak, 0);
    }

    #[test]
    fn test_should_continue_normal() {
        let session = Session::new(1, 1, 0xABCD);
        let (decision, _) =
            should_continue_session(&session, 0xABCD, LLMCallType::Verify);
        assert_eq!(decision, ContinuationDecision::Continue);
    }

    #[test]
    fn test_should_not_continue_context_changed() {
        let session = Session::new(1, 1, 0xABCD);
        let (decision, reason) =
            should_continue_session(&session, 0xDEAD, LLMCallType::Verify);
        assert_eq!(decision, ContinuationDecision::NewSession);
        assert_eq!(reason, Some(NewSessionReason::ContextChanged));
    }

    #[test]
    fn test_should_not_continue_bad_streak() {
        let mut session = Session::new(1, 1, 0xABCD);
        session.bad_response_streak = 3;
        let (decision, reason) =
            should_continue_session(&session, 0xABCD, LLMCallType::Verify);
        assert_eq!(decision, ContinuationDecision::NewSession);
        assert_eq!(reason, Some(NewSessionReason::TooManyBadResponses));
    }

    #[test]
    fn test_should_not_continue_too_many_turns() {
        let mut session = Session::new(1, 1, 0xABCD);
        session.turn_count = 12;
        let (decision, reason) =
            should_continue_session(&session, 0xABCD, LLMCallType::Verify);
        assert_eq!(decision, ContinuationDecision::NewSession);
        assert_eq!(reason, Some(NewSessionReason::TooManyTurns));
    }

    #[test]
    fn test_should_not_continue_non_continuing_call_type() {
        let session = Session::new(1, 1, 0xABCD);
        // Plan doesn't prefer continuation
        let (decision, reason) =
            should_continue_session(&session, 0xABCD, LLMCallType::Plan);
        assert_eq!(decision, ContinuationDecision::NewSession);
        assert_eq!(reason, Some(NewSessionReason::CallTypePrefersFresh));
    }

    #[test]
    fn test_get_or_create_new() {
        let mut mgr = SessionManager::new();
        let (sid, is_new) =
            mgr.get_or_create_session(1, 0xABCD, LLMCallType::Implement);
        assert!(is_new);
        assert!(mgr.get_session(sid).is_some());
    }

    #[test]
    fn test_get_or_create_continues() {
        let mut mgr = SessionManager::new();
        let (sid1, _) =
            mgr.get_or_create_session(1, 0xABCD, LLMCallType::Implement);

        // Record usage so it's a real session
        mgr.get_session_mut(sid1).unwrap().record_usage(100, 50, LLMCallType::Implement);

        // Verify call type prefers continuation
        let (sid2, is_new) =
            mgr.get_or_create_session(1, 0xABCD, LLMCallType::Verify);
        assert!(!is_new);
        assert_eq!(sid1, sid2);
    }

    #[test]
    fn test_cleanup_terminal_nodes() {
        let mut mgr = SessionManager::new();
        mgr.create_session(1, 0);
        mgr.create_session(2, 0);
        mgr.create_session(3, 0);

        assert_eq!(mgr.active_count(), 3);

        mgr.cleanup_terminal_nodes(&[1, 3]);
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.get_session_for_node(2).is_some());
    }

    #[test]
    fn test_metrics() {
        let mut mgr = SessionManager::new();
        mgr.create_session(1, 0);
        mgr.create_session(2, 0);
        let sid = mgr.create_session(3, 0);

        assert_eq!(mgr.total_created(), 3);
        assert_eq!(mgr.total_closed(), 0);

        mgr.close_session(sid);
        assert_eq!(mgr.total_closed(), 1);
    }
}
