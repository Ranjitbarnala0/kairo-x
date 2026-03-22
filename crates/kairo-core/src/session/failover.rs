//! Session-level failover state tracking.
//!
//! This is a thin wrapper providing session-level awareness of provider failover.
//! The real failover logic lives in `kairo_llm::providers::ProviderManager`.
//! This module tracks which sessions were affected by failover events so the
//! runtime can decide whether to restart sessions on a different provider.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// SessionFailoverState
// ---------------------------------------------------------------------------

/// Tracks failover state relevant to session management.
///
/// When the LLM provider fails over (primary -> fallback), active sessions
/// may need to be restarted since different providers have different context
/// windows, behavior characteristics, and token limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionFailoverState {
    /// Whether a failover has occurred during this execution.
    pub failover_occurred: bool,
    /// How many times failover has happened.
    pub failover_count: u32,
    /// Session IDs that were active when failover occurred. These sessions
    /// should be considered for restart since they may have been mid-conversation
    /// with the primary provider.
    pub affected_sessions: HashSet<u32>,
    /// Whether we have recovered back to the primary provider.
    pub recovered: bool,
}

impl SessionFailoverState {
    /// Create a new failover state with no failover history.
    pub fn new() -> Self {
        Self {
            failover_occurred: false,
            failover_count: 0,
            affected_sessions: HashSet::new(),
            recovered: false,
        }
    }

    /// Record a failover event, marking all given active sessions as affected.
    pub fn record_failover(&mut self, active_session_ids: &[u32]) {
        self.failover_occurred = true;
        self.failover_count = self.failover_count.saturating_add(1);
        self.recovered = false;

        for &sid in active_session_ids {
            self.affected_sessions.insert(sid);
        }

        tracing::warn!(
            failover_count = self.failover_count,
            affected_sessions = self.affected_sessions.len(),
            "Provider failover recorded at session level"
        );
    }

    /// Record recovery back to primary provider.
    pub fn record_recovery(&mut self) {
        self.failover_occurred = false;
        self.recovered = true;
        tracing::info!("Provider recovered to primary at session level");
    }

    /// Whether the system is currently in a failover state (failover occurred
    /// and has not yet recovered).
    pub fn is_in_failover(&self) -> bool {
        self.failover_occurred && !self.recovered
    }

    /// Check if a specific session was affected by failover.
    pub fn is_session_affected(&self, session_id: u32) -> bool {
        self.affected_sessions.contains(&session_id)
    }

    /// Mark a session as handled (e.g., after it has been restarted).
    pub fn mark_session_handled(&mut self, session_id: u32) {
        self.affected_sessions.remove(&session_id);
    }

    /// Get all affected session IDs that still need handling.
    pub fn pending_affected_sessions(&self) -> &HashSet<u32> {
        &self.affected_sessions
    }

    /// Whether any sessions still need handling after failover.
    pub fn has_pending_affected(&self) -> bool {
        !self.affected_sessions.is_empty()
    }

    /// Reset all failover state.
    pub fn reset(&mut self) {
        self.failover_occurred = false;
        self.failover_count = 0;
        self.affected_sessions = HashSet::new();
        self.recovered = false;
    }
}

impl Default for SessionFailoverState {
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
    fn test_new_state() {
        let state = SessionFailoverState::new();
        assert!(!state.failover_occurred);
        assert_eq!(state.failover_count, 0);
        assert!(state.affected_sessions.is_empty());
        assert!(!state.recovered);
    }

    #[test]
    fn test_record_failover() {
        let mut state = SessionFailoverState::new();
        state.record_failover(&[1, 2, 3]);

        assert!(state.failover_occurred);
        assert_eq!(state.failover_count, 1);
        assert_eq!(state.affected_sessions.len(), 3);
        assert!(state.is_session_affected(1));
        assert!(state.is_session_affected(2));
        assert!(state.is_session_affected(3));
        assert!(!state.is_session_affected(4));
    }

    #[test]
    fn test_multiple_failovers() {
        let mut state = SessionFailoverState::new();
        state.record_failover(&[1, 2]);
        state.record_failover(&[2, 3]); // session 2 already present

        assert_eq!(state.failover_count, 2);
        assert_eq!(state.affected_sessions.len(), 3); // 1, 2, 3 (no duplicate)
    }

    #[test]
    fn test_mark_handled() {
        let mut state = SessionFailoverState::new();
        state.record_failover(&[1, 2, 3]);

        state.mark_session_handled(2);
        assert!(!state.is_session_affected(2));
        assert_eq!(state.affected_sessions.len(), 2);
    }

    #[test]
    fn test_has_pending() {
        let mut state = SessionFailoverState::new();
        assert!(!state.has_pending_affected());

        state.record_failover(&[1]);
        assert!(state.has_pending_affected());

        state.mark_session_handled(1);
        assert!(!state.has_pending_affected());
    }

    #[test]
    fn test_recovery() {
        let mut state = SessionFailoverState::new();
        state.record_failover(&[1]);
        assert!(!state.recovered);

        state.record_recovery();
        assert!(state.recovered);
    }

    #[test]
    fn test_reset() {
        let mut state = SessionFailoverState::new();
        state.record_failover(&[1, 2, 3]);
        state.record_recovery();

        state.reset();
        assert!(!state.failover_occurred);
        assert_eq!(state.failover_count, 0);
        assert!(state.affected_sessions.is_empty());
        assert!(!state.recovered);
    }
}
