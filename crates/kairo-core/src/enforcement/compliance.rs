//! Compliance tracking — rolling window of LLM output quality.
//!
//! The [`ComplianceTracker`] maintains a fixed-size rolling window of boolean
//! outcomes (good/bad) from recent LLM calls. It provides metrics used by
//! the template selector to escalate enforcement intensity when the LLM
//! produces repeated bad responses.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the rolling compliance window.
const WINDOW_SIZE: usize = 10;

// ---------------------------------------------------------------------------
// ComplianceTracker
// ---------------------------------------------------------------------------

/// Rolling window tracker for LLM response quality.
///
/// Maintains the last [`WINDOW_SIZE`] outcomes and computes:
/// - Recent success rate (fraction of good responses in the window)
/// - Consecutive failure count (how many bad responses in a row)
///
/// These metrics drive enforcement intensity escalation in the template selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceTracker {
    /// Circular buffer of recent outcomes. `true` = good, `false` = bad.
    window: [bool; WINDOW_SIZE],
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Total number of outcomes recorded (may exceed WINDOW_SIZE).
    total_recorded: u64,
    /// Current consecutive failure count (reset on any success).
    consecutive_failures: u32,
}

impl ComplianceTracker {
    /// Create a new tracker. The window starts empty (all entries default to `true`
    /// to avoid false alarms before we have enough data).
    pub fn new() -> Self {
        Self {
            window: [true; WINDOW_SIZE],
            write_pos: 0,
            total_recorded: 0,
            consecutive_failures: 0,
        }
    }

    /// Record a new outcome.
    ///
    /// `good` is `true` if the LLM response was acceptable (correct classification,
    /// no placeholders, no refusal), `false` if it was a bad response.
    pub fn record(&mut self, good: bool) {
        self.window[self.write_pos] = good;
        self.write_pos = (self.write_pos + 1) % WINDOW_SIZE;
        self.total_recorded += 1;

        if good {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
        }
    }

    /// Fraction of successful outcomes in the recent window (0.0 to 1.0).
    ///
    /// If fewer than WINDOW_SIZE outcomes have been recorded, only the recorded
    /// entries are considered.
    pub fn recent_success_rate(&self) -> f32 {
        let count = self.effective_window_size();
        if count == 0 {
            return 1.0; // No data yet — assume success
        }

        let successes = if self.total_recorded >= WINDOW_SIZE as u64 {
            // Full window: count all true values
            self.window.iter().filter(|&&v| v).count()
        } else {
            // Partial window: only count up to total_recorded
            self.window[..count].iter().filter(|&&v| v).count()
        };

        successes as f32 / count as f32
    }

    /// Number of consecutive bad responses (reset by any good response).
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Total number of outcomes ever recorded.
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Effective window size (min of total_recorded and WINDOW_SIZE).
    pub fn effective_window_size(&self) -> usize {
        (self.total_recorded as usize).min(WINDOW_SIZE)
    }

    /// Whether the tracker indicates an unhealthy pattern.
    ///
    /// Returns `true` if:
    /// - 3+ consecutive failures, OR
    /// - Recent success rate below 50% with at least 4 data points
    pub fn is_unhealthy(&self) -> bool {
        self.consecutive_failures >= 3
            || (self.total_recorded >= 4 && self.recent_success_rate() < 0.5)
    }

    /// Reset the tracker to initial state.
    pub fn reset(&mut self) {
        self.window = [true; WINDOW_SIZE];
        self.write_pos = 0;
        self.total_recorded = 0;
        self.consecutive_failures = 0;
    }
}

impl Default for ComplianceTracker {
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
    fn test_new_tracker_healthy() {
        let tracker = ComplianceTracker::new();
        assert_eq!(tracker.consecutive_failures(), 0);
        assert!((tracker.recent_success_rate() - 1.0).abs() < f32::EPSILON);
        assert!(!tracker.is_unhealthy());
    }

    #[test]
    fn test_all_good() {
        let mut tracker = ComplianceTracker::new();
        for _ in 0..5 {
            tracker.record(true);
        }
        assert_eq!(tracker.consecutive_failures(), 0);
        assert!((tracker.recent_success_rate() - 1.0).abs() < f32::EPSILON);
        assert!(!tracker.is_unhealthy());
    }

    #[test]
    fn test_all_bad() {
        let mut tracker = ComplianceTracker::new();
        for _ in 0..10 {
            tracker.record(false);
        }
        assert_eq!(tracker.consecutive_failures(), 10);
        assert!((tracker.recent_success_rate() - 0.0).abs() < f32::EPSILON);
        assert!(tracker.is_unhealthy());
    }

    #[test]
    fn test_mixed_outcomes() {
        let mut tracker = ComplianceTracker::new();
        // Record: good, bad, good, bad, good
        tracker.record(true);
        tracker.record(false);
        tracker.record(true);
        tracker.record(false);
        tracker.record(true);

        assert_eq!(tracker.consecutive_failures(), 0); // Last was good
        assert_eq!(tracker.total_recorded(), 5);

        // 3 out of 5 recent were good = 0.6
        assert!((tracker.recent_success_rate() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consecutive_failures_reset() {
        let mut tracker = ComplianceTracker::new();
        tracker.record(false);
        tracker.record(false);
        tracker.record(false);
        assert_eq!(tracker.consecutive_failures(), 3);

        tracker.record(true); // Reset streak
        assert_eq!(tracker.consecutive_failures(), 0);
    }

    #[test]
    fn test_window_rolling() {
        let mut tracker = ComplianceTracker::new();

        // Fill window with 10 successes
        for _ in 0..10 {
            tracker.record(true);
        }
        assert!((tracker.recent_success_rate() - 1.0).abs() < f32::EPSILON);

        // Now add 5 failures — window should show 5/10 = 50%
        for _ in 0..5 {
            tracker.record(false);
        }
        assert!((tracker.recent_success_rate() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_unhealthy_consecutive() {
        let mut tracker = ComplianceTracker::new();
        tracker.record(true);
        tracker.record(false);
        tracker.record(false);
        assert!(!tracker.is_unhealthy()); // Only 2 consecutive

        tracker.record(false);
        assert!(tracker.is_unhealthy()); // Now 3 consecutive
    }

    #[test]
    fn test_is_unhealthy_low_rate() {
        let mut tracker = ComplianceTracker::new();
        // Record: bad, good, bad, bad (1/4 = 25%)
        tracker.record(false);
        tracker.record(true);
        tracker.record(false);
        tracker.record(false);

        // consecutive_failures = 2, success_rate = 0.25 < 0.5
        assert!(tracker.is_unhealthy());
    }

    #[test]
    fn test_reset() {
        let mut tracker = ComplianceTracker::new();
        tracker.record(false);
        tracker.record(false);
        tracker.record(false);

        tracker.reset();
        assert_eq!(tracker.consecutive_failures(), 0);
        assert_eq!(tracker.total_recorded(), 0);
        assert!((tracker.recent_success_rate() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_effective_window_size() {
        let mut tracker = ComplianceTracker::new();
        assert_eq!(tracker.effective_window_size(), 0);

        tracker.record(true);
        assert_eq!(tracker.effective_window_size(), 1);

        for _ in 0..20 {
            tracker.record(true);
        }
        assert_eq!(tracker.effective_window_size(), WINDOW_SIZE);
    }

    #[test]
    fn test_window_wraps_correctly() {
        let mut tracker = ComplianceTracker::new();

        // Fill the window twice to ensure wrapping works
        for _ in 0..20 {
            tracker.record(true);
        }

        // Replace all with failures
        for _ in 0..10 {
            tracker.record(false);
        }

        assert!((tracker.recent_success_rate() - 0.0).abs() < f32::EPSILON);
    }
}
