//! Verification runner — coordinates L1 and L2 verification (§6.3, §6.4).
//!
//! Decides which verification layers to run based on the priority x cost-mode
//! matrix and orchestrates them in sequence.

use crate::arena::node::Priority;
use crate::session::token_tracker::CostMode;

use super::deterministic::DeterministicVerifier;
use super::llm_audit::LlmAuditor;
// VerificationScope, VerificationResult, and OverallVerdict are used by
// callers via the module's pub use; the runner coordinates the verifiers.

// ---------------------------------------------------------------------------
// Verification runner
// ---------------------------------------------------------------------------

/// Coordinates Layer 1 (deterministic) and Layer 2 (LLM audit) verification.
///
/// The runner uses the priority x cost-mode matrix from §6.4 to decide
/// which layers to execute:
///
/// | Priority    | Thorough   | Balanced   | Efficient   |
/// |-------------|------------|------------|-------------|
/// | Critical    | L1 + L2    | L1 + L2    | L1 + L2     |
/// | Standard    | L1 + L2    | L1 + L2    | L1 only     |
/// | Mechanical  | L1 + L2    | L1 only    | L1 only     |
pub struct VerificationRunner {
    /// Deterministic verifier for L1 checks.
    pub deterministic: DeterministicVerifier,
    /// LLM auditor for L2 checks.
    pub auditor: LlmAuditor,
    /// Cost mode governing verification depth.
    pub cost_mode: CostMode,
}

impl VerificationRunner {
    /// Create a new verification runner.
    pub fn new(cost_mode: CostMode) -> Self {
        Self {
            deterministic: DeterministicVerifier::default(),
            auditor: LlmAuditor::default(),
            cost_mode,
        }
    }

    /// Whether L2 (LLM audit) should be run for the given priority.
    pub fn should_run_l2(&self, priority: Priority) -> bool {
        self.cost_mode.should_audit(priority)
    }

    /// Determine the verification layers to run.
    pub fn layers_for(&self, priority: Priority) -> VerificationLayers {
        let run_l2 = self.should_run_l2(priority);
        VerificationLayers {
            run_l1: true, // L1 always runs when available
            run_l2,
            stop_on_l1_failure: !run_l2, // If no L2, stop on L1 failure
        }
    }
}

/// Which verification layers to execute.
#[derive(Debug, Clone, Copy)]
pub struct VerificationLayers {
    /// Whether to run Layer 1 (deterministic checks).
    pub run_l1: bool,
    /// Whether to run Layer 2 (LLM audit).
    pub run_l2: bool,
    /// Whether to stop immediately if L1 fails (skip L2).
    pub stop_on_l1_failure: bool,
}

// Re-export CostMode for the module-level pub use.
pub use crate::session::token_tracker::CostMode as VerificationCostMode;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thorough_mode_all_l2() {
        let runner = VerificationRunner::new(CostMode::Thorough);
        assert!(runner.should_run_l2(Priority::Critical));
        assert!(runner.should_run_l2(Priority::Standard));
        assert!(runner.should_run_l2(Priority::Mechanical));
    }

    #[test]
    fn test_balanced_mode_selective_l2() {
        let runner = VerificationRunner::new(CostMode::Balanced);
        assert!(runner.should_run_l2(Priority::Critical));
        assert!(runner.should_run_l2(Priority::Standard));
        assert!(!runner.should_run_l2(Priority::Mechanical));
    }

    #[test]
    fn test_efficient_mode_minimal_l2() {
        let runner = VerificationRunner::new(CostMode::Efficient);
        assert!(runner.should_run_l2(Priority::Critical));
        assert!(!runner.should_run_l2(Priority::Standard));
        assert!(!runner.should_run_l2(Priority::Mechanical));
    }

    #[test]
    fn test_layers_with_l2() {
        let runner = VerificationRunner::new(CostMode::Thorough);
        let layers = runner.layers_for(Priority::Critical);
        assert!(layers.run_l1);
        assert!(layers.run_l2);
        assert!(!layers.stop_on_l1_failure);
    }

    #[test]
    fn test_layers_without_l2() {
        let runner = VerificationRunner::new(CostMode::Efficient);
        let layers = runner.layers_for(Priority::Standard);
        assert!(layers.run_l1);
        assert!(!layers.run_l2);
        assert!(layers.stop_on_l1_failure);
    }
}
