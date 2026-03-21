//! Verification result types and scope definitions (§6).
//!
//! These types are the vocabulary shared across the deterministic verifier,
//! LLM auditor, and the runner that coordinates them.

use crate::arena::node::{DeterministicVerdict, LLMVerdict, Priority};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// VerificationScope — what to verify
// ---------------------------------------------------------------------------

/// Defines the scope of a verification run.
///
/// Created by the runner from the node's implementation output:
/// which files changed, which node produced them, and at what priority.
#[derive(Debug, Clone)]
pub struct VerificationScope {
    /// Files that were created or modified by this node's implementation.
    pub changed_files: Vec<PathBuf>,
    /// The graph node ID that produced these changes.
    pub node_id: u32,
    /// Priority of the node (drives verification depth).
    pub priority: Priority,
    /// Node title (for audit context).
    pub node_title: String,
    /// Node specification text (for audit context).
    pub node_spec: String,
    /// Project root directory.
    pub project_root: PathBuf,
}

impl VerificationScope {
    /// Whether any files are in scope.
    pub fn has_files(&self) -> bool {
        !self.changed_files.is_empty()
    }

    /// Number of files in scope.
    pub fn file_count(&self) -> usize {
        self.changed_files.len()
    }

    /// File extensions present in the scope (deduplicated).
    pub fn extensions(&self) -> Vec<String> {
        let mut exts: Vec<String> = self
            .changed_files
            .iter()
            .filter_map(|p| p.extension())
            .filter_map(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .collect();
        exts.sort();
        exts.dedup();
        exts
    }
}

// ---------------------------------------------------------------------------
// OverallVerdict — combined L1 + L2
// ---------------------------------------------------------------------------

/// The final verdict after both verification layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverallVerdict {
    /// Both layers passed (or skipped layers passed by policy).
    Pass,
    /// At least one layer failed.
    Fail,
    /// Verification could not be performed (no toolchain, no LLM, etc.)
    Unavailable,
}

impl OverallVerdict {
    /// Convert to arena-level verdicts.
    pub fn to_det_verdict(&self) -> DeterministicVerdict {
        match self {
            Self::Pass => DeterministicVerdict::Pass,
            Self::Fail => DeterministicVerdict::Fail,
            Self::Unavailable => DeterministicVerdict::Unavailable,
        }
    }

    pub fn to_llm_verdict(&self) -> LLMVerdict {
        match self {
            Self::Pass => LLMVerdict::Pass,
            Self::Fail => LLMVerdict::Fail,
            Self::Unavailable => LLMVerdict::Skipped,
        }
    }
}

// ---------------------------------------------------------------------------
// VerificationResult — the full result
// ---------------------------------------------------------------------------

/// Complete verification result combining Layer 1 and Layer 2 outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Node that was verified.
    pub node_id: u32,
    /// Deterministic (L1) result.
    pub det_result: Option<super::deterministic::DeterministicResult>,
    /// LLM audit (L2) result.
    pub llm_result: Option<super::llm_audit::AuditResult>,
    /// Combined verdict.
    pub overall: OverallVerdict,
    /// Human-readable summary of issues found (empty if passed).
    pub issues: Vec<String>,
    /// Total wall-clock duration of the verification in milliseconds.
    pub duration_ms: u64,
}

impl VerificationResult {
    /// Create a passing result with no issues.
    pub fn pass(node_id: u32, duration_ms: u64) -> Self {
        Self {
            node_id,
            det_result: None,
            llm_result: None,
            overall: OverallVerdict::Pass,
            issues: Vec::new(),
            duration_ms,
        }
    }

    /// Create a failing result from deterministic checks.
    pub fn fail_deterministic(
        node_id: u32,
        det_result: super::deterministic::DeterministicResult,
        duration_ms: u64,
    ) -> Self {
        let issues = det_result
            .failed_steps()
            .map(|s| format!("{}: {}", s.step_name, s.stderr_summary()))
            .collect();
        Self {
            node_id,
            det_result: Some(det_result),
            llm_result: None,
            overall: OverallVerdict::Fail,
            issues,
            duration_ms,
        }
    }

    /// Create a failing result from LLM audit.
    pub fn fail_audit(
        node_id: u32,
        llm_result: super::llm_audit::AuditResult,
        det_result: Option<super::deterministic::DeterministicResult>,
        duration_ms: u64,
    ) -> Self {
        let issues = llm_result.issues.clone();
        Self {
            node_id,
            det_result,
            llm_result: Some(llm_result),
            overall: OverallVerdict::Fail,
            issues,
            duration_ms,
        }
    }

    /// Create an unavailable result (no toolchain).
    pub fn unavailable(node_id: u32) -> Self {
        Self {
            node_id,
            det_result: None,
            llm_result: None,
            overall: OverallVerdict::Unavailable,
            issues: Vec::new(),
            duration_ms: 0,
        }
    }

    /// Whether verification passed.
    pub fn passed(&self) -> bool {
        self.overall == OverallVerdict::Pass
    }

    /// Whether verification failed.
    pub fn failed(&self) -> bool {
        self.overall == OverallVerdict::Fail
    }

    /// Arena-compatible deterministic verdict.
    pub fn det_verdict(&self) -> DeterministicVerdict {
        match &self.det_result {
            Some(dr) => {
                if dr.passed() {
                    DeterministicVerdict::Pass
                } else {
                    DeterministicVerdict::Fail
                }
            }
            None => DeterministicVerdict::NotRun,
        }
    }

    /// Arena-compatible LLM verdict.
    pub fn llm_verdict(&self) -> LLMVerdict {
        match &self.llm_result {
            Some(ar) => {
                if ar.passed {
                    LLMVerdict::Pass
                } else {
                    LLMVerdict::Fail
                }
            }
            None => LLMVerdict::NotRun,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_scope_extensions() {
        let scope = VerificationScope {
            changed_files: vec![
                PathBuf::from("src/main.rs"),
                PathBuf::from("src/lib.rs"),
                PathBuf::from("src/utils.ts"),
            ],
            node_id: 1,
            priority: Priority::Standard,
            node_title: "test".to_string(),
            node_spec: "spec".to_string(),
            project_root: PathBuf::from("/tmp"),
        };

        let exts = scope.extensions();
        assert_eq!(exts, vec!["rs", "ts"]);
    }

    #[test]
    fn test_verification_scope_has_files() {
        let scope = VerificationScope {
            changed_files: vec![],
            node_id: 1,
            priority: Priority::Mechanical,
            node_title: "test".to_string(),
            node_spec: "spec".to_string(),
            project_root: PathBuf::from("/tmp"),
        };

        assert!(!scope.has_files());
        assert_eq!(scope.file_count(), 0);
    }

    #[test]
    fn test_overall_verdict_conversions() {
        assert_eq!(
            OverallVerdict::Pass.to_det_verdict(),
            DeterministicVerdict::Pass
        );
        assert_eq!(
            OverallVerdict::Fail.to_det_verdict(),
            DeterministicVerdict::Fail
        );
        assert_eq!(
            OverallVerdict::Unavailable.to_det_verdict(),
            DeterministicVerdict::Unavailable
        );
    }

    #[test]
    fn test_verification_result_pass() {
        let result = VerificationResult::pass(1, 100);
        assert!(result.passed());
        assert!(!result.failed());
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_verification_result_unavailable() {
        let result = VerificationResult::unavailable(1);
        assert_eq!(result.overall, OverallVerdict::Unavailable);
        assert_eq!(result.det_verdict(), DeterministicVerdict::NotRun);
        assert_eq!(result.llm_verdict(), LLMVerdict::NotRun);
    }
}
