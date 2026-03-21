//! Verification subsystem (§6).
//!
//! Two-layer verification ensures code correctness:
//!
//! **Layer 1 — Deterministic** (`deterministic.rs`):
//! Runs a chain of automated checks: Syntax -> Typecheck -> Build -> Lint ->
//! Targeted Tests -> Regression Tests. Each step is a shell command derived
//! from the project fingerprint.
//!
//! **Layer 2 — LLM Audit** (`llm_audit.rs`):
//! An adversarial LLM review with temperature 0.3 that checks for spec
//! compliance, edge cases, error handling, and logic correctness.
//!
//! **Runner** (`runner.rs`):
//! Coordinates L1 and L2 based on the priority x cost-mode matrix from §6.4.
//!
//! **Infrastructure** (`infrastructure.rs`):
//! Detects missing test/lint/type infrastructure and generates prerequisite
//! setup nodes before verification can proceed.
//!
//! **Verdicts** (`verdicts.rs`):
//! Combined verification result types.

pub mod deterministic;
pub mod infrastructure;
pub mod llm_audit;
pub mod runner;
pub mod verdicts;

pub use deterministic::{CheckResult, DeterministicResult, DeterministicVerifier, StepName};
pub use infrastructure::{plan_prerequisites, PrerequisiteNode};
pub use llm_audit::{AuditRequest, AuditResult, LlmAuditor};
pub use runner::{VerificationCostMode, VerificationRunner};
pub use verdicts::{OverallVerdict, VerificationResult, VerificationScope};
