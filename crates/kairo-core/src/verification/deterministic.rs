//! Deterministic verification (Layer 1, §6.1).
//!
//! Runs a chain of automated checks derived from the project fingerprint:
//! Syntax -> Typecheck -> Build -> Lint -> Targeted Tests -> Regression Tests.
//! Each step is a shell command. Steps are run in order; failure at any step
//! short-circuits the chain.

use crate::fingerprint::detector::ProjectFingerprint;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Step names
// ---------------------------------------------------------------------------

/// Named verification steps in order of execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepName {
    Syntax,
    Build,
    Lint,
    Typecheck,
    Test,
    TargetedTest,
    RegressionTest,
}

impl std::fmt::Display for StepName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Syntax => write!(f, "syntax"),
            Self::Build => write!(f, "build"),
            Self::Lint => write!(f, "lint"),
            Self::Typecheck => write!(f, "typecheck"),
            Self::Test => write!(f, "test"),
            Self::TargetedTest => write!(f, "targeted_test"),
            Self::RegressionTest => write!(f, "regression_test"),
        }
    }
}

// ---------------------------------------------------------------------------
// Check result (per step)
// ---------------------------------------------------------------------------

/// Result of running a single verification step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Which step this result is for.
    pub step_name: StepName,
    /// The command that was executed.
    pub command: String,
    /// Whether the step passed (exit code 0).
    pub passed: bool,
    /// Exit code from the process.
    pub exit_code: Option<i32>,
    /// Captured stdout (truncated to 4KB).
    pub stdout: String,
    /// Captured stderr (truncated to 4KB).
    pub stderr: String,
    /// Duration of this step in milliseconds.
    pub duration_ms: u64,
}

impl CheckResult {
    /// Summary of stderr (first ~500 chars) for issue reporting.
    ///
    /// Uses char-boundary-safe truncation to avoid panicking on multi-byte
    /// UTF-8 sequences.
    pub fn stderr_summary(&self) -> String {
        if self.stderr.len() <= 500 {
            self.stderr.clone()
        } else {
            let boundary = self
                .stderr
                .char_indices()
                .take_while(|&(i, _)| i < 500)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            format!("{}... (truncated)", &self.stderr[..boundary])
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic result (all steps)
// ---------------------------------------------------------------------------

/// Combined result of all deterministic verification steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicResult {
    /// Results for each step that was run.
    pub steps: Vec<CheckResult>,
    /// Whether all steps passed.
    pub all_passed: bool,
    /// Total duration of all steps.
    pub total_duration_ms: u64,
}

impl DeterministicResult {
    /// Whether all deterministic checks passed.
    pub fn passed(&self) -> bool {
        self.all_passed
    }

    /// Iterator over failed steps.
    pub fn failed_steps(&self) -> impl Iterator<Item = &CheckResult> {
        self.steps.iter().filter(|s| !s.passed)
    }

    /// Number of steps that passed.
    pub fn passed_count(&self) -> usize {
        self.steps.iter().filter(|s| s.passed).count()
    }

    /// Number of steps that failed.
    pub fn failed_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.passed).count()
    }
}

// ---------------------------------------------------------------------------
// Deterministic verifier
// ---------------------------------------------------------------------------

/// Errors from the deterministic verifier.
#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("Failed to execute command '{command}': {source}")]
    Execution {
        command: String,
        source: std::io::Error,
    },

    #[error("Command timed out after {timeout_secs}s: {command}")]
    Timeout {
        command: String,
        timeout_secs: u64,
    },
}

/// Runs deterministic verification steps against a project.
pub struct DeterministicVerifier {
    /// Timeout per individual step.
    step_timeout: Duration,
    /// Maximum stderr/stdout capture size.
    max_capture_bytes: usize,
}

impl DeterministicVerifier {
    /// Create a new verifier with the given step timeout.
    pub fn new(step_timeout: Duration) -> Self {
        Self {
            step_timeout,
            max_capture_bytes: 4096,
        }
    }

    /// Run all verification steps from the project fingerprint.
    ///
    /// Steps are executed in chain order. If a step fails and `stop_on_failure`
    /// is true, subsequent steps are skipped.
    pub async fn run(
        &self,
        fingerprint: &ProjectFingerprint,
        project_root: &Path,
        stop_on_failure: bool,
    ) -> Result<DeterministicResult, VerifierError> {
        let chain = fingerprint.verification_chain();
        let mut steps = Vec::with_capacity(chain.len());
        let mut all_passed = true;
        let start = Instant::now();

        for (step_label, command) in &chain {
            let step_name = match *step_label {
                "syntax" => StepName::Syntax,
                "build" => StepName::Build,
                "lint" => StepName::Lint,
                "typecheck" => StepName::Typecheck,
                "test" => StepName::Test,
                "targeted_test" | "targeted" => StepName::TargetedTest,
                "regression_test" | "regression" => StepName::RegressionTest,
                unknown => {
                    tracing::warn!(
                        label = unknown,
                        "Unknown verification step label, skipping"
                    );
                    continue;
                }
            };

            let result = self.run_step(step_name, command, project_root).await?;

            if !result.passed {
                all_passed = false;
            }

            let failed = !result.passed;
            steps.push(result);

            if failed && stop_on_failure {
                break;
            }
        }

        Ok(DeterministicResult {
            steps,
            all_passed,
            total_duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Run a single verification step with timeout enforcement.
    ///
    /// Spawns the command as a child process. Stdout/stderr are drained via
    /// background tasks while `child.wait()` is wrapped in
    /// `tokio::time::timeout`. On timeout the child is killed and a
    /// [`VerifierError::Timeout`] is returned.
    async fn run_step(
        &self,
        step_name: StepName,
        command: &str,
        project_root: &Path,
    ) -> Result<CheckResult, VerifierError> {
        use tokio::io::AsyncReadExt;

        let start = Instant::now();

        tracing::debug!(
            step = %step_name,
            command,
            timeout_secs = self.step_timeout.as_secs(),
            "Running verification step"
        );

        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(project_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| VerifierError::Execution {
                command: command.to_string(),
                source: e,
            })?;

        // Take stdout/stderr handles *before* waiting. `child.wait()` only
        // borrows `&mut child` (it does not consume it), so we retain
        // ownership and can call `child.kill()` on timeout.
        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        // Drain stdout/stderr in background tasks so pipes don't fill up.
        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut h) = stdout_handle {
                let _ = h.read_to_end(&mut buf).await;
            }
            buf
        });
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut h) = stderr_handle {
                let _ = h.read_to_end(&mut buf).await;
            }
            buf
        });

        match tokio::time::timeout(self.step_timeout, child.wait()).await {
            Ok(Ok(status)) => {
                let stdout_bytes = stdout_task.await.unwrap_or_default();
                let stderr_bytes = stderr_task.await.unwrap_or_default();

                let duration_ms = start.elapsed().as_millis() as u64;
                let passed = status.success();
                let stdout = self.truncate_output(&stdout_bytes);
                let stderr = self.truncate_output(&stderr_bytes);

                tracing::debug!(
                    step = %step_name,
                    passed,
                    duration_ms,
                    "Verification step completed"
                );

                Ok(CheckResult {
                    step_name,
                    command: command.to_string(),
                    passed,
                    exit_code: status.code(),
                    stdout,
                    stderr,
                    duration_ms,
                })
            }
            Ok(Err(io_err)) => Err(VerifierError::Execution {
                command: command.to_string(),
                source: io_err,
            }),
            Err(_elapsed) => {
                tracing::warn!(
                    step = %step_name,
                    command,
                    timeout_secs = self.step_timeout.as_secs(),
                    "Verification step timed out, killing process"
                );
                let _ = child.kill().await;
                let _ = child.wait().await;

                Err(VerifierError::Timeout {
                    command: command.to_string(),
                    timeout_secs: self.step_timeout.as_secs(),
                })
            }
        }
    }

    /// Truncate output to the maximum capture size.
    ///
    /// Uses char-boundary-safe truncation to avoid panicking on multi-byte
    /// UTF-8 sequences produced by `from_utf8_lossy` replacement chars.
    fn truncate_output(&self, bytes: &[u8]) -> String {
        let s = String::from_utf8_lossy(bytes);
        if s.len() <= self.max_capture_bytes {
            s.to_string()
        } else {
            let boundary = s
                .char_indices()
                .take_while(|&(i, _)| i < self.max_capture_bytes)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            format!(
                "{}... (truncated, {} bytes total)",
                &s[..boundary],
                s.len()
            )
        }
    }
}

impl Default for DeterministicVerifier {
    fn default() -> Self {
        Self::new(Duration::from_secs(120))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_stderr_summary_short() {
        let result = CheckResult {
            step_name: StepName::Build,
            command: "cargo build".to_string(),
            passed: false,
            exit_code: Some(1),
            stdout: String::new(),
            stderr: "error[E0308]: mismatched types".to_string(),
            duration_ms: 100,
        };
        assert_eq!(result.stderr_summary(), "error[E0308]: mismatched types");
    }

    #[test]
    fn test_check_result_stderr_summary_long() {
        let result = CheckResult {
            step_name: StepName::Build,
            command: "cargo build".to_string(),
            passed: false,
            exit_code: Some(1),
            stdout: String::new(),
            stderr: "x".repeat(1000),
            duration_ms: 100,
        };
        let summary = result.stderr_summary();
        assert!(summary.len() < 600);
        assert!(summary.contains("truncated"));
    }

    #[test]
    fn test_deterministic_result_counts() {
        let result = DeterministicResult {
            steps: vec![
                CheckResult {
                    step_name: StepName::Build,
                    command: "build".to_string(),
                    passed: true,
                    exit_code: Some(0),
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 50,
                },
                CheckResult {
                    step_name: StepName::Test,
                    command: "test".to_string(),
                    passed: false,
                    exit_code: Some(1),
                    stdout: String::new(),
                    stderr: "test failed".to_string(),
                    duration_ms: 100,
                },
            ],
            all_passed: false,
            total_duration_ms: 150,
        };

        assert_eq!(result.passed_count(), 1);
        assert_eq!(result.failed_count(), 1);
        assert!(!result.passed());
    }

    #[test]
    fn test_step_name_display() {
        assert_eq!(format!("{}", StepName::Build), "build");
        assert_eq!(format!("{}", StepName::Test), "test");
    }
}
