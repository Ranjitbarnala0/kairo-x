//! LLM audit verification (Layer 2, §6.2).
//!
//! Sends the implementation to an adversarial LLM review with temperature 0.3.
//! The auditor checks for spec compliance, edge cases, error handling, and
//! logic correctness. Uses a separate model when configured (cross-model defense).

use kairo_llm::response::ResponseClass;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Audit request
// ---------------------------------------------------------------------------

/// Request for an LLM audit of a node's implementation.
#[derive(Debug, Clone)]
pub struct AuditRequest {
    /// Node ID being audited.
    pub node_id: u32,
    /// Node title.
    pub node_title: String,
    /// Node specification.
    pub node_spec: String,
    /// The implementation code to audit (full file contents or diff).
    pub implementation: String,
    /// File paths of the implementation.
    pub file_paths: Vec<String>,
}

// ---------------------------------------------------------------------------
// Audit result
// ---------------------------------------------------------------------------

/// Result of an LLM audit call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    /// Whether the audit passed.
    pub passed: bool,
    /// Issues found (empty if passed).
    pub issues: Vec<String>,
    /// Raw LLM response text.
    pub raw_response: String,
    /// How the response was classified.
    pub classification: ResponseClass,
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens consumed.
    pub output_tokens: u32,
}

impl AuditResult {
    /// Create a passing audit result.
    pub fn pass(raw_response: String, input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            passed: true,
            issues: Vec::new(),
            raw_response,
            classification: ResponseClass::VerificationPass,
            input_tokens,
            output_tokens,
        }
    }

    /// Create a failing audit result.
    pub fn fail(
        issues: Vec<String>,
        raw_response: String,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Self {
        Self {
            passed: false,
            issues,
            raw_response,
            classification: ResponseClass::VerificationFail,
            input_tokens,
            output_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// LLM Auditor
// ---------------------------------------------------------------------------

/// Performs LLM-based adversarial audits on implementations.
///
/// The auditor assembles an audit prompt and sends it through the LLM bridge.
/// It classifies the response and extracts issues if the audit fails.
pub struct LlmAuditor {
    /// Whether to use the audit-specific provider (cross-model defense).
    pub use_audit_provider: bool,
}

impl LlmAuditor {
    /// Create a new auditor.
    pub fn new(use_audit_provider: bool) -> Self {
        Self { use_audit_provider }
    }

    /// Parse numbered issues from an LLM audit response.
    ///
    /// Expected format:
    /// ```text
    /// FAIL
    /// 1. Missing error handling for empty input
    /// 2. Off-by-one error in loop bound
    /// ```
    pub fn parse_issues(response: &str) -> Vec<String> {
        let mut issues = Vec::new();

        for line in response.lines() {
            let trimmed = line.trim();
            // Match lines starting with a number followed by a period or parenthesis
            if trimmed.len() >= 3 {
                let first_char = trimmed.as_bytes()[0];
                if first_char.is_ascii_digit() {
                    if let Some(rest) = trimmed
                        .strip_prefix(|c: char| c.is_ascii_digit())
                        .and_then(|s| {
                            s.strip_prefix(|c: char| c.is_ascii_digit())
                                .or(Some(s))
                        })
                    {
                        if rest.starts_with('.') || rest.starts_with(')') {
                            let issue_text = rest[1..].trim();
                            if !issue_text.is_empty() {
                                issues.push(issue_text.to_string());
                            }
                        }
                    }
                }
            }
        }

        issues
    }
}

impl Default for LlmAuditor {
    fn default() -> Self {
        Self::new(false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_issues_standard_format() {
        let response = "FAIL\n1. Missing error handling for empty input\n2. Off-by-one error in loop bound\n3. No validation of user ID parameter";
        let issues = LlmAuditor::parse_issues(response);
        assert_eq!(issues.len(), 3);
        assert_eq!(issues[0], "Missing error handling for empty input");
        assert_eq!(issues[1], "Off-by-one error in loop bound");
    }

    #[test]
    fn test_parse_issues_with_parenthesis() {
        let response = "FAIL\n1) Missing null check\n2) Incorrect return type";
        let issues = LlmAuditor::parse_issues(response);
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn test_parse_issues_pass_response() {
        let response = "PASS\nThe implementation is correct and handles all edge cases.";
        let issues = LlmAuditor::parse_issues(response);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_audit_result_pass() {
        let result = AuditResult::pass("PASS".to_string(), 1000, 50);
        assert!(result.passed);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_audit_result_fail() {
        let result = AuditResult::fail(
            vec!["issue 1".to_string(), "issue 2".to_string()],
            "FAIL\n1. issue 1\n2. issue 2".to_string(),
            1000,
            100,
        );
        assert!(!result.passed);
        assert_eq!(result.issues.len(), 2);
    }
}
