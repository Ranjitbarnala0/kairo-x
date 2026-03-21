//! Deterministic response classification (§Flaw 3, Layer 1).
//!
//! Classifies LLM responses using pattern-matching rules applied in priority order.
//! When the deterministic layer cannot make a confident determination, it delegates
//! to the heuristic fallback classifier.

use kairo_llm::call::LLMCallType;
use kairo_llm::response::ResponseClass;

use super::fallback::heuristic_classify;
use super::patterns::{
    self, ERROR_PATTERNS, FAIL_PATTERN, PASS_PATTERN, QUESTION_PATTERNS, REFUSAL_PATTERNS,
    UNIVERSAL_PLACEHOLDER_PATTERNS,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Classify an LLM response into a [`ResponseClass`] using deterministic rules.
///
/// The classification proceeds through layers in priority order:
///
/// 1. **PASS/FAIL keywords** — for Verify and Audit call types, check for
///    explicit PASS or FAIL markers at the start of lines.
/// 2. **Refusal detection** — check for known refusal/evasion phrases.
/// 3. **Error detection** — check for error reporting patterns.
/// 4. **Placeholder detection** — check for TODO, stub, and placeholder patterns.
/// 5. **Question detection** — check for clarifying questions.
/// 6. **Incomplete detection** — check for truncation signals.
/// 7. **Heuristic fallback** — when no deterministic rule fires, use
///    response characteristics to infer the class.
pub fn classify_response(response: &str, call_type: LLMCallType) -> ClassificationResult {
    // Layer 1: PASS/FAIL for verification call types
    if matches!(call_type, LLMCallType::Verify | LLMCallType::Audit) {
        if let Some(result) = classify_verification_response(response) {
            return result;
        }
    }

    // Layer 1: Refusal detection (high priority — override everything)
    if let Some(matched) = match_any_pattern(response, &REFUSAL_PATTERNS) {
        return ClassificationResult {
            class: ResponseClass::Refusal,
            confidence: Confidence::High,
            matched_pattern: Some(matched),
            layer: ClassificationLayer::Deterministic,
        };
    }

    // Layer 1: Error detection
    if let Some(matched) = match_any_pattern(response, &ERROR_PATTERNS) {
        // Only classify as error if the response is short (long responses with error
        // mentions are probably discussing errors, not reporting them)
        if response.len() < 500 {
            return ClassificationResult {
                class: ResponseClass::Error,
                confidence: Confidence::Medium,
                matched_pattern: Some(matched),
                layer: ClassificationLayer::Deterministic,
            };
        }
    }

    // Layer 1: Placeholder detection
    // Skip for Plan/Decompose calls since plan descriptions naturally contain
    // words like "implement", "add", etc. that are stub indicators in code.
    if !matches!(call_type, LLMCallType::Plan | LLMCallType::Decompose) {
        if let Some(matched) = match_any_pattern(response, &UNIVERSAL_PLACEHOLDER_PATTERNS) {
            return ClassificationResult {
                class: ResponseClass::PlaceholderDetected,
                confidence: Confidence::High,
                matched_pattern: Some(matched),
                layer: ClassificationLayer::Deterministic,
            };
        }
    }

    // Layer 1: Question detection
    if let Some(matched) = match_any_pattern(response, &QUESTION_PATTERNS) {
        // Questions should dominate the response, not just appear in passing
        let question_marks = response.chars().filter(|&c| c == '?').count();
        if question_marks >= 2 || response.len() < 500 {
            return ClassificationResult {
                class: ResponseClass::Question,
                confidence: Confidence::Medium,
                matched_pattern: Some(matched),
                layer: ClassificationLayer::Deterministic,
            };
        }
    }

    // Layer 1: Incomplete detection (truncation signals)
    if is_likely_incomplete(response) {
        return ClassificationResult {
            class: ResponseClass::Incomplete,
            confidence: Confidence::Medium,
            matched_pattern: Some("truncation signal".to_string()),
            layer: ClassificationLayer::Deterministic,
        };
    }

    // Layer 2: Heuristic fallback for ambiguous responses
    heuristic_classify(response, call_type)
}

/// Stub for future neural classifier integration.
///
/// Currently returns `Implementation` as the default class. When a neural
/// model is integrated, this will perform inference on ambiguous responses.
pub fn neural_classify(_response: &str, _call_type: LLMCallType) -> ClassificationResult {
    ClassificationResult {
        class: ResponseClass::Implementation,
        confidence: Confidence::Low,
        matched_pattern: None,
        layer: ClassificationLayer::Neural,
    }
}

// ---------------------------------------------------------------------------
// Classification result types
// ---------------------------------------------------------------------------

/// The result of classifying an LLM response.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The determined response class.
    pub class: ResponseClass,
    /// How confident we are in this classification.
    pub confidence: Confidence,
    /// The pattern that triggered this classification, if any.
    pub matched_pattern: Option<String>,
    /// Which classification layer produced this result.
    pub layer: ClassificationLayer,
}

/// Confidence level of a classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Low confidence — heuristic guess.
    Low,
    /// Medium confidence — some signals present but ambiguous.
    Medium,
    /// High confidence — strong deterministic match.
    High,
}

/// Which classification layer produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationLayer {
    /// Deterministic regex-based rules.
    Deterministic,
    /// Heuristic fallback.
    Heuristic,
    /// Neural network inference (future).
    Neural,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Classify verification responses by looking for PASS/FAIL keywords.
fn classify_verification_response(response: &str) -> Option<ClassificationResult> {
    let has_pass = PASS_PATTERN.is_match(response);
    let has_fail = FAIL_PATTERN.is_match(response);

    match (has_pass, has_fail) {
        (true, false) => Some(ClassificationResult {
            class: ResponseClass::VerificationPass,
            confidence: Confidence::High,
            matched_pattern: Some("PASS keyword".to_string()),
            layer: ClassificationLayer::Deterministic,
        }),
        (false, true) => Some(ClassificationResult {
            class: ResponseClass::VerificationFail,
            confidence: Confidence::High,
            matched_pattern: Some("FAIL keyword".to_string()),
            layer: ClassificationLayer::Deterministic,
        }),
        (true, true) => {
            // Both PASS and FAIL present — FAIL takes precedence (conservative)
            Some(ClassificationResult {
                class: ResponseClass::VerificationFail,
                confidence: Confidence::Medium,
                matched_pattern: Some("FAIL keyword (PASS also present — conservative)".to_string()),
                layer: ClassificationLayer::Deterministic,
            })
        }
        (false, false) => None,
    }
}

/// Check if the response shows signs of being incomplete/truncated.
fn is_likely_incomplete(response: &str) -> bool {
    let trimmed = response.trim_end();

    // Trailing ellipsis
    if trimmed.ends_with("...") && trimmed.len() > 20 {
        return true;
    }

    // Unclosed code fences
    let fence_count = response.matches("```").count();
    if fence_count % 2 != 0 {
        return true;
    }

    // Very short response for implementation calls (less than 50 chars)
    // is suspicious but not conclusive — handled by heuristic layer
    false
}

/// Check if any pattern in the given list matches the response.
/// Returns the description of the first matching pattern.
fn match_any_pattern(
    response: &str,
    patterns: &[patterns::CompiledPattern],
) -> Option<String> {
    for pattern in patterns {
        if pattern.is_match(response) {
            return Some(pattern.description.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_pass() {
        let result = classify_response("PASS\nEverything looks correct.", LLMCallType::Verify);
        assert_eq!(result.class, ResponseClass::VerificationPass);
        assert_eq!(result.confidence, Confidence::High);
    }

    #[test]
    fn test_verify_fail() {
        let result = classify_response(
            "FAIL\n1. Missing error handling\n2. No input validation",
            LLMCallType::Audit,
        );
        assert_eq!(result.class, ResponseClass::VerificationFail);
        assert_eq!(result.confidence, Confidence::High);
    }

    #[test]
    fn test_verify_both_pass_fail_conservative() {
        // Both PASS and FAIL at line starts — FAIL should take precedence
        let result = classify_response(
            "PASS on the structure.\nFAIL on error handling.",
            LLMCallType::Verify,
        );
        assert_eq!(result.class, ResponseClass::VerificationFail);
    }

    #[test]
    fn test_refusal_detection() {
        let result = classify_response(
            "I'm unable to help with that request.",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::Refusal);
    }

    #[test]
    fn test_refusal_sorry() {
        let result = classify_response(
            "I'm sorry, but I can't write malicious code.",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::Refusal);
    }

    #[test]
    fn test_placeholder_detection_todo() {
        let result = classify_response(
            "```rust\nfn process() {\n    // TODO: implement the processing logic\n}\n```",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::PlaceholderDetected);
    }

    #[test]
    fn test_placeholder_detection_implement_this() {
        let result = classify_response(
            "```python\ndef process():\n    # implement this later\n    pass\n```",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::PlaceholderDetected);
    }

    #[test]
    fn test_error_short_response() {
        let result = classify_response(
            "error: cannot find module 'express'",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::Error);
    }

    #[test]
    fn test_question_detection() {
        let result = classify_response(
            "Before I proceed, could you clarify the expected return type?",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::Question);
    }

    #[test]
    fn test_incomplete_unclosed_fence() {
        let result = classify_response(
            "Here is the implementation:\n```rust\nfn main() {\n    let x = 42;\n",
            LLMCallType::Implement,
        );
        assert_eq!(result.class, ResponseClass::Incomplete);
    }

    #[test]
    fn test_clean_implementation_falls_through_to_heuristic() {
        let result = classify_response(
            "```rust\nfn calculate_sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n```\n\nThis function takes two integers and returns their sum.",
            LLMCallType::Implement,
        );
        // Should be classified as Implementation by the heuristic fallback
        assert_eq!(result.class, ResponseClass::Implementation);
    }

    #[test]
    fn test_plan_response() {
        let result = classify_response(
            "## Plan\n\n1. Create the database schema\n2. Implement the API endpoints\n3. Add authentication middleware\n4. Write integration tests\n\nDependencies: Step 2 depends on Step 1.",
            LLMCallType::Plan,
        );
        assert_eq!(result.class, ResponseClass::Plan);
    }

    #[test]
    fn test_neural_classify_stub() {
        let result = neural_classify("anything", LLMCallType::Implement);
        assert_eq!(result.class, ResponseClass::Implementation);
        assert_eq!(result.layer, ClassificationLayer::Neural);
        assert_eq!(result.confidence, Confidence::Low);
    }
}
