//! Heuristic fallback classifier (§Flaw 3, Layer 2).
//!
//! When deterministic rules from `rules.rs` don't produce a confident match,
//! the heuristic classifier uses response characteristics to infer the class:
//!
//! - **Response length**: very short responses are suspicious
//! - **Code-to-text ratio**: implementation responses are code-heavy
//! - **Keyword presence**: plan-like keywords vs code-like keywords
//! - **Structure signals**: numbered lists, markdown headers, code fences

use kairo_llm::call::LLMCallType;
use kairo_llm::response::ResponseClass;

use super::rules::{ClassificationLayer, ClassificationResult, Confidence};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Heuristic-based classification for ambiguous responses.
///
/// Uses response length, code-to-text ratio, keyword presence, and structural
/// signals to determine the most likely [`ResponseClass`].
pub fn heuristic_classify(response: &str, call_type: LLMCallType) -> ClassificationResult {
    let metrics = ResponseMetrics::compute(response);

    // Very short responses for implementation calls are suspicious
    if matches!(call_type, LLMCallType::Implement | LLMCallType::Fix)
        && metrics.total_chars < 50
    {
        return ClassificationResult {
            class: ResponseClass::Incomplete,
            confidence: Confidence::Low,
            matched_pattern: Some(format!(
                "very short response ({} chars) for implementation call",
                metrics.total_chars
            )),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // For Plan calls, check plan-like signals
    if call_type == LLMCallType::Plan && metrics.has_plan_signals() {
        return ClassificationResult {
            class: ResponseClass::Plan,
            confidence: Confidence::Medium,
            matched_pattern: Some("plan-like structure detected".to_string()),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // For Verify/Audit, if we get here it means no PASS/FAIL was found
    if matches!(call_type, LLMCallType::Verify | LLMCallType::Audit) {
        // Ambiguous verification — assume fail (conservative)
        return ClassificationResult {
            class: ResponseClass::VerificationFail,
            confidence: Confidence::Low,
            matched_pattern: Some("no PASS/FAIL keyword found — conservative fail".to_string()),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // For implementation calls, check code ratio
    if matches!(
        call_type,
        LLMCallType::Implement | LLMCallType::Fix | LLMCallType::Debug
    ) && (metrics.code_ratio > 0.3 || metrics.has_code_fence)
    {
        return ClassificationResult {
            class: ResponseClass::Implementation,
            confidence: Confidence::Medium,
            matched_pattern: Some(format!(
                "code ratio {:.1}%, has_fence={}",
                metrics.code_ratio * 100.0,
                metrics.has_code_fence
            )),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // For Explain calls
    if call_type == LLMCallType::Explain {
        // Explanations are usually text-heavy
        return ClassificationResult {
            class: ResponseClass::Implementation, // explanations are "successful" responses
            confidence: Confidence::Medium,
            matched_pattern: Some("explanation response".to_string()),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // For Decompose calls
    if call_type == LLMCallType::Decompose
        && (metrics.has_plan_signals() || metrics.has_json_array)
    {
        return ClassificationResult {
            class: ResponseClass::Plan,
            confidence: Confidence::Medium,
            matched_pattern: Some("decomposition with plan signals".to_string()),
            layer: ClassificationLayer::Heuristic,
        };
    }

    // Default: assume implementation for code-heavy, plan for text-heavy
    if metrics.code_ratio > 0.2 {
        ClassificationResult {
            class: ResponseClass::Implementation,
            confidence: Confidence::Low,
            matched_pattern: Some(format!("code ratio {:.1}%", metrics.code_ratio * 100.0)),
            layer: ClassificationLayer::Heuristic,
        }
    } else {
        ClassificationResult {
            class: ResponseClass::Implementation,
            confidence: Confidence::Low,
            matched_pattern: Some("default fallback".to_string()),
            layer: ClassificationLayer::Heuristic,
        }
    }
}

// ---------------------------------------------------------------------------
// Response metrics
// ---------------------------------------------------------------------------

/// Computed metrics about a response used for heuristic classification.
#[derive(Debug)]
struct ResponseMetrics {
    /// Total character count.
    total_chars: usize,
    /// Ratio of characters inside code fences to total characters.
    code_ratio: f32,
    /// Whether the response contains at least one code fence.
    has_code_fence: bool,
    /// Number of numbered list items (e.g., "1. ", "2. ").
    numbered_items: usize,
    /// Number of markdown headers (lines starting with #).
    header_count: usize,
    /// Whether the response contains what looks like a JSON array.
    has_json_array: bool,
}

impl ResponseMetrics {
    /// Compute metrics for a response string.
    fn compute(response: &str) -> Self {
        let total_chars = response.len();
        let lines: Vec<&str> = response.lines().collect();

        // Count code fence characters
        let (code_chars, has_code_fence) = Self::measure_code_content(response);
        let code_ratio = if total_chars > 0 {
            code_chars as f32 / total_chars as f32
        } else {
            0.0
        };

        // Count numbered list items
        let numbered_items = lines
            .iter()
            .filter(|l| {
                let trimmed = l.trim_start();
                trimmed.len() >= 3
                    && trimmed.as_bytes()[0].is_ascii_digit()
                    && (trimmed.as_bytes().get(1) == Some(&b'.')
                        || (trimmed.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit())
                            && trimmed.as_bytes().get(2) == Some(&b'.')))
            })
            .count();

        // Count markdown headers
        let header_count = lines
            .iter()
            .filter(|l| l.trim_start().starts_with('#'))
            .count();

        // Check for JSON array — require both opening bracket and an "id" key
        // to avoid false positives on arbitrary bracket-containing text
        let trimmed_response = response.trim();
        let has_json_array =
            trimmed_response.starts_with('[') && trimmed_response.contains("\"id\":");

        Self {
            total_chars,
            code_ratio,
            has_code_fence,
            numbered_items,
            header_count,
            has_json_array,
        }
    }

    /// Measure characters inside code fences and whether any fences exist.
    fn measure_code_content(response: &str) -> (usize, bool) {
        let mut in_fence = false;
        let mut code_chars = 0usize;
        let mut has_fence = false;

        for line in response.lines() {
            if line.trim_start().starts_with("```") {
                if in_fence {
                    in_fence = false;
                } else {
                    in_fence = true;
                    has_fence = true;
                }
                continue;
            }

            if in_fence {
                code_chars += line.len() + 1; // +1 for newline
            }
        }

        (code_chars, has_fence)
    }

    /// Whether the response shows plan-like structural signals.
    fn has_plan_signals(&self) -> bool {
        // Plan-like if it has headers and numbered lists, or several numbered items
        (self.header_count >= 1 && self.numbered_items >= 3)
            || self.numbered_items >= 4
            || self.has_json_array
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_implementation_is_incomplete() {
        let result = heuristic_classify("fn x() {}", LLMCallType::Implement);
        assert_eq!(result.class, ResponseClass::Incomplete);
    }

    #[test]
    fn test_code_heavy_response_is_implementation() {
        let response = "Here is the implementation:\n\n```rust\nfn calculate_hash(data: &[u8]) -> u64 {\n    let mut hasher = DefaultHasher::new();\n    hasher.write(data);\n    hasher.finish()\n}\n\nfn verify_hash(data: &[u8], expected: u64) -> bool {\n    calculate_hash(data) == expected\n}\n```\n\nThis implements hash calculation and verification.";
        let result = heuristic_classify(response, LLMCallType::Implement);
        assert_eq!(result.class, ResponseClass::Implementation);
    }

    #[test]
    fn test_plan_like_response() {
        let response = "## Architecture Plan\n\n### Phase 1: Core\n1. Create database schema\n2. Implement models\n3. Add migrations\n\n### Phase 2: API\n4. REST endpoints\n5. Authentication\n6. Rate limiting\n\n### Phase 3: Testing\n7. Unit tests\n8. Integration tests";
        let result = heuristic_classify(response, LLMCallType::Plan);
        assert_eq!(result.class, ResponseClass::Plan);
    }

    #[test]
    fn test_verify_without_pass_fail_is_conservative_fail() {
        let response = "The code looks mostly okay but there might be an edge case with empty input.";
        let result = heuristic_classify(response, LLMCallType::Verify);
        assert_eq!(result.class, ResponseClass::VerificationFail);
    }

    #[test]
    fn test_decompose_with_json() {
        let response = r#"[{"id": 1, "title": "Parser", "spec": "Parse input", "depends_on": []}, {"id": 2, "title": "Evaluator", "spec": "Evaluate AST", "depends_on": [1]}]"#;
        let result = heuristic_classify(response, LLMCallType::Decompose);
        assert_eq!(result.class, ResponseClass::Plan);
    }

    #[test]
    fn test_response_metrics_code_ratio() {
        let response = "Explanation text.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nDone.";
        let metrics = ResponseMetrics::compute(response);
        assert!(metrics.code_ratio > 0.0);
        assert!(metrics.has_code_fence);
    }

    #[test]
    fn test_response_metrics_numbered_items() {
        let response = "1. First\n2. Second\n3. Third\n4. Fourth\n5. Fifth";
        let metrics = ResponseMetrics::compute(response);
        assert_eq!(metrics.numbered_items, 5);
    }

    #[test]
    fn test_response_metrics_headers() {
        let response = "# Title\n## Section\n### Subsection";
        let metrics = ResponseMetrics::compute(response);
        assert_eq!(metrics.header_count, 3);
    }
}
