//! Truncation detection and continuation protocol (§Flaw 12).
//!
//! LLMs have output token limits. When a response is truncated, we detect it
//! and issue continuation requests in the same session.

use crate::call::LLMCallType;
use crate::response::StopReason;

/// Maximum number of continuation attempts before we give up and decompose the node.
pub const MAX_CONTINUATIONS: u32 = 3;

/// Detect whether an LLM response was truncated.
///
/// Uses multiple signals:
/// 1. API-reported stop_reason (most reliable)
/// 2. Trailing ellipsis
/// 3. Unclosed code block
/// 4. Significant brace/bracket imbalance (for code responses)
pub fn is_truncated(response: &str, stop_reason: StopReason, call_type: LLMCallType) -> bool {
    // Signal 1: API explicitly says we hit the token limit
    if stop_reason.is_truncated() {
        return true;
    }

    let trimmed = response.trim_end();

    // Signal 2: Response ends with ellipsis (common truncation artifact).
    // Only flag if the response is longer than 500 chars — short responses
    // legitimately end with ellipsis.
    if trimmed.ends_with("...") && trimmed.len() > 500 {
        return true;
    }

    // Signal 3: Unclosed code fence — response has an odd number of ``` markers
    let fence_count = response.matches("```").count();
    if fence_count % 2 != 0 {
        return true;
    }

    // Signal 4: For code responses, check brace/bracket balance
    if matches!(call_type, LLMCallType::Implement | LLMCallType::Fix)
        && has_significant_brace_imbalance(response)
    {
        return true;
    }

    false
}

/// Check for significant brace/bracket imbalance in code.
///
/// A small imbalance (1-2) could be intentional (e.g., a partial edit block).
/// A large imbalance (>2) strongly suggests truncation.
fn has_significant_brace_imbalance(text: &str) -> bool {
    let mut in_string = false;
    let mut string_char = '\0';
    let mut escaped = false;
    let mut brace_depth: i32 = 0;
    let mut paren_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;

    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if escaped {
            escaped = false;
            continue;
        }

        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }

        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }

        // Detect ``` code fences — skip everything inside them so that
        // braces in fenced code blocks don't pollute the balance count.
        if ch == '`' && chars.as_str().starts_with("``") {
            chars.next(); // second `
            chars.next(); // third `
            // Skip until the closing ``` sequence
            let mut fence_count = 0;
            for fence_ch in chars.by_ref() {
                if fence_ch == '`' {
                    fence_count += 1;
                    if fence_count >= 3 {
                        break;
                    }
                } else {
                    fence_count = 0;
                }
            }
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
            }
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            _ => {}
        }
    }

    // Significant imbalance: more than 2 unclosed delimiters total
    let total_imbalance = brace_depth.max(0) + paren_depth.max(0) + bracket_depth.max(0);
    total_imbalance > 2
}

/// Generate the continuation prompt to send in the same session.
pub fn continuation_prompt() -> &'static str {
    "Your response was truncated. Continue from exactly where you stopped. \
     Do not repeat any content you already generated."
}

/// Generate the decomposition prompt when continuation fails.
pub fn decomposition_prompt(node_title: &str, node_spec: &str) -> String {
    format!(
        "The component \"{node_title}\" is too large to implement in a single response \
         (multiple continuation attempts were truncated).\n\n\
         Specification:\n{node_spec}\n\n\
         Break this into smaller, independently implementable sub-components.\n\
         Each sub-component should produce working code on its own.\n\
         List dependencies between sub-components.\n\
         Output as JSON: [{{\"id\": 1, \"title\": \"...\", \"spec\": \"...\", \
         \"priority\": \"standard\", \"depends_on\": []}}]"
    )
}

/// Concatenate a continued response onto the original.
///
/// Handles edge cases like duplicate content at the boundary.
pub fn concatenate_responses(original: &str, continuation: &str) -> String {
    let original_trimmed = original.trim_end();
    let continuation_trimmed = continuation.trim_start();

    // Check for overlap at the boundary: if the continuation starts with text
    // that appears at the end of the original, skip the duplicate.
    let overlap = find_overlap(original_trimmed, continuation_trimmed);

    if overlap > 0 {
        // The overlap was found via byte comparison, so ensure the slice point
        // falls on a valid UTF-8 char boundary to avoid panics on multibyte text.
        let safe_overlap = if continuation_trimmed.is_char_boundary(overlap) {
            overlap
        } else {
            // Walk backwards to find the nearest valid boundary
            (0..overlap)
                .rev()
                .find(|&i| continuation_trimmed.is_char_boundary(i))
                .unwrap_or(0)
        };

        if safe_overlap > 0 {
            format!(
                "{}{}",
                original_trimmed,
                &continuation_trimmed[safe_overlap..]
            )
        } else {
            // Boundary adjustment collapsed the overlap — fall through to no-overlap path
            format!("{}\n{}", original_trimmed, continuation_trimmed)
        }
    } else {
        format!("{}\n{}", original_trimmed, continuation_trimmed)
    }
}

/// Find the length of overlapping text between the end of `a` and the start of `b`.
///
/// The overlap is found via byte comparison but is guaranteed to fall on a
/// UTF-8 char boundary in `b` so the caller can safely slice at the returned offset.
fn find_overlap(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let max_check = a_bytes.len().min(b_bytes.len()).min(200); // Check at most 200 bytes

    for overlap_len in (1..=max_check).rev() {
        let a_tail = &a_bytes[a_bytes.len() - overlap_len..];
        let b_head = &b_bytes[..overlap_len];
        if a_tail == b_head {
            // Verify the overlap lands on a UTF-8 char boundary in b.
            // If it doesn't, this match splits a multibyte character and is spurious.
            if b.is_char_boundary(overlap_len) {
                return overlap_len;
            }
        }
    }

    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncation_by_stop_reason() {
        assert!(is_truncated("hello", StopReason::MaxTokens, LLMCallType::Implement));
        assert!(!is_truncated("hello", StopReason::EndTurn, LLMCallType::Implement));
    }

    #[test]
    fn test_truncation_by_ellipsis_long_response() {
        // Responses longer than 500 chars ending with "..." are flagged as truncated
        let long_response = format!("{}{}", "x".repeat(501), "...");
        assert!(is_truncated(
            &long_response,
            StopReason::EndTurn,
            LLMCallType::Implement,
        ));
    }

    #[test]
    fn test_no_truncation_by_ellipsis_short_response() {
        // Short responses ending with "..." are NOT flagged (legitimate use)
        assert!(!is_truncated(
            "fn main() {\n    let x = 42...",
            StopReason::EndTurn,
            LLMCallType::Implement,
        ));
    }

    #[test]
    fn test_truncation_by_unclosed_fence() {
        let response = "Here is the code:\n```rust\nfn main() {\n    println!(\"hello\");\n}\n";
        assert!(is_truncated(response, StopReason::EndTurn, LLMCallType::Implement));
    }

    #[test]
    fn test_no_truncation_complete_response() {
        let response = "```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n";
        assert!(!is_truncated(response, StopReason::EndTurn, LLMCallType::Implement));
    }

    #[test]
    fn test_brace_imbalance() {
        assert!(has_significant_brace_imbalance("fn a() { fn b() { fn c() {"));
        assert!(!has_significant_brace_imbalance("fn a() { } fn b() { }"));
    }

    #[test]
    fn test_brace_imbalance_ignores_strings() {
        assert!(!has_significant_brace_imbalance(r#"let s = "{{{";"#));
    }

    #[test]
    fn test_concatenate_no_overlap() {
        let result = concatenate_responses("hello world", "goodbye world");
        assert_eq!(result, "hello world\ngoodbye world");
    }

    #[test]
    fn test_concatenate_with_overlap() {
        let result = concatenate_responses("hello world", "world goodbye");
        assert_eq!(result, "hello world goodbye");
    }
}
