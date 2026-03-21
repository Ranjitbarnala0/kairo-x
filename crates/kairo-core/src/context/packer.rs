//! Greedy context packing (§9.4).
//!
//! Sorts candidates by relevance score descending and greedily fills a
//! token budget. When the remaining budget is larger than a minimum
//! threshold (200 tokens), the next candidate is included with truncation
//! rather than being skipped entirely.

use crate::context::candidates::ContextCandidate;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum remaining tokens to allow partial (truncated) inclusion of a
/// candidate. Below this threshold, we stop packing.
const PARTIAL_INCLUSION_THRESHOLD: u32 = 200;

/// Overhead tokens added per candidate for the label/separator lines.
const PER_CANDIDATE_OVERHEAD: u32 = 8;

// ---------------------------------------------------------------------------
// PackedContext
// ---------------------------------------------------------------------------

/// The output of the context packing step: the selected candidates and
/// metadata about token usage.
#[derive(Debug, Clone)]
pub struct PackedContext {
    /// Candidates selected (in display order: highest score first).
    pub items: Vec<PackedItem>,
    /// Total tokens consumed by included items (including overhead).
    pub total_tokens: u32,
    /// Token budget that was provided.
    pub budget: u32,
    /// Number of candidates that were considered but not included.
    pub dropped_count: usize,
}

impl PackedContext {
    /// Render the packed context to a single string for LLM consumption.
    ///
    /// Format:
    /// ```text
    /// --- [source] label ---
    /// content
    ///
    /// --- [source] label ---
    /// content
    /// ```
    pub fn render(&self) -> String {
        let mut parts = Vec::with_capacity(self.items.len() * 3);
        for item in &self.items {
            parts.push(format!("--- [{}] {} ---", item.source_label, item.label));
            parts.push(item.content.clone());
            parts.push(String::new());
        }
        parts.join("\n")
    }

    /// Whether the packing is empty (no items selected).
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Number of items in the packed context.
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// A single item in the packed context.
#[derive(Debug, Clone)]
pub struct PackedItem {
    /// Human-readable label (file path, symbol name, etc.).
    pub label: String,
    /// Source category string.
    pub source_label: String,
    /// The (possibly truncated) content.
    pub content: String,
    /// XXH3 hash of the original full content.
    pub content_hash: u64,
    /// Whether this item was truncated to fit the budget.
    pub truncated: bool,
    /// Relevance score that determined this item's rank.
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Packing function
// ---------------------------------------------------------------------------

/// Pack candidates into a context package that fits within `token_budget`.
///
/// Algorithm (§9.4):
/// 1. Sort candidates by score descending.
/// 2. For each candidate, if it fits within the remaining budget (including
///    overhead), include it fully.
/// 3. If the candidate does not fit but the remaining budget exceeds
///    `PARTIAL_INCLUSION_THRESHOLD`, truncate the candidate's content and
///    include it partially.
/// 4. Otherwise, skip the candidate.
pub fn pack_context(
    mut candidates: Vec<ContextCandidate>,
    token_budget: u32,
) -> PackedContext {
    // Sort by score descending. Ties broken by smaller estimated_tokens first
    // (prefer cheaper candidates when equally relevant).
    candidates.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.estimated_tokens.cmp(&b.estimated_tokens))
    });

    let mut items = Vec::new();
    let mut remaining = token_budget;
    let mut dropped = 0usize;

    for candidate in &candidates {
        let cost = candidate.estimated_tokens.saturating_add(PER_CANDIDATE_OVERHEAD);

        if cost <= remaining {
            // Full inclusion
            items.push(PackedItem {
                label: candidate.label.clone(),
                source_label: candidate.source.to_string(),
                content: candidate.content.clone(),
                content_hash: candidate.content_hash,
                truncated: false,
                score: candidate.score(),
            });
            remaining = remaining.saturating_sub(cost);
        } else if remaining >= PARTIAL_INCLUSION_THRESHOLD {
            // Partial inclusion: truncate content to fit
            let available_tokens = remaining.saturating_sub(PER_CANDIDATE_OVERHEAD);
            if available_tokens > 0 {
                let truncated_content = truncate_to_tokens(&candidate.content, available_tokens);
                if !truncated_content.is_empty() {
                    items.push(PackedItem {
                        label: candidate.label.clone(),
                        source_label: candidate.source.to_string(),
                        content: truncated_content,
                        content_hash: candidate.content_hash,
                        truncated: true,
                        score: candidate.score(),
                    });
                    remaining = 0;
                } else {
                    dropped += 1;
                }
            } else {
                dropped += 1;
            }
        } else {
            dropped += 1;
        }

        if remaining == 0 {
            // Count remaining candidates as dropped
            dropped += candidates.len() - items.len() - dropped;
            break;
        }
    }

    PackedContext {
        items,
        total_tokens: token_budget.saturating_sub(remaining),
        budget: token_budget,
        dropped_count: dropped,
    }
}

// ---------------------------------------------------------------------------
// Session-aware packing (§9.5)
// ---------------------------------------------------------------------------

/// Pack candidates for a *continuation* within an existing session.
///
/// Only includes candidates whose `content_hash` is NOT in `sent_hashes`.
/// This avoids re-sending content the LLM already has in its context window.
pub fn pack_context_incremental(
    candidates: Vec<ContextCandidate>,
    token_budget: u32,
    sent_hashes: &std::collections::HashSet<u64>,
) -> PackedContext {
    let new_candidates: Vec<ContextCandidate> = candidates
        .into_iter()
        .filter(|c| !sent_hashes.contains(&c.content_hash))
        .collect();

    pack_context(new_candidates, token_budget)
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

/// Truncate content to approximately `max_tokens` tokens.
///
/// Tries to break at a line boundary to avoid splitting mid-line.
/// Appends a `[... truncated ...]` marker.
fn truncate_to_tokens(content: &str, max_tokens: u32) -> String {
    // Rough: 1 token ~ 4 bytes
    let max_bytes = (max_tokens as usize) * 4;

    if content.len() <= max_bytes {
        return content.to_string();
    }

    // Find the last newline before the byte limit (don't split mid-line)
    let search_region = &content[..max_bytes.min(content.len())];
    let break_point = search_region.rfind('\n').unwrap_or(max_bytes);

    if break_point == 0 {
        return String::new();
    }

    let mut truncated = content[..break_point].to_string();
    truncated.push_str("\n[... truncated ...]");
    truncated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::candidates::{ContextCandidate, ContextFeatures, ContextSource};
    use std::collections::HashSet;

    fn make_candidate(label: &str, content: &str, score_features: ContextFeatures) -> ContextCandidate {
        ContextCandidate::new(
            ContextSource::File,
            label,
            content,
            score_features,
        )
    }

    fn high_features() -> ContextFeatures {
        ContextFeatures {
            imported_by_target: true,
            mentioned_in_spec: true,
            same_language: true,
            ..Default::default()
        }
    }

    fn medium_features() -> ContextFeatures {
        ContextFeatures {
            same_directory: true,
            same_language: true,
            ..Default::default()
        }
    }

    fn low_features() -> ContextFeatures {
        ContextFeatures::default()
    }

    #[test]
    fn test_pack_empty_candidates() {
        let packed = pack_context(vec![], 1000);
        assert!(packed.is_empty());
        assert_eq!(packed.total_tokens, 0);
        assert_eq!(packed.dropped_count, 0);
    }

    #[test]
    fn test_pack_single_candidate_fits() {
        let c = make_candidate("a.rs", "fn main() {}", high_features());
        let packed = pack_context(vec![c], 1000);
        assert_eq!(packed.len(), 1);
        assert!(!packed.items[0].truncated);
        assert_eq!(packed.items[0].label, "a.rs");
    }

    #[test]
    fn test_pack_respects_budget() {
        // Create a large candidate (big content)
        let big_content = "x".repeat(8000); // ~2000 tokens
        let c = make_candidate("big.rs", &big_content, high_features());
        let packed = pack_context(vec![c], 100); // budget too small

        // Should be partially included (budget 100 > PARTIAL_INCLUSION_THRESHOLD of 200? No, 100 < 200)
        // So it should be dropped
        assert!(packed.is_empty());
        assert_eq!(packed.dropped_count, 1);
    }

    #[test]
    fn test_pack_partial_inclusion() {
        let big_content = "line one\nline two\nline three\nline four\n".repeat(100);
        let c = make_candidate("big.rs", &big_content, high_features());
        let packed = pack_context(vec![c], 300); // enough for partial

        assert_eq!(packed.len(), 1);
        assert!(packed.items[0].truncated);
        assert!(packed.items[0].content.contains("[... truncated ...]"));
    }

    #[test]
    fn test_pack_orders_by_score() {
        let c_high = make_candidate("high.rs", "fn high() {}", high_features());
        let c_low = make_candidate("low.rs", "fn low() {}", low_features());
        let c_med = make_candidate("med.rs", "fn med() {}", medium_features());

        let packed = pack_context(vec![c_low, c_high, c_med], 10000);
        assert_eq!(packed.len(), 3);
        // Highest score should come first
        assert_eq!(packed.items[0].label, "high.rs");
        assert!(packed.items[0].score >= packed.items[1].score);
        assert!(packed.items[1].score >= packed.items[2].score);
    }

    #[test]
    fn test_pack_greedy_fill() {
        // Create two candidates, budget only fits one
        let content = "x".repeat(400); // ~100 tokens each
        let c1 = make_candidate("a.rs", &content, high_features());
        let c2 = make_candidate("b.rs", &content, medium_features());

        // Budget fits only one candidate (100 tokens + 8 overhead = 108)
        let packed = pack_context(vec![c1, c2], 115);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed.items[0].label, "a.rs"); // higher score
        assert_eq!(packed.dropped_count, 1);
    }

    #[test]
    fn test_pack_render() {
        let c = make_candidate("main.rs", "fn main() {}", high_features());
        let packed = pack_context(vec![c], 1000);
        let rendered = packed.render();
        assert!(rendered.contains("--- [file] main.rs ---"));
        assert!(rendered.contains("fn main() {}"));
    }

    #[test]
    fn test_truncate_to_tokens_short_content() {
        let content = "short";
        let result = truncate_to_tokens(content, 100);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_truncate_to_tokens_breaks_at_line() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5";
        // 2 tokens = 8 bytes — should fit "line 1\n" (7 bytes) and break before "line 2"
        let result = truncate_to_tokens(content, 2);
        assert!(result.contains("line 1"));
        assert!(result.contains("[... truncated ...]"));
        assert!(!result.contains("line 5"));
    }

    #[test]
    fn test_incremental_packing_filters_sent() {
        let c1 = make_candidate("a.rs", "content_a", high_features());
        let c2 = make_candidate("b.rs", "content_b", medium_features());

        let mut sent = HashSet::new();
        sent.insert(c1.content_hash);

        let packed = pack_context_incremental(vec![c1, c2], 10000, &sent);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed.items[0].label, "b.rs");
    }

    #[test]
    fn test_incremental_packing_all_sent() {
        let c1 = make_candidate("a.rs", "content_a", high_features());
        let c2 = make_candidate("b.rs", "content_b", medium_features());

        let mut sent = HashSet::new();
        sent.insert(c1.content_hash);
        sent.insert(c2.content_hash);

        let packed = pack_context_incremental(vec![c1, c2], 10000, &sent);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_zero_budget() {
        let c = make_candidate("a.rs", "fn main() {}", high_features());
        let packed = pack_context(vec![c], 0);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_packed_context_total_tokens() {
        let content = "x".repeat(40); // 10 tokens
        let c = make_candidate("a.rs", &content, high_features());
        let packed = pack_context(vec![c], 1000);
        // Total tokens should be estimated_tokens + overhead
        assert!(packed.total_tokens > 0);
        assert!(packed.total_tokens <= 1000);
    }

    #[test]
    fn test_tie_breaking_prefers_smaller() {
        // Two candidates with identical features but different sizes
        let small_content = "x".repeat(40);
        let large_content = "x".repeat(4000);
        let c_small = make_candidate("small.rs", &small_content, high_features());
        let c_large = make_candidate("large.rs", &large_content, high_features());

        let packed = pack_context(vec![c_large, c_small], 10000);
        assert_eq!(packed.len(), 2);
        // Smaller should come first due to tie-breaking
        assert_eq!(packed.items[0].label, "small.rs");
    }
}
