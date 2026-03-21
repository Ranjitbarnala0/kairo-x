//! Search-and-replace edit protocol with 4-level fallback chain (§10).
//!
//! LLM output contains `<<<SEARCH ... >>> <<<REPLACE ... >>>` blocks.
//! We parse these and apply edits to file content with graceful degradation:
//!
//! 1. **Exact match** — byte-for-byte identical substring.
//! 2. **Whitespace-normalized** — strip leading/trailing whitespace per line,
//!    normalize internal runs of whitespace to single spaces.
//! 3. **Fuzzy match** — Levenshtein distance via the `similar` crate,
//!    accepting matches where distance < 20% of block length.
//! 4. **NoMatch** — all levels failed, return an error.
//!
//! Edits are applied sequentially with offset tracking so that earlier edits
//! don't invalidate the positions of later ones.

use similar::TextDiff;
use thiserror::Error;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single search-and-replace block parsed from LLM output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchReplaceBlock {
    /// The text to search for in the original content.
    pub search: String,
    /// The replacement text.
    pub replace: String,
}

/// Which fallback level succeeded for an edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchLevel {
    Exact,
    WhitespaceNormalized,
    Fuzzy { distance: usize },
}

/// Error applying an edit.
#[derive(Debug, Error)]
pub enum EditError {
    #[error("no match found for search block ({length} chars): {preview}...")]
    NoMatch { preview: String, length: usize },

    #[error("ambiguous match: search block matches {count} locations")]
    AmbiguousMatch { count: usize },

    #[error("no search/replace blocks found in input")]
    NoParsedBlocks,
}

/// Result of a single edit application, with provenance.
#[derive(Debug)]
pub struct EditApplication {
    /// Which fallback level was used.
    pub level: MatchLevel,
    /// Byte offset where the match was found in the original content.
    pub offset: usize,
    /// Length of the matched region in the original content.
    pub matched_len: usize,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse `<<<SEARCH ... >>> <<<REPLACE ... >>>` blocks from LLM output.
///
/// Supports the format:
/// ```text
/// <<<SEARCH
/// original code here
/// >>>
/// <<<REPLACE
/// replacement code here
/// >>>
/// ```
///
/// Returns an empty vec if no valid blocks are found.
pub fn parse_search_replace_blocks(response: &str) -> Vec<SearchReplaceBlock> {
    let mut blocks = Vec::new();
    let mut remaining = response;

    while let Some(search_start) = remaining.find("<<<SEARCH") {

        // Advance past the marker and optional newline.
        let after_marker = &remaining[search_start + "<<<SEARCH".len()..];
        let search_content_start = if after_marker.starts_with('\n') {
            1
        } else if after_marker.starts_with("\r\n") {
            2
        } else {
            0
        };
        let search_body = &after_marker[search_content_start..];

        // Find the closing >>>.
        let search_end = match search_body.find(">>>") {
            Some(pos) => pos,
            None => break,
        };

        // Extract search content, trimming trailing newline before >>>.
        let mut search_text = &search_body[..search_end];
        if search_text.ends_with('\n') {
            search_text = &search_text[..search_text.len() - 1];
            if search_text.ends_with('\r') {
                search_text = &search_text[..search_text.len() - 1];
            }
        }

        // Now find <<<REPLACE after the search block's >>>.
        let after_search_close = &search_body[search_end + ">>>".len()..];
        let replace_marker = match after_search_close.find("<<<REPLACE") {
            Some(pos) => pos,
            None => {
                remaining = after_search_close;
                continue;
            }
        };

        let after_replace_marker =
            &after_search_close[replace_marker + "<<<REPLACE".len()..];
        let replace_content_start = if after_replace_marker.starts_with('\n') {
            1
        } else if after_replace_marker.starts_with("\r\n") {
            2
        } else {
            0
        };
        let replace_body = &after_replace_marker[replace_content_start..];

        // Find the closing >>> for the replace block.
        let replace_end = match replace_body.find(">>>") {
            Some(pos) => pos,
            None => break,
        };

        let mut replace_text = &replace_body[..replace_end];
        if replace_text.ends_with('\n') {
            replace_text = &replace_text[..replace_text.len() - 1];
            if replace_text.ends_with('\r') {
                replace_text = &replace_text[..replace_text.len() - 1];
            }
        }

        blocks.push(SearchReplaceBlock {
            search: search_text.to_string(),
            replace: replace_text.to_string(),
        });

        // Advance past the replace block's closing >>>.
        remaining = &replace_body[replace_end + ">>>".len()..];
    }

    debug!(count = blocks.len(), "parsed search/replace blocks");
    blocks
}

// ---------------------------------------------------------------------------
// Whitespace normalization
// ---------------------------------------------------------------------------

/// Normalize whitespace in a string:
/// - Strip leading and trailing whitespace from each line.
/// - Collapse internal runs of whitespace to a single space.
/// - Preserve line structure (newlines are kept).
pub fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                String::new()
            } else {
                // Collapse internal whitespace runs.
                let mut result = String::with_capacity(trimmed.len());
                let mut prev_ws = false;
                for ch in trimmed.chars() {
                    if ch.is_whitespace() {
                        if !prev_ws {
                            result.push(' ');
                            prev_ws = true;
                        }
                    } else {
                        result.push(ch);
                        prev_ws = false;
                    }
                }
                result
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find a whitespace-normalized match of `needle` in `haystack`.
///
/// Returns `Some((start, end))` byte offsets in the *original* haystack
/// where the normalized needle matches. Works by normalizing both and
/// then mapping back to the original positions.
pub fn find_normalized(haystack: &str, needle: &str) -> Option<(usize, usize)> {
    let norm_needle = normalize_whitespace(needle);
    if norm_needle.is_empty() {
        return None;
    }

    // Split haystack into lines and try to find a contiguous span of lines
    // whose normalization matches the normalized needle lines.
    let needle_lines: Vec<&str> = norm_needle.lines().collect();
    let haystack_lines: Vec<&str> = haystack.lines().collect();

    if needle_lines.is_empty() || haystack_lines.is_empty() {
        return None;
    }

    let n_lines = needle_lines.len();
    if n_lines > haystack_lines.len() {
        return None;
    }

    // Normalize each haystack line for comparison.
    let norm_hay_lines: Vec<String> = haystack_lines
        .iter()
        .map(|l| normalize_whitespace(l))
        .collect();

    // Sliding window: find contiguous lines that match.
    'outer: for start_line in 0..=(haystack_lines.len() - n_lines) {
        for (i, needle_line) in needle_lines.iter().enumerate() {
            if norm_hay_lines[start_line + i] != *needle_line {
                continue 'outer;
            }
        }

        // Found a match. Compute byte offsets in the original haystack.
        let byte_start = haystack_lines[..start_line]
            .iter()
            .map(|l| l.len() + 1) // +1 for newline
            .sum::<usize>();
        let end_line = start_line + n_lines - 1;
        let byte_end = haystack_lines[..end_line]
            .iter()
            .map(|l| l.len() + 1)
            .sum::<usize>()
            + haystack_lines[end_line].len();

        // Ensure byte_end doesn't exceed haystack length.
        let byte_end = byte_end.min(haystack.len());

        return Some((byte_start, byte_end));
    }

    None
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/// Find a fuzzy match of `needle` in `haystack` using Levenshtein distance.
///
/// Scans the haystack with a sliding window (sized around the needle length)
/// and finds the window with the lowest edit distance. Accepts the match only
/// if `distance < max_distance_ratio * needle_len`.
///
/// Returns `Some((start, end, distance))` byte offsets and edit distance,
/// or `None` if no acceptable match is found.
pub fn find_fuzzy(
    haystack: &str,
    needle: &str,
    max_distance_ratio: f64,
) -> Option<(usize, usize, usize)> {
    if needle.is_empty() || haystack.is_empty() {
        return None;
    }

    let needle_lines: Vec<&str> = needle.lines().collect();
    let haystack_lines: Vec<&str> = haystack.lines().collect();
    let n_lines = needle_lines.len();

    if n_lines > haystack_lines.len() {
        return None;
    }

    let max_distance = (max_distance_ratio * needle.len() as f64).ceil() as usize;
    let mut best: Option<(usize, usize, usize)> = None;

    // Try windows of size n_lines, n_lines-1, n_lines+1 to handle minor
    // line count differences.
    let window_sizes: Vec<usize> = {
        let mut sizes = vec![n_lines];
        if n_lines > 1 {
            sizes.push(n_lines - 1);
        }
        if n_lines < haystack_lines.len() {
            sizes.push(n_lines + 1);
        }
        sizes
    };

    for &win_size in &window_sizes {
        if win_size > haystack_lines.len() {
            continue;
        }

        for start in 0..=(haystack_lines.len() - win_size) {
            let window_text = haystack_lines[start..start + win_size].join("\n");
            let diff = TextDiff::from_chars(needle, &window_text);
            let distance = diff
                .ops()
                .iter()
                .map(|op| match op {
                    similar::DiffOp::Equal { .. } => 0,
                    similar::DiffOp::Insert { new_index, .. } => {
                        window_text[*new_index..].chars().count().min(
                            op.new_range().end - op.new_range().start,
                        )
                    }
                    similar::DiffOp::Delete { old_index, .. } => {
                        needle[*old_index..].chars().count().min(
                            op.old_range().end - op.old_range().start,
                        )
                    }
                    similar::DiffOp::Replace {
                        old_index: _,
                        new_index: _,
                        ..
                    } => {
                        let old_len = op.old_range().end - op.old_range().start;
                        let new_len = op.new_range().end - op.new_range().start;
                        old_len.max(new_len)
                    }
                })
                .sum::<usize>();

            if distance <= max_distance {
                let is_better = best
                    .as_ref()
                    .map(|b| distance < b.2)
                    .unwrap_or(true);
                if is_better {
                    // Compute byte offsets.
                    let byte_start: usize = haystack_lines[..start]
                        .iter()
                        .map(|l| l.len() + 1)
                        .sum();
                    let end_line = start + win_size - 1;
                    let byte_end: usize = haystack_lines[..end_line]
                        .iter()
                        .map(|l| l.len() + 1)
                        .sum::<usize>()
                        + haystack_lines[end_line].len();
                    let byte_end = byte_end.min(haystack.len());

                    best = Some((byte_start, byte_end, distance));

                    // Perfect match — short-circuit.
                    if distance == 0 {
                        return best;
                    }
                }
            }
        }
    }

    if let Some((s, e, d)) = best {
        trace!(distance = d, start = s, end = e, "fuzzy match found");
    }
    best
}

// ---------------------------------------------------------------------------
// Apply edits
// ---------------------------------------------------------------------------

/// Apply a sequence of search/replace edits to file content.
///
/// Uses the 4-level fallback chain for each edit:
/// 1. Exact match
/// 2. Whitespace-normalized match
/// 3. Fuzzy match (Levenshtein distance < 20% of search block length)
/// 4. Return `EditError::NoMatch`
///
/// Edits are applied sequentially, with offset tracking so that earlier
/// replacements don't invalidate the byte positions of later ones.
pub fn apply_edits(
    content: &str,
    edits: &[SearchReplaceBlock],
) -> Result<(String, Vec<EditApplication>), EditError> {
    if edits.is_empty() {
        return Err(EditError::NoParsedBlocks);
    }

    let mut result = content.to_string();
    let mut applications = Vec::with_capacity(edits.len());

    for edit in edits {
        let applied = apply_single_edit(&result, edit)?;
        // Perform the replacement.
        let new_result = format!(
            "{}{}{}",
            &result[..applied.offset],
            &edit.replace,
            &result[applied.offset + applied.matched_len..],
        );
        debug!(
            level = ?applied.level,
            offset = applied.offset,
            matched_len = applied.matched_len,
            "applied edit"
        );
        result = new_result;
        applications.push(applied);
    }

    Ok((result, applications))
}

/// Apply a single edit using the 4-level fallback chain.
fn apply_single_edit(
    content: &str,
    edit: &SearchReplaceBlock,
) -> Result<EditApplication, EditError> {
    // Guard: empty search block would match at position 0, which is
    // almost certainly not what the LLM intended.
    if edit.search.is_empty() {
        return Err(EditError::NoMatch {
            preview: String::new(),
            length: 0,
        });
    }

    // Level 1: Exact match.
    if let Some(offset) = content.find(&edit.search) {
        // Verify uniqueness — check there isn't a second occurrence.
        if let Some(second) = content[offset + edit.search.len()..].find(&edit.search) {
            let _ = second; // suppress unused warning
            // Multiple exact matches. We still use the first one but log a warning.
            warn!("multiple exact matches found; using first occurrence");
        }
        return Ok(EditApplication {
            level: MatchLevel::Exact,
            offset,
            matched_len: edit.search.len(),
        });
    }

    // Level 2: Whitespace-normalized match.
    if let Some((start, end)) = find_normalized(content, &edit.search) {
        debug!("falling back to whitespace-normalized match");
        return Ok(EditApplication {
            level: MatchLevel::WhitespaceNormalized,
            offset: start,
            matched_len: end - start,
        });
    }

    // Level 3: Fuzzy match (distance < 20% of search block length).
    //
    // SAFETY GUARD: Only attempt fuzzy matching if the search block has at
    // least 3 lines. Short search blocks (e.g., a single `return true;`) are
    // too generic and could match the wrong location in large files. The LLM
    // will be asked to retry with more context instead.
    let search_line_count = edit.search.lines().count();
    if search_line_count >= 3 {
        if let Some((start, end, distance)) = find_fuzzy(content, &edit.search, 0.20) {
            // SAFETY GUARD: Verify uniqueness — if there are multiple plausible
            // fuzzy matches, reject to avoid modifying the wrong location.
            // Check by also searching from the end of the first match.
            let remainder = &content[end..];
            if let Some((_, _, dist2)) = find_fuzzy(remainder, &edit.search, 0.20) {
                warn!(
                    first_distance = distance,
                    second_distance = dist2,
                    "fuzzy match found multiple plausible locations — rejecting for safety"
                );
                // Fall through to NoMatch so the LLM is asked to retry with more context
            } else {
                debug!(distance, "falling back to fuzzy match");
                return Ok(EditApplication {
                    level: MatchLevel::Fuzzy { distance },
                    offset: start,
                    matched_len: end - start,
                });
            }
        }
    } else {
        debug!(
            lines = search_line_count,
            "skipping fuzzy match — search block too short (need >= 3 lines)"
        );
    }

    // Level 4: No match.
    let preview: String = edit.search.chars().take(80).collect();
    Err(EditError::NoMatch {
        preview,
        length: edit.search.len(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Parsing tests --

    #[test]
    fn test_parse_single_block() {
        let input = r#"Here is the edit:
<<<SEARCH
fn old_function() {
    println!("old");
}
>>>
<<<REPLACE
fn new_function() {
    println!("new");
}
>>>
Done."#;

        let blocks = parse_search_replace_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].search,
            "fn old_function() {\n    println!(\"old\");\n}"
        );
        assert_eq!(
            blocks[0].replace,
            "fn new_function() {\n    println!(\"new\");\n}"
        );
    }

    #[test]
    fn test_parse_multiple_blocks() {
        let input = r#"
<<<SEARCH
line_a
>>>
<<<REPLACE
line_a_new
>>>

Some text between blocks.

<<<SEARCH
line_b
>>>
<<<REPLACE
line_b_new
>>>
"#;

        let blocks = parse_search_replace_blocks(input);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].search, "line_a");
        assert_eq!(blocks[0].replace, "line_a_new");
        assert_eq!(blocks[1].search, "line_b");
        assert_eq!(blocks[1].replace, "line_b_new");
    }

    #[test]
    fn test_parse_empty_replace() {
        let input = r#"<<<SEARCH
delete_this_line
>>>
<<<REPLACE
>>>
"#;
        let blocks = parse_search_replace_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].search, "delete_this_line");
        assert_eq!(blocks[0].replace, "");
    }

    #[test]
    fn test_parse_no_blocks() {
        let blocks = parse_search_replace_blocks("just regular text, no markers");
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_parse_incomplete_block() {
        let input = "<<<SEARCH\nhello\n>>>\n"; // No REPLACE block
        let blocks = parse_search_replace_blocks(input);
        assert!(blocks.is_empty());
    }

    // -- Normalization tests --

    #[test]
    fn test_normalize_whitespace_basic() {
        assert_eq!(
            normalize_whitespace("  hello   world  "),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_whitespace_multiline() {
        let input = "  fn foo()  {\n    let   x =  1;\n  }";
        let expected = "fn foo() {\nlet x = 1;\n}";
        assert_eq!(normalize_whitespace(input), expected);
    }

    #[test]
    fn test_normalize_whitespace_empty_lines() {
        let input = "  hello\n\n  world  ";
        let expected = "hello\n\nworld";
        assert_eq!(normalize_whitespace(input), expected);
    }

    #[test]
    fn test_normalize_whitespace_tabs() {
        let input = "\t\thello\t\tworld\t";
        assert_eq!(normalize_whitespace(input), "hello world");
    }

    // -- find_normalized tests --

    #[test]
    fn test_find_normalized_match() {
        let haystack = "fn main() {\n    let  x  = 1;\n    println!(\"hi\");\n}";
        let needle = "let x = 1;\nprintln!(\"hi\");";

        let result = find_normalized(haystack, needle);
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        // The matched region should cover the two middle lines.
        let matched = &haystack[start..end];
        assert!(matched.contains("let"));
        assert!(matched.contains("println!"));
    }

    #[test]
    fn test_find_normalized_no_match() {
        let haystack = "fn main() {\n    let x = 1;\n}";
        let needle = "let y = 2;";
        assert!(find_normalized(haystack, needle).is_none());
    }

    // -- Fuzzy match tests --

    #[test]
    fn test_find_fuzzy_close_match() {
        let haystack = "fn calculate_total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.price).sum()\n}";
        // Slightly different: "calcualte" typo, "price" vs "prce"
        let needle = "fn calcualte_total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.prce).sum()\n}";

        let result = find_fuzzy(haystack, needle, 0.20);
        assert!(result.is_some());
        let (_, _, distance) = result.unwrap();
        assert!(distance > 0);
        assert!(distance < (needle.len() as f64 * 0.20) as usize + 1);
    }

    #[test]
    fn test_find_fuzzy_too_different() {
        let haystack = "fn main() {\n    println!(\"hello\");\n}";
        let needle = "completely different code that has nothing in common";

        let result = find_fuzzy(haystack, needle, 0.20);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_fuzzy_exact() {
        let haystack = "line one\nline two\nline three";
        let needle = "line two";

        let result = find_fuzzy(haystack, needle, 0.20);
        assert!(result.is_some());
        let (_, _, distance) = result.unwrap();
        assert_eq!(distance, 0);
    }

    // -- Apply edits tests --

    #[test]
    fn test_apply_exact_match() {
        let content = "fn main() {\n    println!(\"hello\");\n}";
        let edits = vec![SearchReplaceBlock {
            search: "println!(\"hello\")".to_string(),
            replace: "println!(\"world\")".to_string(),
        }];

        let (result, apps) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("println!(\"world\")"));
        assert!(!result.contains("println!(\"hello\")"));
        assert_eq!(apps[0].level, MatchLevel::Exact);
    }

    #[test]
    fn test_apply_whitespace_normalized_match() {
        let content = "fn main() {\n    let   x  =  1;\n    let y = 2;\n}";
        // Search with different whitespace.
        let edits = vec![SearchReplaceBlock {
            search: "let x = 1;".to_string(),
            replace: "let x = 42;".to_string(),
        }];

        let (result, apps) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("let x = 42;"));
        assert_eq!(apps[0].level, MatchLevel::WhitespaceNormalized);
    }

    #[test]
    fn test_apply_fuzzy_match() {
        let content = "fn calculate_total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.price).sum()\n}\n";
        // Search with a small typo — "calcualte" instead of "calculate".
        let edits = vec![SearchReplaceBlock {
            search: "fn calcualte_total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.price).sum()\n}".to_string(),
            replace: "fn calculate_total(items: &[Item]) -> u64 {\n    items.iter().map(|i| i.price as u64).sum()\n}".to_string(),
        }];

        let (result, apps) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("-> u64"));
        assert!(matches!(apps[0].level, MatchLevel::Fuzzy { .. }));
    }

    #[test]
    fn test_apply_no_match() {
        let content = "fn main() {}";
        let edits = vec![SearchReplaceBlock {
            search: "this text does not exist anywhere in content at all whatsoever".to_string(),
            replace: "replacement".to_string(),
        }];

        let result = apply_edits(content, &edits);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EditError::NoMatch { .. }));
    }

    #[test]
    fn test_apply_multiple_edits_sequential() {
        let content = "let a = 1;\nlet b = 2;\nlet c = 3;";
        let edits = vec![
            SearchReplaceBlock {
                search: "let a = 1;".to_string(),
                replace: "let a = 10;".to_string(),
            },
            SearchReplaceBlock {
                search: "let c = 3;".to_string(),
                replace: "let c = 30;".to_string(),
            },
        ];

        let (result, apps) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("let a = 10;"));
        assert!(result.contains("let b = 2;"));
        assert!(result.contains("let c = 30;"));
        assert_eq!(apps.len(), 2);
    }

    #[test]
    fn test_apply_empty_edits() {
        let content = "hello";
        let result = apply_edits(content, &[]);
        assert!(matches!(result.unwrap_err(), EditError::NoParsedBlocks));
    }

    #[test]
    fn test_apply_empty_search_block() {
        let content = "fn main() {}";
        let edits = vec![SearchReplaceBlock {
            search: String::new(),
            replace: "inserted".to_string(),
        }];
        let result = apply_edits(content, &edits);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EditError::NoMatch { .. }));
    }

    #[test]
    fn test_apply_delete_edit() {
        let content = "line1\ndelete_me\nline3";
        let edits = vec![SearchReplaceBlock {
            search: "delete_me\n".to_string(),
            replace: String::new(),
        }];

        let (result, _) = apply_edits(content, &edits).unwrap();
        assert_eq!(result, "line1\nline3");
    }

    #[test]
    fn test_apply_insert_via_empty_context() {
        // Use a search that matches the insertion point.
        let content = "fn main() {\n}";
        let edits = vec![SearchReplaceBlock {
            search: "fn main() {\n}".to_string(),
            replace: "fn main() {\n    println!(\"inserted\");\n}".to_string(),
        }];

        let (result, _) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("println!(\"inserted\")"));
    }

    // -- Round-trip: parse then apply --

    #[test]
    fn test_round_trip_parse_and_apply() {
        let file_content = "pub fn greet(name: &str) {\n    println!(\"Hello, {name}\");\n}";
        let llm_response = r#"I'll update the greeting function:

<<<SEARCH
pub fn greet(name: &str) {
    println!("Hello, {name}");
}
>>>
<<<REPLACE
pub fn greet(name: &str) {
    println!("Welcome, {name}!");
}
>>>
"#;

        let blocks = parse_search_replace_blocks(llm_response);
        assert_eq!(blocks.len(), 1);

        let (result, apps) = apply_edits(file_content, &blocks).unwrap();
        assert!(result.contains("Welcome, {name}!"));
        assert!(!result.contains("Hello, {name}"));
        assert_eq!(apps[0].level, MatchLevel::Exact);
    }

    #[test]
    fn test_round_trip_whitespace_fallback() {
        let file_content = "    fn  compute(x:  i32)  ->  i32  {\n        x  *  2\n    }";
        let llm_response = r#"
<<<SEARCH
fn compute(x: i32) -> i32 {
    x * 2
}
>>>
<<<REPLACE
fn compute(x: i32) -> i32 {
    x * 3
}
>>>
"#;

        let blocks = parse_search_replace_blocks(llm_response);
        let (result, apps) = apply_edits(file_content, &blocks).unwrap();
        assert!(result.contains("x * 3"));
        assert_eq!(apps[0].level, MatchLevel::WhitespaceNormalized);
    }

    #[test]
    fn test_multiline_search_preserves_structure() {
        let content = "use std::io;\nuse std::fs;\n\nfn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{x} {y}\");\n}\n";
        let edits = vec![SearchReplaceBlock {
            search: "    let x = 1;\n    let y = 2;".to_string(),
            replace: "    let x = 10;\n    let y = 20;".to_string(),
        }];

        let (result, _) = apply_edits(content, &edits).unwrap();
        assert!(result.contains("let x = 10;"));
        assert!(result.contains("let y = 20;"));
        assert!(result.contains("use std::io;"));
        assert!(result.contains("println!"));
    }
}
