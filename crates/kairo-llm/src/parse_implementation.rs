//! Implementation response parser.
//!
//! Parses LLM implementation responses into structured `ParsedImplementation`
//! containing file modifications (new files, edits via SEARCH/REPLACE blocks,
//! or full rewrites).
//!
//! Supports two patterns:
//! 1. Fenced code blocks with file path comments on the opening line
//! 2. SEARCH/REPLACE blocks for editing existing files

use crate::response::{FileModification, ParsedImplementation, SearchReplaceBlock};
use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Regex patterns
// ---------------------------------------------------------------------------

/// Matches a fenced code block with an optional file path comment on the first line.
/// Captures: language (optional), file path, and content.
///
/// Example patterns matched (triple-backtick fenced blocks with path comments):
///   \`\`\`rust // path/to/file.rs
///   \`\`\`python # path/to/file.py
///   \`\`\`js // path/to/file.js
static CODE_BLOCK_WITH_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)```(\w+)?\s*(?://|#)\s*(\S+\.[\w.]+)\s*\n([\s\S]*?)```"
    ).expect("code block regex must compile")
});

/// Matches SEARCH/REPLACE blocks.
///
/// Pattern:
///   <<<SEARCH
///   exact existing code
///   >>>
///   <<<REPLACE
///   replacement code
///   >>>
static SEARCH_REPLACE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)<<<SEARCH\n([\s\S]*?)\n>>>\n<<<REPLACE\n([\s\S]*?)\n>>>"
    ).expect("search/replace regex must compile")
});

/// Matches a file path header before SEARCH/REPLACE blocks.
///
/// Pattern (case-insensitive):
///   File: path/to/file.rs
///   file: `path/to/file.rs`
static FILE_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?im)^(?:file|in|editing|modify):\s*`?(\S+\.[\w.]+)`?\s*$"
    ).expect("file header regex must compile")
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse an LLM implementation response into structured file modifications.
///
/// Detects:
/// - Code blocks with file paths (```lang // path/to/file.ext)
/// - SEARCH/REPLACE blocks (<<<SEARCH ... >>> <<<REPLACE ... >>>)
///
/// Returns a `ParsedImplementation` with the detected modifications and any
/// explanatory text outside code blocks.
pub fn parse_implementation(response: &str) -> ParsedImplementation {
    let mut modifications: Vec<FileModification> = Vec::new();
    let mut explanation_parts: Vec<&str> = Vec::new();

    // Track which byte ranges are consumed by code blocks / search-replace
    let mut consumed_ranges: Vec<(usize, usize)> = Vec::new();

    // 1. Parse SEARCH/REPLACE blocks grouped by file header
    parse_search_replace_blocks(response, &mut modifications, &mut consumed_ranges);

    // 2. Parse code blocks with file paths
    parse_code_blocks(response, &mut modifications, &mut consumed_ranges);

    // 3. Collect explanation text (everything not consumed)
    consumed_ranges.sort_by_key(|&(start, _)| start);
    let mut pos = 0;
    for &(start, end) in &consumed_ranges {
        if start > pos {
            let text = response[pos..start].trim();
            if !text.is_empty() {
                explanation_parts.push(text);
            }
        }
        pos = end;
    }
    if pos < response.len() {
        let text = response[pos..].trim();
        if !text.is_empty() {
            explanation_parts.push(text);
        }
    }

    ParsedImplementation {
        modifications,
        explanation: explanation_parts.join("\n\n"),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse SEARCH/REPLACE blocks, associating each group with the nearest
/// preceding file path header.
fn parse_search_replace_blocks(
    response: &str,
    modifications: &mut Vec<FileModification>,
    consumed_ranges: &mut Vec<(usize, usize)>,
) {
    // Collect all file headers with their positions
    let file_headers: Vec<(usize, usize, String)> = FILE_HEADER_RE
        .captures_iter(response)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            let path = caps.get(1)?.as_str().to_string();
            Some((m.start(), m.end(), path))
        })
        .collect();

    // Collect all SEARCH/REPLACE blocks with their positions
    let sr_blocks: Vec<(usize, usize, String, String)> = SEARCH_REPLACE_RE
        .captures_iter(response)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            let search = caps.get(1)?.as_str().to_string();
            let replace = caps.get(2)?.as_str().to_string();
            Some((m.start(), m.end(), search, replace))
        })
        .collect();

    if sr_blocks.is_empty() {
        return;
    }

    // Group SR blocks by the nearest preceding file header
    // Use a map from file path -> Vec<SearchReplaceBlock>
    let mut file_edits: Vec<(String, Vec<SearchReplaceBlock>)> = Vec::new();

    for &(sr_start, sr_end, ref search, ref replace) in &sr_blocks {
        consumed_ranges.push((sr_start, sr_end));

        // Find the nearest file header before this block
        let associated_path = file_headers
            .iter()
            .rfind(|&&(_, hdr_end, _)| hdr_end <= sr_start)
            .map(|(hdr_start, hdr_end, path)| {
                // Also consume the header
                consumed_ranges.push((*hdr_start, *hdr_end));
                path.clone()
            })
            .unwrap_or_else(|| "unknown".to_string());

        let block = SearchReplaceBlock {
            search: search.clone(),
            replace: replace.clone(),
        };

        // Append to existing group or create new one
        if let Some(entry) = file_edits.iter_mut().find(|(p, _)| *p == associated_path) {
            entry.1.push(block);
        } else {
            file_edits.push((associated_path, vec![block]));
        }
    }

    for (path, edits) in file_edits {
        modifications.push(FileModification::EditFile { path, edits });
    }
}

/// Parse fenced code blocks that include a file path comment on the opening line.
fn parse_code_blocks(
    response: &str,
    modifications: &mut Vec<FileModification>,
    consumed_ranges: &mut Vec<(usize, usize)>,
) {
    for caps in CODE_BLOCK_WITH_PATH_RE.captures_iter(response) {
        let full_match = caps.get(0).unwrap();
        let path = caps.get(2).unwrap().as_str().to_string();
        let content = caps.get(3).unwrap().as_str().to_string();

        consumed_ranges.push((full_match.start(), full_match.end()));

        // If the content contains SEARCH/REPLACE markers, skip (already handled)
        if content.contains("<<<SEARCH") {
            continue;
        }

        // Determine if this is a new file creation or a full rewrite.
        // Heuristic: if the content looks like a complete file (has function/struct/class
        // definitions), treat it as a CreateFile.
        modifications.push(FileModification::CreateFile {
            path,
            content: content.trim_end().to_string(),
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_code_block_with_path() {
        let response = r#"Here is the implementation:

```rust // src/lib.rs
fn hello() {
    println!("hello");
}
```

That should work.
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::CreateFile { path, content } => {
                assert_eq!(path, "src/lib.rs");
                assert!(content.contains("fn hello()"));
            }
            other => panic!("expected CreateFile, got {:?}", other),
        }
        assert!(parsed.explanation.contains("Here is the implementation"));
        assert!(parsed.explanation.contains("That should work"));
    }

    #[test]
    fn test_parse_search_replace_blocks() {
        let response = r#"I'll fix the issue.

File: src/main.rs

<<<SEARCH
fn old_code() {
    println!("old");
}
>>>
<<<REPLACE
fn new_code() {
    println!("new");
}
>>>

Done.
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::EditFile { path, edits } => {
                assert_eq!(path, "src/main.rs");
                assert_eq!(edits.len(), 1);
                assert!(edits[0].search.contains("old_code"));
                assert!(edits[0].replace.contains("new_code"));
            }
            other => panic!("expected EditFile, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_multiple_search_replace_same_file() {
        let response = r#"File: src/lib.rs

<<<SEARCH
fn a() {}
>>>
<<<REPLACE
fn a_fixed() {}
>>>

<<<SEARCH
fn b() {}
>>>
<<<REPLACE
fn b_fixed() {}
>>>
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::EditFile { path, edits } => {
                assert_eq!(path, "src/lib.rs");
                assert_eq!(edits.len(), 2);
            }
            other => panic!("expected EditFile, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_mixed_code_blocks_and_search_replace() {
        let response = r#"Create a new file:

```rust // src/new_module.rs
pub fn new_func() -> bool {
    true
}
```

And edit an existing file:

File: src/main.rs

<<<SEARCH
use old_module;
>>>
<<<REPLACE
use new_module;
>>>
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 2);

        let has_create = parsed.modifications.iter().any(|m| matches!(m, FileModification::CreateFile { path, .. } if path == "src/new_module.rs"));
        let has_edit = parsed.modifications.iter().any(|m| matches!(m, FileModification::EditFile { path, .. } if path == "src/main.rs"));

        assert!(has_create, "should have a CreateFile for src/new_module.rs");
        assert!(has_edit, "should have an EditFile for src/main.rs");
    }

    #[test]
    fn test_parse_no_modifications() {
        let response = "I cannot implement this without more context about the database schema.";
        let parsed = parse_implementation(response);
        assert!(parsed.modifications.is_empty());
        assert!(!parsed.explanation.is_empty());
    }

    #[test]
    fn test_parse_python_file_path() {
        let response = r#"```python # scripts/deploy.py
def deploy():
    pass
```
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::CreateFile { path, .. } => {
                assert_eq!(path, "scripts/deploy.py");
            }
            other => panic!("expected CreateFile, got {:?}", other),
        }
    }
}
