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
// Path validation
// ---------------------------------------------------------------------------

/// Validate an extracted file path for safety. Rejects absolute paths, path
/// traversal (`..`), and empty strings.
fn validate_file_path(path: &str) -> Result<String, String> {
    let path = path.trim();
    // Reject empty paths
    if path.is_empty() {
        return Err("empty path".to_string());
    }
    // Reject absolute paths (Unix and Windows)
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(format!("absolute path not allowed: {}", path));
    }
    // Reject Windows drive-letter absolute paths (e.g. C:\)
    if path.len() >= 2 && path.as_bytes()[1] == b':' && path.as_bytes()[0].is_ascii_alphabetic() {
        return Err(format!("absolute path not allowed: {}", path));
    }
    // Reject path traversal via any `..` component
    for component in std::path::Path::new(path).components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(format!("path traversal not allowed: {}", path));
        }
    }
    Ok(path.to_string())
}

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
        r"(?m)```(\w+)?\s*(?://|#)\s*(\S+\.[\w.]+)\s*\r?\n([\s\S]*?)```"
    ).expect("code block regex must compile")
});

/// Matches SEARCH/REPLACE blocks. Handles both `\n` and `\r\n` line endings.
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
        r"(?m)<<<SEARCH\r?\n([\s\S]*?)\r?\n>>>\r?\n<<<REPLACE\r?\n([\s\S]*?)\r?\n>>>"
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

/// Matches phrases near a code block that hint at file creation rather than rewrite.
/// Used by the CreateFile vs RewriteFile heuristic.
static CREATE_HINT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:create|new file|add(?:ing)?\s+(?:a\s+)?(?:new\s+)?file|write(?:ing)?\s+(?:a\s+)?(?:new\s+)?file)"
    ).expect("create hint regex must compile")
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

/// Parse SEARCH/REPLACE blocks, associating each with the file header whose
/// *section* it falls within.
///
/// Section-based association: each file header "owns" the region from its
/// position until the next file header (or end-of-string). An SR block is
/// associated with whichever section it starts in. This correctly handles
/// interleaved patterns where multiple files have SR blocks interspersed with
/// commentary.
fn parse_search_replace_blocks(
    response: &str,
    modifications: &mut Vec<FileModification>,
    consumed_ranges: &mut Vec<(usize, usize)>,
) {
    // Collect all file headers with their byte positions
    let file_headers: Vec<(usize, usize, String)> = FILE_HEADER_RE
        .captures_iter(response)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            let path = caps.get(1)?.as_str().to_string();
            Some((m.start(), m.end(), path))
        })
        .collect();

    // Collect all SEARCH/REPLACE blocks with their byte positions
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

    // Build sections: each header owns [header_start .. next_header_start).
    // The last header owns [header_start .. response.len()).
    // SR blocks before any header get associated with "unknown".
    struct Section {
        header_start: usize,
        header_end: usize,
        section_end: usize,
        path: String,
    }
    let sections: Vec<Section> = file_headers
        .iter()
        .enumerate()
        .map(|(i, &(hdr_start, hdr_end, ref path))| {
            let section_end = file_headers
                .get(i + 1)
                .map(|&(next_start, _, _)| next_start)
                .unwrap_or(response.len());
            Section {
                header_start: hdr_start,
                header_end: hdr_end,
                section_end,
                path: path.clone(),
            }
        })
        .collect();

    // Group SR blocks by owning section
    let mut file_edits: Vec<(String, Vec<SearchReplaceBlock>)> = Vec::new();
    // Track which headers we've consumed (deduplicate)
    let mut consumed_headers: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for &(sr_start, sr_end, ref search, ref replace) in &sr_blocks {
        consumed_ranges.push((sr_start, sr_end));

        // Find the section that contains this SR block: the section whose
        // range [header_start .. section_end) includes sr_start.
        let associated_path = sections
            .iter()
            .rfind(|sec| sr_start >= sec.header_start && sr_start < sec.section_end)
            .map(|sec| {
                // Consume the header (only once)
                if consumed_headers.insert(sec.header_start) {
                    consumed_ranges.push((sec.header_start, sec.header_end));
                }
                sec.path.clone()
            })
            .unwrap_or_else(|| "unknown".to_string());

        // Validate the path before using it
        let validated_path = match validate_file_path(&associated_path) {
            Ok(p) => p,
            Err(reason) => {
                tracing::warn!(
                    path = %associated_path,
                    reason = %reason,
                    "skipping SEARCH/REPLACE block with invalid file path"
                );
                continue;
            }
        };

        let block = SearchReplaceBlock {
            search: search.clone(),
            replace: replace.clone(),
        };

        // Append to existing group or create new one
        if let Some(entry) = file_edits.iter_mut().find(|(p, _)| *p == validated_path) {
            entry.1.push(block);
        } else {
            file_edits.push((validated_path, vec![block]));
        }
    }

    for (path, edits) in file_edits {
        modifications.push(FileModification::EditFile { path, edits });
    }
}

/// Parse fenced code blocks that include a file path comment on the opening line.
///
/// Applies a CreateFile vs RewriteFile heuristic:
/// - **CreateFile** when the surrounding text explicitly says "create", "new file",
///   "add a new file", "write a new file", etc.
/// - **RewriteFile** otherwise (the common case where the LLM is showing the full
///   updated content of an existing file).
fn parse_code_blocks(
    response: &str,
    modifications: &mut Vec<FileModification>,
    consumed_ranges: &mut Vec<(usize, usize)>,
) {
    for caps in CODE_BLOCK_WITH_PATH_RE.captures_iter(response) {
        let full_match = caps.get(0).unwrap();
        let raw_path = caps.get(2).unwrap().as_str().to_string();
        let content = caps.get(3).unwrap().as_str().to_string();

        consumed_ranges.push((full_match.start(), full_match.end()));

        // If the content contains SEARCH/REPLACE markers, skip (already handled)
        if content.contains("<<<SEARCH") {
            continue;
        }

        // Validate the extracted file path
        let path = match validate_file_path(&raw_path) {
            Ok(p) => p,
            Err(reason) => {
                tracing::warn!(
                    path = %raw_path,
                    reason = %reason,
                    "skipping code block with invalid file path"
                );
                continue;
            }
        };

        let trimmed_content = content.trim_end().to_string();

        // CreateFile vs RewriteFile heuristic:
        // Look at the text in the ~200 chars preceding this code block for
        // creation-related language. If found, treat as CreateFile; otherwise
        // assume the LLM is showing a full rewrite of an existing file.
        let context_start = full_match.start().saturating_sub(200);
        let preceding_text = &response[context_start..full_match.start()];
        let is_new_file = CREATE_HINT_RE.is_match(preceding_text);

        if is_new_file {
            modifications.push(FileModification::CreateFile {
                path,
                content: trimmed_content,
            });
        } else {
            modifications.push(FileModification::RewriteFile {
                path,
                content: trimmed_content,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Code-block parsing
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_code_block_rewrite_default() {
        // No "create"/"new file" hint => produces RewriteFile
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
            FileModification::RewriteFile { path, content } => {
                assert_eq!(path, "src/lib.rs");
                assert!(content.contains("fn hello()"));
            }
            other => panic!("expected RewriteFile, got {:?}", other),
        }
        assert!(parsed.explanation.contains("Here is the implementation"));
        assert!(parsed.explanation.contains("That should work"));
    }

    #[test]
    fn test_parse_code_block_create_with_hint() {
        // "Create a new file" hint => produces CreateFile
        let response = r#"Create a new file:

```rust // src/brand_new.rs
pub fn init() {}
```
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::CreateFile { path, content } => {
                assert_eq!(path, "src/brand_new.rs");
                assert!(content.contains("pub fn init()"));
            }
            other => panic!("expected CreateFile, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------
    // SEARCH/REPLACE parsing
    // -------------------------------------------------------------------

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
    fn test_parse_search_replace_crlf() {
        // Verify that CRLF line-endings are handled correctly
        let response =
            "File: src/main.rs\r\n\r\n<<<SEARCH\r\nfn old() {}\r\n>>>\r\n<<<REPLACE\r\nfn new() {}\r\n>>>\r\n";
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::EditFile { path, edits } => {
                assert_eq!(path, "src/main.rs");
                assert_eq!(edits.len(), 1);
                assert!(edits[0].search.contains("fn old()"));
                assert!(edits[0].replace.contains("fn new()"));
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

    // -------------------------------------------------------------------
    // Section-based file header association
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_interleaved_file_sections() {
        // Two file sections with SR blocks interleaved with commentary
        let response = r#"File: src/alpha.rs

Some explanation about alpha changes.

<<<SEARCH
fn alpha_old() {}
>>>
<<<REPLACE
fn alpha_new() {}
>>>

File: src/beta.rs

Now fixing beta:

<<<SEARCH
fn beta_old() {}
>>>
<<<REPLACE
fn beta_new() {}
>>>
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 2);

        let alpha = parsed.modifications.iter().find(|m| {
            matches!(m, FileModification::EditFile { path, .. } if path == "src/alpha.rs")
        });
        let beta = parsed.modifications.iter().find(|m| {
            matches!(m, FileModification::EditFile { path, .. } if path == "src/beta.rs")
        });

        assert!(alpha.is_some(), "should have EditFile for src/alpha.rs");
        assert!(beta.is_some(), "should have EditFile for src/beta.rs");

        if let Some(FileModification::EditFile { edits, .. }) = alpha {
            assert_eq!(edits.len(), 1);
            assert!(edits[0].search.contains("alpha_old"));
        }
        if let Some(FileModification::EditFile { edits, .. }) = beta {
            assert_eq!(edits.len(), 1);
            assert!(edits[0].search.contains("beta_old"));
        }
    }

    // -------------------------------------------------------------------
    // Mixed code blocks + search/replace
    // -------------------------------------------------------------------

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

        let has_create = parsed.modifications.iter().any(|m| {
            matches!(m, FileModification::CreateFile { path, .. } if path == "src/new_module.rs")
        });
        let has_edit = parsed.modifications.iter().any(|m| {
            matches!(m, FileModification::EditFile { path, .. } if path == "src/main.rs")
        });

        assert!(has_create, "should have a CreateFile for src/new_module.rs");
        assert!(has_edit, "should have an EditFile for src/main.rs");
    }

    // -------------------------------------------------------------------
    // Path validation
    // -------------------------------------------------------------------

    #[test]
    fn test_validate_file_path_relative_ok() {
        assert!(validate_file_path("src/lib.rs").is_ok());
        assert!(validate_file_path("Cargo.toml").is_ok());
        assert!(validate_file_path("crates/foo/src/main.rs").is_ok());
    }

    #[test]
    fn test_validate_file_path_rejects_absolute() {
        assert!(validate_file_path("/etc/passwd").is_err());
        assert!(validate_file_path("\\Windows\\System32\\cmd.exe").is_err());
        assert!(validate_file_path("C:\\Users\\file.txt").is_err());
    }

    #[test]
    fn test_validate_file_path_rejects_traversal() {
        assert!(validate_file_path("../../../etc/passwd").is_err());
        assert!(validate_file_path("src/../../secret.key").is_err());
    }

    #[test]
    fn test_validate_file_path_rejects_empty() {
        assert!(validate_file_path("").is_err());
        assert!(validate_file_path("   ").is_err());
    }

    #[test]
    fn test_path_traversal_skips_modification() {
        // A code block with a path-traversal path should be silently dropped
        let response = r#"Here is the fix:

```rust // ../../etc/shadow
root::0:0:root
```
"#;
        let parsed = parse_implementation(response);
        assert!(
            parsed.modifications.is_empty(),
            "path traversal should have been rejected"
        );
    }

    #[test]
    fn test_absolute_path_skips_sr_block() {
        let response = r#"File: /etc/shadow.conf

<<<SEARCH
old
>>>
<<<REPLACE
new
>>>
"#;
        let parsed = parse_implementation(response);
        assert!(
            parsed.modifications.is_empty(),
            "absolute path should have been rejected"
        );
    }

    // -------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_no_modifications() {
        let response = "I cannot implement this without more context about the database schema.";
        let parsed = parse_implementation(response);
        assert!(parsed.modifications.is_empty());
        assert!(!parsed.explanation.is_empty());
    }

    #[test]
    fn test_parse_python_file_path() {
        // No creation hint => RewriteFile
        let response = r#"```python # scripts/deploy.py
def deploy():
    pass
```
"#;
        let parsed = parse_implementation(response);
        assert_eq!(parsed.modifications.len(), 1);
        match &parsed.modifications[0] {
            FileModification::RewriteFile { path, .. } => {
                assert_eq!(path, "scripts/deploy.py");
            }
            other => panic!("expected RewriteFile, got {:?}", other),
        }
    }

    #[test]
    fn test_rewrite_file_hint_variations() {
        // "new file" hint
        let response = r#"Adding a new file now:

```rust // src/utils.rs
pub fn util() {}
```
"#;
        let parsed = parse_implementation(response);
        assert!(matches!(
            &parsed.modifications[0],
            FileModification::CreateFile { path, .. } if path == "src/utils.rs"
        ));

        // "write a new file" hint
        let response2 = r#"I'll write a new file for this:

```rust // src/writer.rs
pub fn write() {}
```
"#;
        let parsed2 = parse_implementation(response2);
        assert!(matches!(
            &parsed2.modifications[0],
            FileModification::CreateFile { path, .. } if path == "src/writer.rs"
        ));
    }
}
