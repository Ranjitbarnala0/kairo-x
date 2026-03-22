//! NEED_CONTEXT protocol parser (§Flaw 11).
//!
//! When the LLM needs additional context to complete a task, it responds with
//! structured NEED_CONTEXT markers. This module parses those markers into
//! actionable context requests.

use crate::response::{ContextRequest, ContextRequestKind};
use regex::Regex;
use std::path::{Component, Path};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Regex patterns for context request parsing
// ---------------------------------------------------------------------------

static FILE_REQUEST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^-\s*file:\s*(?P<path>[^\s(,]+)(?:\s*,\s*lines?\s+(?P<start>\d+)\s*-\s*(?P<end>\d+))?\s*(?:\(reason:\s*(?P<reason>[^)]+)\))?\s*$"
    ).expect("file request regex must compile")
});

static SYMBOL_REQUEST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^-\s*symbol:\s*(?P<name>[^\s(]+)\s*(?:\(reason:\s*(?P<reason>[^)]+)\))?\s*$"
    ).expect("symbol request regex must compile")
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Maximum number of context-request rounds per LLM call before forcing
/// the LLM to proceed with available context.
pub const MAX_CONTEXT_REQUEST_ROUNDS: u32 = 3;

/// Parse NEED_CONTEXT markers from an LLM response.
///
/// Returns `None` if the response doesn't contain a NEED_CONTEXT block.
/// Returns `Some(vec)` with the parsed requests if found.
///
/// Expected format in the LLM response:
/// ```text
/// NEED_CONTEXT:
/// - file: path/to/file.ts (reason: need to see the User interface definition)
/// - file: path/to/other.ts, lines 50-80 (reason: need the validation logic)
/// - symbol: ClassName.methodName (reason: need to understand the return type)
/// ```
pub fn parse_context_request(response: &str) -> Option<Vec<ContextRequest>> {
    // Check for the NEED_CONTEXT marker
    let marker_pos = response.find("NEED_CONTEXT:")?;
    let after_marker = &response[marker_pos + "NEED_CONTEXT:".len()..];

    let requests: Vec<ContextRequest> = after_marker
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with("- file:") || l.starts_with("- symbol:"))
        .filter_map(parse_single_request)
        .collect();

    if requests.is_empty() {
        None
    } else {
        Some(requests)
    }
}

/// Validate that a context-requested file path is safe to serve.
///
/// Rejects absolute paths and any path containing `..` components to prevent
/// directory-traversal attacks from LLM-generated context requests.
fn validate_context_path(path: &str) -> bool {
    let p = Path::new(path);
    // Reject absolute paths (e.g. /etc/passwd, C:\Windows\...)
    if p.is_absolute() {
        return false;
    }
    // Reject any parent-directory traversal component
    for component in p.components() {
        if matches!(component, Component::ParentDir) {
            return false;
        }
    }
    true
}

/// Parse a single context request line.
fn parse_single_request(line: &str) -> Option<ContextRequest> {
    // Try file request first
    if let Some(caps) = FILE_REQUEST_RE.captures(line) {
        let path = caps.name("path")?.as_str().to_string();

        if !validate_context_path(&path) {
            tracing::warn!(path = %path, "Rejecting context request with unsafe path");
            return None;
        }

        let line_start = caps
            .name("start")
            .and_then(|m| m.as_str().parse::<u32>().ok());
        let line_end = caps
            .name("end")
            .and_then(|m| m.as_str().parse::<u32>().ok());
        let reason = caps.name("reason").map(|m| m.as_str().to_string());

        return Some(ContextRequest {
            kind: ContextRequestKind::File {
                path,
                line_start,
                line_end,
            },
            reason,
        });
    }

    // Try symbol request
    if let Some(caps) = SYMBOL_REQUEST_RE.captures(line) {
        let name = caps.name("name")?.as_str().to_string();
        let reason = caps.name("reason").map(|m| m.as_str().to_string());

        return Some(ContextRequest {
            kind: ContextRequestKind::Symbol { name },
            reason,
        });
    }

    None
}

/// Generate the response message to inject requested context back into the session.
pub fn format_context_injection(
    fulfilled: &[(ContextRequest, String)],
    not_found: &[(ContextRequest, Vec<String>)],
) -> String {
    let mut parts = Vec::new();

    if !fulfilled.is_empty() {
        parts.push("Here is the context you requested:".to_string());
        for (req, content) in fulfilled {
            let label = match &req.kind {
                ContextRequestKind::File {
                    path,
                    line_start,
                    line_end,
                } => {
                    if let (Some(start), Some(end)) = (line_start, line_end) {
                        format!("--- {path} (lines {start}-{end}) ---")
                    } else {
                        format!("--- {path} ---")
                    }
                }
                ContextRequestKind::Symbol { name } => {
                    format!("--- symbol: {name} ---")
                }
            };
            parts.push(label);
            parts.push(content.clone());
            parts.push(String::new());
        }
    }

    if !not_found.is_empty() {
        for (req, alternatives) in not_found {
            match &req.kind {
                ContextRequestKind::File { path, .. } => {
                    if alternatives.is_empty() {
                        parts.push(format!("File {path} does not exist."));
                    } else {
                        parts.push(format!(
                            "File {path} does not exist. Available files in that directory: {}. Did you mean one of these?",
                            alternatives.join(", ")
                        ));
                    }
                }
                ContextRequestKind::Symbol { name } => {
                    if alternatives.is_empty() {
                        parts.push(format!("Symbol {name} was not found in the codebase."));
                    } else {
                        parts.push(format!(
                            "Symbol {name} was not found. Similar symbols: {}",
                            alternatives.join(", ")
                        ));
                    }
                }
            }
        }
    }

    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_file_request() {
        let response = "I need to see some types.\nNEED_CONTEXT:\n- file: src/types.ts (reason: need User interface)\n- file: src/auth.ts, lines 50-80 (reason: validation logic)\n";
        let requests = parse_context_request(response).unwrap();
        assert_eq!(requests.len(), 2);

        match &requests[0].kind {
            ContextRequestKind::File {
                path,
                line_start,
                line_end,
            } => {
                assert_eq!(path, "src/types.ts");
                assert!(line_start.is_none());
                assert!(line_end.is_none());
            }
            _ => panic!("expected file request"),
        }
        assert_eq!(
            requests[0].reason.as_deref(),
            Some("need User interface")
        );

        match &requests[1].kind {
            ContextRequestKind::File {
                path,
                line_start,
                line_end,
            } => {
                assert_eq!(path, "src/auth.ts");
                assert_eq!(*line_start, Some(50));
                assert_eq!(*line_end, Some(80));
            }
            _ => panic!("expected file request"),
        }
    }

    #[test]
    fn test_parse_symbol_request() {
        let response =
            "NEED_CONTEXT:\n- symbol: AuthService.validateToken (reason: need return type)\n";
        let requests = parse_context_request(response).unwrap();
        assert_eq!(requests.len(), 1);

        match &requests[0].kind {
            ContextRequestKind::Symbol { name } => {
                assert_eq!(name, "AuthService.validateToken");
            }
            _ => panic!("expected symbol request"),
        }
    }

    #[test]
    fn test_no_context_request() {
        let response = "Here is the implementation:\n```rust\nfn main() {}\n```\n";
        assert!(parse_context_request(response).is_none());
    }

    #[test]
    fn test_empty_context_request() {
        let response = "NEED_CONTEXT:\nSome unstructured text";
        assert!(parse_context_request(response).is_none());
    }

    #[test]
    fn test_validate_context_path_rejects_absolute() {
        assert!(!validate_context_path("/etc/passwd"));
        assert!(!validate_context_path("/home/user/.ssh/id_rsa"));
    }

    #[test]
    fn test_validate_context_path_rejects_traversal() {
        assert!(!validate_context_path("../../etc/passwd"));
        assert!(!validate_context_path("src/../../secrets.txt"));
        assert!(!validate_context_path(".."));
    }

    #[test]
    fn test_validate_context_path_accepts_safe_paths() {
        assert!(validate_context_path("src/types.ts"));
        assert!(validate_context_path("crates/core/src/lib.rs"));
        assert!(validate_context_path("README.md"));
        assert!(validate_context_path("src/nested/deep/file.rs"));
    }

    #[test]
    fn test_parse_rejects_absolute_path_request() {
        let response =
            "NEED_CONTEXT:\n- file: /etc/passwd (reason: need user list)\n";
        assert!(parse_context_request(response).is_none());
    }

    #[test]
    fn test_parse_rejects_traversal_path_request() {
        let response =
            "NEED_CONTEXT:\n- file: ../../etc/shadow (reason: need hashes)\n";
        assert!(parse_context_request(response).is_none());
    }

    #[test]
    fn test_parse_keeps_safe_requests_drops_unsafe() {
        let response = "NEED_CONTEXT:\n- file: src/lib.rs (reason: need trait def)\n- file: /etc/passwd (reason: malicious)\n- file: ../../secrets (reason: traversal)\n";
        let requests = parse_context_request(response).unwrap();
        assert_eq!(requests.len(), 1);
        match &requests[0].kind {
            ContextRequestKind::File { path, .. } => {
                assert_eq!(path, "src/lib.rs");
            }
            _ => panic!("expected file request"),
        }
    }
}
