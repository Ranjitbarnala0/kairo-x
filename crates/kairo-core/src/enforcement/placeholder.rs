//! Language-aware placeholder detection for LLM-generated code.
//!
//! Uses the compiled regex patterns from [`classify::patterns`] to scan code
//! for placeholder/stub patterns and report matches with line numbers.

use crate::classify::patterns::{
    self, CompiledPattern, Language, UNIVERSAL_PLACEHOLDER_PATTERNS,
};

// ---------------------------------------------------------------------------
// Placeholder match result
// ---------------------------------------------------------------------------

/// A single placeholder detection result with location and pattern info.
#[derive(Debug, Clone)]
pub struct PlaceholderMatch {
    /// 1-based line number where the placeholder was found.
    pub line_number: usize,
    /// The text of the line containing the placeholder.
    pub line_text: String,
    /// Description of the pattern that matched.
    pub pattern_description: String,
    /// The specific matched text within the line.
    pub matched_text: String,
}

impl std::fmt::Display for PlaceholderMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "line {}: {} ({})",
            self.line_number, self.matched_text, self.pattern_description
        )
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect placeholder patterns in code, using both universal and language-specific patterns.
///
/// Returns a list of [`PlaceholderMatch`] with line numbers and the pattern that matched.
/// An empty result means no placeholders were detected.
///
/// # Arguments
/// * `code` - The code text to scan.
/// * `language` - The programming language for language-specific pattern matching.
pub fn detect_placeholders(code: &str, language: Language) -> Vec<PlaceholderMatch> {
    let mut matches = Vec::new();

    let universal = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
    let language_specific = patterns::placeholder_patterns_for(language);

    for (line_idx, line) in code.lines().enumerate() {
        let line_number = line_idx + 1;

        // Skip empty lines and very short lines (less than 3 chars)
        let trimmed = line.trim();
        if trimmed.len() < 3 {
            continue;
        }

        // Check universal patterns
        check_patterns(line, line_number, universal, &mut matches);

        // Check language-specific patterns
        check_patterns(line, line_number, language_specific, &mut matches);
    }

    // Deduplicate by line number (keep first match per line)
    matches.dedup_by_key(|m| m.line_number);

    matches
}

/// Detect placeholders with automatic language inference from file path.
pub fn detect_placeholders_auto(code: &str, file_path: &str) -> Vec<PlaceholderMatch> {
    let language = Language::from_path(file_path);
    detect_placeholders(code, language)
}

/// Quick check: does the code contain any placeholder patterns?
///
/// More efficient than [`detect_placeholders`] when you only need a boolean answer,
/// as it short-circuits on the first match.
pub fn has_placeholders(code: &str, language: Language) -> bool {
    let universal = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
    let language_specific = patterns::placeholder_patterns_for(language);

    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.len() < 3 {
            continue;
        }

        for pattern in universal.iter().chain(language_specific.iter()) {
            if pattern.is_match(line) {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn check_patterns(
    line: &str,
    line_number: usize,
    patterns: &[CompiledPattern],
    matches: &mut Vec<PlaceholderMatch>,
) {
    for pattern in patterns {
        if let Some(matched) = pattern.find(line) {
            matches.push(PlaceholderMatch {
                line_number,
                line_text: line.to_string(),
                pattern_description: pattern.description.clone(),
                matched_text: matched.to_string(),
            });
            // Only report the first pattern match per line to avoid noise
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_rust_todo_macro() {
        let code = r#"
fn process(data: &[u8]) -> Result<(), Error> {
    todo!()
}
"#;
        let matches = detect_placeholders(code, Language::Rust);
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.line_number == 3));
    }

    #[test]
    fn test_detect_rust_unimplemented() {
        let code = r#"
fn handler(req: Request) -> Response {
    unimplemented!()
}
"#;
        let matches = detect_placeholders(code, Language::Rust);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_python_pass() {
        let code = r#"
def process_data(data):
    pass
"#;
        let matches = detect_placeholders(code, Language::Python);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_python_not_implemented() {
        let code = r#"
class Handler:
    def process(self):
        raise NotImplementedError
"#;
        let matches = detect_placeholders(code, Language::Python);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_typescript_throw() {
        let code = r#"
function processOrder(order: Order): Result {
    throw new Error("Not implemented");
}
"#;
        let matches = detect_placeholders(code, Language::TypeScript);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_go_panic() {
        let code = r#"
func ProcessData(data []byte) error {
    panic("not implemented")
}
"#;
        let matches = detect_placeholders(code, Language::Go);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_java_unsupported_operation() {
        let code = r#"
public class Handler {
    public void process() {
        throw new UnsupportedOperationException();
    }
}
"#;
        let matches = detect_placeholders(code, Language::Java);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_detect_universal_todo_comment() {
        let code = "// TODO: implement error handling for edge cases";
        let matches = detect_placeholders(code, Language::Unknown);
        assert!(!matches.is_empty());
        assert!(matches[0].pattern_description.contains("TODO"));
    }

    #[test]
    fn test_clean_code_no_matches() {
        let code = r#"
fn calculate_sum(a: i32, b: i32) -> i32 {
    a.checked_add(b).unwrap_or(i32::MAX)
}

fn calculate_product(a: i32, b: i32) -> i32 {
    a.checked_mul(b).unwrap_or(i32::MAX)
}
"#;
        let matches = detect_placeholders(code, Language::Rust);
        assert!(matches.is_empty(), "False positive: {:?}", matches);
    }

    #[test]
    fn test_has_placeholders_quick_check() {
        let clean = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let dirty = "fn add(a: i32, b: i32) -> i32 { todo!() }";

        assert!(!has_placeholders(clean, Language::Rust));
        assert!(has_placeholders(dirty, Language::Rust));
    }

    #[test]
    fn test_auto_language_detection() {
        let code = "todo!()";
        let matches = detect_placeholders_auto(code, "src/lib.rs");
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_line_numbers_are_one_based() {
        let code = "line 1\nline 2\n// TODO: fix this\nline 4";
        let matches = detect_placeholders(code, Language::Unknown);
        assert_eq!(matches[0].line_number, 3);
    }

    #[test]
    fn test_multiple_placeholders_different_lines() {
        let code = r#"
fn a() {
    todo!()
}

fn b() {
    unimplemented!()
}
"#;
        let matches = detect_placeholders(code, Language::Rust);
        assert!(matches.len() >= 2);
    }

    #[test]
    fn test_placeholder_match_display() {
        let m = PlaceholderMatch {
            line_number: 42,
            line_text: "    todo!()".to_string(),
            pattern_description: "todo! macro".to_string(),
            matched_text: "todo!(".to_string(),
        };
        let display = format!("{}", m);
        assert!(display.contains("42"));
        assert!(display.contains("todo!"));
    }
}
