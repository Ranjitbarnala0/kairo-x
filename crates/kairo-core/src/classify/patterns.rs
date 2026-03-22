//! Compiled regex pattern sets for response classification (§Flaw 3, §Flaw 8).
//!
//! All patterns are compiled once on first use via [`LazyLock<Regex>`].
//! Pattern categories:
//! - **Universal placeholder patterns**: language-agnostic stubs
//! - **Language-specific placeholder patterns**: per-language stub idioms
//! - **Refusal patterns**: LLM refusal/evasion phrases
//! - **Error patterns**: error reporting markers

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Language enum
// ---------------------------------------------------------------------------

/// Programming language for language-aware pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Python,
    TypeScript,
    JavaScript,
    Rust,
    Go,
    Java,
    /// Fallback — only universal patterns apply.
    Unknown,
}

impl Language {
    /// Attempt to infer language from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "py" | "pyi" | "pyx" => Self::Python,
            "ts" | "tsx" | "mts" | "cts" => Self::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "rs" => Self::Rust,
            "go" => Self::Go,
            "java" => Self::Java,
            _ => Self::Unknown,
        }
    }

    /// Attempt to infer language from a file path.
    pub fn from_path(path: &str) -> Self {
        path.rsplit('.')
            .next()
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Python => write!(f, "Python"),
            Self::TypeScript => write!(f, "TypeScript"),
            Self::JavaScript => write!(f, "JavaScript"),
            Self::Rust => write!(f, "Rust"),
            Self::Go => write!(f, "Go"),
            Self::Java => write!(f, "Java"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Universal placeholder patterns
// ---------------------------------------------------------------------------

/// Patterns that indicate placeholder/stub code in any language.
///
/// These match common lazy-output patterns like `// TODO`, `pass`,
/// `...`, `/* implement */`, etc.
pub static UNIVERSAL_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // Explicit TODO/FIXME/HACK markers
        (r"(?i)\b(TODO|FIXME|HACK|XXX)\b\s*[:\-]?\s*\w", "TODO/FIXME marker"),
        // "implement this", "implement later", "implement here"
        (r"(?i)\bimplement\s+(this|later|here|me|the\s+\w+)\b", "implement-this stub"),
        // "add .* here", "add .* later"
        (r"(?i)\badd\s+\w+.*\b(here|later)\b", "add-here stub"),
        // Ellipsis as implementation placeholder (standalone on a line)
        (r"^\s*\.\.\.\s*$", "ellipsis placeholder"),
        // "your .* here" pattern (e.g., "your code here", "your implementation here")
        (r"(?i)\byour\s+\w+\s+here\b", "your-X-here stub"),
        // "placeholder" as a standalone word in comments
        (r"(?i)\bplaceholder\b", "placeholder keyword"),
        // "stub" in comments
        (r"(?i)//\s*stub|#\s*stub|/\*\s*stub", "stub comment"),
        // "not implemented" phrase
        (r"(?i)\bnot\s+(?:yet\s+)?implemented\b", "not-implemented phrase"),
        // Empty function/method bodies — require function definition context
        (r"(?m)^\s*(?:fn|func|function|def|pub fn|pub\(crate\) fn)\s+\w+[^{]*\{\s*\}", "empty function body"),
    ];

    raw.iter()
        .map(|(pat, desc)| CompiledPattern {
            regex: Regex::new(pat).expect("universal placeholder pattern must compile"),
            description: desc.to_string(),
        })
        .collect()
});

// ---------------------------------------------------------------------------
// Language-specific placeholder patterns (§Flaw 8)
// ---------------------------------------------------------------------------

/// Python-specific placeholder patterns.
pub static PYTHON_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // bare `pass` as the only statement in a block
        (r"^\s*pass\s*$", "bare pass statement"),
        // `raise NotImplementedError`
        (r"raise\s+NotImplementedError", "raise NotImplementedError"),
        // Ellipsis literal as function body
        (r"^\s*\.\.\.\s*$", "ellipsis function body"),
        // `print("TODO")` or `print("not implemented")`
        (r#"print\s*\(\s*["'](?i)(todo|not implemented|placeholder)"#, "print-TODO pattern"),
        // `return None` as the only return in a function that should return something
        (r"^\s*return\s+None\s*$", "bare return None"),
    ];
    compile_patterns(&raw)
});

/// TypeScript/JavaScript-specific placeholder patterns.
pub static TYPESCRIPT_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // `throw new Error("Not implemented")`
        (r#"throw\s+new\s+Error\s*\(\s*["'](?i)(not\s*implemented|todo|implement)"#, "throw-not-implemented"),
        // `console.log("TODO")`
        (r#"console\.(log|warn|error)\s*\(\s*["'](?i)(todo|placeholder|implement)"#, "console-TODO"),
        // `return undefined` or `return null` as only return
        (r"^\s*return\s+(undefined|null)\s*;?\s*$", "bare return null/undefined"),
        // Empty arrow function body
        (r"=>\s*\{\s*\}", "empty arrow function"),
        // `// @ts-ignore` followed by suspicious stub
        (r"//\s*@ts-ignore", "ts-ignore directive"),
    ];
    compile_patterns(&raw)
});

/// Rust-specific placeholder patterns.
pub static RUST_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // `todo!()` macro
        (r"todo!\s*\(", "todo! macro"),
        // `unimplemented!()` macro
        (r"unimplemented!\s*\(", "unimplemented! macro"),
        // `panic!("not implemented")`
        (r#"panic!\s*\(\s*"(?i)(not\s*implemented|todo|implement)"#, "panic-not-implemented"),
        // `unreachable!()` used as stub (sometimes abused)
        (r"unreachable!\s*\(\s*\)", "unreachable! as stub"),
        // Empty match arms with `=> {}`
        (r"=>\s*\{\s*\}", "empty match arm"),
        // `Default::default()` as return value placeholder
        (r"^\s*Default::default\(\)\s*$", "Default::default() placeholder"),
    ];
    compile_patterns(&raw)
});

/// Go-specific placeholder patterns.
pub static GO_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // `panic("not implemented")`
        (r#"panic\s*\(\s*"(?i)(not\s*implemented|todo|implement)"#, "panic-not-implemented"),
        // `return nil` as stub (when function should return non-nil)
        (r"^\s*return\s+nil\s*$", "bare return nil"),
        // `_ = err` (ignoring errors as placeholder)
        (r"_\s*=\s*err\b", "ignored error"),
        // Empty function body `func ... { }`
        (r"func\s+\w+[^{]*\{\s*\}", "empty function body"),
        // `log.Println("TODO")`
        (r#"(?:log|fmt)\.\w+\s*\(\s*"(?i)(todo|placeholder|implement)"#, "log-TODO"),
    ];
    compile_patterns(&raw)
});

/// Java-specific placeholder patterns.
pub static JAVA_PLACEHOLDER_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        // `throw new UnsupportedOperationException`
        (r"throw\s+new\s+UnsupportedOperationException", "throw-UnsupportedOperationException"),
        // `throw new RuntimeException("not implemented")`
        (r#"throw\s+new\s+RuntimeException\s*\(\s*"(?i)(not\s*implemented|todo|implement)"#, "throw-RuntimeException-TODO"),
        // `return null;` as stub
        (r"^\s*return\s+null\s*;\s*$", "bare return null"),
        // `System.out.println("TODO")`
        (r#"System\.(out|err)\.println\s*\(\s*"(?i)(todo|placeholder|implement)"#, "sysout-TODO"),
        // Auto-generated method stub marker
        (r"(?i)auto.generated\s+method\s+stub", "auto-generated stub"),
    ];
    compile_patterns(&raw)
});

// ---------------------------------------------------------------------------
// Refusal patterns
// ---------------------------------------------------------------------------

/// Patterns that indicate the LLM refused to comply with the request.
pub static REFUSAL_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        (r"(?i)^I('m| am)\s+(unable|not able|cannot|can't)\s+to\s+(help|assist|do|complete|implement|write|create|generate)", "I-cannot refusal"),
        (r"(?i)\bI\s+(cannot|can't|won't|refuse to)\s+(implement|write|create|generate|provide|produce)\b", "I-refuse refusal"),
        (r"(?i)\bI('m| am)\s+sorry,?\s+but\s+I\s+(cannot|can't|won't)\b", "apologetic refusal"),
        (r"(?i)\bthis\s+(?:is\s+)?(?:beyond|outside)\s+(?:my|the\s+scope)", "scope refusal"),
        (r"(?i)\bI\s+don't\s+think\s+(?:I\s+)?(?:should|can)\s+(?:help|assist)\b", "hedging refusal"),
        (r"(?i)\bas\s+an?\s+AI\s+(?:language\s+)?model\b", "as-an-AI hedge"),
        (r"(?i)\bI\s+(?:must|need\s+to)\s+(?:decline|refuse)\b", "explicit decline"),
        (r"(?i)\bI\s+(?:can't|cannot)\s+(?:help|assist)\s+with\s+that\b", "cannot-help-with-that"),
    ];
    compile_patterns(&raw)
});

// ---------------------------------------------------------------------------
// Error patterns
// ---------------------------------------------------------------------------

/// Patterns that indicate the LLM is reporting an error or inability.
pub static ERROR_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        (r"(?i)^\s*error\s*:", "error-colon prefix"),
        (r"(?i)\b(compilation|syntax|runtime|type)\s+error\b", "error type keyword"),
        (r"(?i)\bfailed\s+to\s+(compile|parse|build|run|execute)\b", "failed-to-X"),
        (r"(?i)\bcannot\s+find\s+(module|package|crate|import|file|symbol)\b", "cannot-find dependency"),
        (r"(?i)\bundefined\s+(reference|symbol|variable|function)\b", "undefined-X"),
        (r"(?i)\bmissing\s+(dependency|import|module|package|type)\b", "missing-dependency"),
    ];
    compile_patterns(&raw)
});

// ---------------------------------------------------------------------------
// PASS/FAIL keyword patterns (for verify/audit calls)
// ---------------------------------------------------------------------------

/// Pattern for PASS keyword in verification responses.
pub static PASS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*PASS\b").expect("PASS pattern must compile")
});

/// Pattern for FAIL keyword in verification responses.
pub static FAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*FAIL\b").expect("FAIL pattern must compile")
});

// ---------------------------------------------------------------------------
// Question detection patterns
// ---------------------------------------------------------------------------

/// Patterns that suggest the LLM is asking a clarifying question rather than providing output.
pub static QUESTION_PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    let raw = [
        (r"(?i)^(?:could|can|would|should|do|did|is|are|was|were|will|shall)\s+(?:you|I|we|it)\b.*\?\s*$", "question sentence"),
        (r"(?i)\bclarif(?:y|ication)\b", "clarification keyword"),
        (r"(?i)\bwhich\s+(?:approach|option|method|way|implementation)\b.*\?", "which-approach question"),
        (r"(?i)\bbefore\s+I\s+(?:proceed|continue|implement|start)\b", "before-I-proceed"),
        (r"(?i)\bdo\s+you\s+(?:want|prefer|mean)\b", "do-you-want question"),
        (r"(?i)\bplease\s+(?:confirm|clarify|specify)\b", "please-confirm request"),
    ];
    compile_patterns(&raw)
});

// ---------------------------------------------------------------------------
// Compiled pattern wrapper
// ---------------------------------------------------------------------------

/// A pre-compiled regex pattern with a human-readable description.
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    pub regex: Regex,
    pub description: String,
}

impl CompiledPattern {
    /// Test whether this pattern matches anywhere in the given text.
    pub fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }

    /// Find the first match in the given text, returning the matched string.
    pub fn find<'a>(&self, text: &'a str) -> Option<&'a str> {
        self.regex.find(text).map(|m| m.as_str())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the language-specific placeholder patterns for a given language.
pub fn placeholder_patterns_for(language: Language) -> &'static [CompiledPattern] {
    match language {
        Language::Python => &PYTHON_PLACEHOLDER_PATTERNS,
        Language::TypeScript | Language::JavaScript => &TYPESCRIPT_PLACEHOLDER_PATTERNS,
        Language::Rust => &RUST_PLACEHOLDER_PATTERNS,
        Language::Go => &GO_PLACEHOLDER_PATTERNS,
        Language::Java => &JAVA_PLACEHOLDER_PATTERNS,
        Language::Unknown => &[],
    }
}

fn compile_patterns(raw: &[(&str, &str)]) -> Vec<CompiledPattern> {
    raw.iter()
        .map(|(pat, desc)| CompiledPattern {
            regex: Regex::new(pat).unwrap_or_else(|e| panic!("pattern '{pat}' must compile: {e}")),
            description: desc.to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("java"), Language::Java);
        assert_eq!(Language::from_extension("xyz"), Language::Unknown);
    }

    #[test]
    fn test_language_from_path() {
        assert_eq!(Language::from_path("src/main.rs"), Language::Rust);
        assert_eq!(Language::from_path("app/index.tsx"), Language::TypeScript);
        assert_eq!(Language::from_path("scripts/deploy.py"), Language::Python);
    }

    #[test]
    fn test_universal_placeholder_todo() {
        let patterns = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
        let todo_text = "// TODO: implement this function";
        assert!(patterns.iter().any(|p| p.is_match(todo_text)));
    }

    #[test]
    fn test_universal_placeholder_fixme() {
        let patterns = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
        let fixme_text = "# FIXME: broken implementation";
        assert!(patterns.iter().any(|p| p.is_match(fixme_text)));
    }

    #[test]
    fn test_universal_placeholder_implement_this() {
        let patterns = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("// implement this")));
        assert!(patterns.iter().any(|p| p.is_match("// implement later")));
    }

    #[test]
    fn test_python_pass_placeholder() {
        let patterns = &*PYTHON_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("    pass")));
    }

    #[test]
    fn test_python_not_implemented() {
        let patterns = &*PYTHON_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("raise NotImplementedError")));
    }

    #[test]
    fn test_rust_todo_macro() {
        let patterns = &*RUST_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("todo!()")));
        assert!(patterns.iter().any(|p| p.is_match("todo!(\"not done\")")));
    }

    #[test]
    fn test_rust_unimplemented() {
        let patterns = &*RUST_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("unimplemented!()")));
    }

    #[test]
    fn test_typescript_throw_not_implemented() {
        let patterns = &*TYPESCRIPT_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match(
            r#"throw new Error("Not implemented")"#
        )));
    }

    #[test]
    fn test_go_panic_not_implemented() {
        let patterns = &*GO_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match(
            r#"panic("not implemented")"#
        )));
    }

    #[test]
    fn test_java_unsupported_operation() {
        let patterns = &*JAVA_PLACEHOLDER_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match(
            "throw new UnsupportedOperationException"
        )));
    }

    #[test]
    fn test_refusal_patterns() {
        let patterns = &*REFUSAL_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match(
            "I'm unable to help with that request"
        )));
        assert!(patterns.iter().any(|p| p.is_match(
            "I cannot implement this feature"
        )));
        assert!(patterns.iter().any(|p| p.is_match(
            "I'm sorry, but I can't write that code"
        )));
    }

    #[test]
    fn test_error_patterns() {
        let patterns = &*ERROR_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match("error: something went wrong")));
        assert!(patterns.iter().any(|p| p.is_match("compilation error in module")));
        assert!(patterns.iter().any(|p| p.is_match("failed to compile the module")));
    }

    #[test]
    fn test_pass_fail_patterns() {
        assert!(PASS_PATTERN.is_match("PASS\nEverything looks good."));
        assert!(PASS_PATTERN.is_match("  PASS"));
        assert!(FAIL_PATTERN.is_match("FAIL\n1. Missing error handling"));
        assert!(!PASS_PATTERN.is_match("The test passed"));
        assert!(!FAIL_PATTERN.is_match("The test failed"));
    }

    #[test]
    fn test_question_patterns() {
        let patterns = &*QUESTION_PATTERNS;
        assert!(patterns.iter().any(|p| p.is_match(
            "Could you clarify what you mean?"
        )));
        assert!(patterns.iter().any(|p| p.is_match(
            "Before I proceed, I need some information"
        )));
    }

    #[test]
    fn test_legitimate_code_not_matched_as_placeholder() {
        let patterns = &*UNIVERSAL_PLACEHOLDER_PATTERNS;
        // Real implementation should NOT trigger
        let real_code = "fn calculate_sum(a: i32, b: i32) -> i32 { a + b }";
        // "implement" by itself without "this/later/here" shouldn't match
        assert!(!patterns.iter().any(|p| p.is_match(real_code)));
    }
}
