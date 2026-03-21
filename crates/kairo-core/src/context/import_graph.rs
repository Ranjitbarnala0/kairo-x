//! Lightweight import graph (§14.2).
//!
//! Scans source files for `import`, `require`, `use`, `from`, and `#include`
//! statements using per-language regex patterns. Builds a bidirectional graph
//! mapping file hashes to their imports and reverse-imports.
//!
//! The graph is intentionally approximate: it resolves textual path strings
//! rather than performing full module resolution. This is sufficient for
//! context ranking — false positives are cheap (slightly lower precision in
//! candidate scoring), while the speed benefit is large.

use fnv::FnvHashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Per-language import regex patterns
// ---------------------------------------------------------------------------

/// Matches ES/TS imports: `import ... from "..."` and `require("...")`
static JS_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?m)(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))"#,
    )
    .expect("JS import regex must compile")
});

/// Matches Rust `use` statements: `use crate::foo::bar;` or `use super::baz;`
static RUST_USE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*use\s+((?:crate|super|self|[a-zA-Z_][a-zA-Z0-9_]*)(?:::[a-zA-Z_*{][a-zA-Z0-9_:*{}, ]*)?)\s*;")
        .expect("Rust use regex must compile")
});

/// Matches Python imports: `import foo`, `from foo import bar`
static PYTHON_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+))")
        .expect("Python import regex must compile")
});

/// Matches Go imports: `import "path"` and grouped imports
static GO_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)"([a-zA-Z0-9_./-]+)""#).expect("Go import regex must compile")
});

/// Matches C/C++ includes: `#include "file.h"` and `#include <file.h>`
static C_INCLUDE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)^[ \t]*#\s*include\s*[<"]([^>"]+)[>"]"#)
        .expect("C include regex must compile")
});

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

/// Supported language categories for import extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    JavaScript,
    TypeScript,
    Rust,
    Python,
    Go,
    C,
    Cpp,
    Unknown,
}

impl Language {
    /// Detect language from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "tsx" | "mts" | "cts" => Self::TypeScript,
            "rs" => Self::Rust,
            "py" | "pyi" => Self::Python,
            "go" => Self::Go,
            "c" | "h" => Self::C,
            "cpp" | "cxx" | "cc" | "hpp" | "hxx" | "hh" => Self::Cpp,
            _ => Self::Unknown,
        }
    }

    /// Detect language from a full file path.
    pub fn from_path(path: &str) -> Self {
        Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Whether two paths share the same language family.
    pub fn same_family(path_a: &str, path_b: &str) -> bool {
        let la = Self::from_path(path_a);
        let lb = Self::from_path(path_b);
        if la == Self::Unknown || lb == Self::Unknown {
            return false;
        }
        // JS/TS are the same family; C/C++ are the same family.
        match (la, lb) {
            (Self::JavaScript, Self::TypeScript)
            | (Self::TypeScript, Self::JavaScript) => true,
            (Self::C, Self::Cpp) | (Self::Cpp, Self::C) => true,
            (a, b) => a == b,
        }
    }
}

// ---------------------------------------------------------------------------
// ImportGraph
// ---------------------------------------------------------------------------

/// Bidirectional import graph: maps file hashes to their import edges.
///
/// `imports[A]` = list of file hashes that A imports.
/// `imported_by[B]` = list of file hashes that import B.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportGraph {
    /// file_hash → hashes of files it imports.
    pub imports: FnvHashMap<u64, Vec<u64>>,
    /// file_hash → hashes of files that import it.
    pub imported_by: FnvHashMap<u64, Vec<u64>>,
}

impl ImportGraph {
    /// Create an empty import graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a directed edge: `from_hash` imports `to_hash`.
    ///
    /// Maintains both the forward and reverse maps. Deduplicates edges.
    pub fn add_edge(&mut self, from_hash: u64, to_hash: u64) {
        let fwd = self.imports.entry(from_hash).or_default();
        if !fwd.contains(&to_hash) {
            fwd.push(to_hash);
        }

        let rev = self.imported_by.entry(to_hash).or_default();
        if !rev.contains(&from_hash) {
            rev.push(from_hash);
        }
    }

    /// Remove all edges originating from `from_hash` (used before re-scanning a file).
    pub fn remove_outgoing(&mut self, from_hash: u64) {
        if let Some(targets) = self.imports.remove(&from_hash) {
            for target in &targets {
                if let Some(rev) = self.imported_by.get_mut(target) {
                    rev.retain(|&h| h != from_hash);
                    if rev.is_empty() {
                        self.imported_by.remove(target);
                    }
                }
            }
        }
    }

    /// Get the direct imports of a file (files it depends on).
    pub fn imports_of(&self, file_hash: u64) -> &[u64] {
        self.imports.get(&file_hash).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get the files that import a given file (reverse dependencies).
    pub fn importers_of(&self, file_hash: u64) -> &[u64] {
        self.imported_by.get(&file_hash).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Return all neighbors of a file: both its imports and its importers.
    pub fn neighbors(&self, file_hash: u64) -> Vec<u64> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        if let Some(fwd) = self.imports.get(&file_hash) {
            for &h in fwd {
                if seen.insert(h) {
                    result.push(h);
                }
            }
        }
        if let Some(rev) = self.imported_by.get(&file_hash) {
            for &h in rev {
                if seen.insert(h) {
                    result.push(h);
                }
            }
        }
        result
    }

    /// Rebuild the entire graph from a set of `(file_path, file_content)` pairs.
    ///
    /// `path_to_hash` is a closure that resolves a file path string to its
    /// canonical hash. This allows the caller to control how paths are
    /// normalized and registered.
    pub fn rebuild_from_files<F>(
        &mut self,
        files: &[(String, String)],
        path_to_hash: F,
    ) where
        F: Fn(&str) -> u64,
    {
        self.imports.clear();
        self.imported_by.clear();

        // Pre-build the known-paths set once for O(1) resolution lookups,
        // instead of rebuilding it inside resolve_import_path for every
        // import statement (which was O(n_files * n_imports * n_files)).
        let known_paths: HashSet<&str> = files.iter().map(|(p, _)| p.as_str()).collect();

        for (path, content) in files {
            let from_hash = path_to_hash(path);
            let raw_imports = extract_imports(path, content);

            for raw_import in raw_imports {
                if let Some(resolved) = resolve_import_path_with_set(path, &raw_import, &known_paths) {
                    let to_hash = path_to_hash(&resolved);
                    self.add_edge(from_hash, to_hash);
                }
            }
        }
    }

    /// Scan a single file and update its outgoing edges.
    ///
    /// This is the incremental variant of `rebuild_from_files`.
    pub fn update_file<F>(
        &mut self,
        path: &str,
        content: &str,
        known_files: &[(String, String)],
        path_to_hash: F,
    ) where
        F: Fn(&str) -> u64,
    {
        let from_hash = path_to_hash(path);
        self.remove_outgoing(from_hash);

        let raw_imports = extract_imports(path, content);
        for raw_import in raw_imports {
            if let Some(resolved) = resolve_import_path(path, &raw_import, known_files) {
                let to_hash = path_to_hash(&resolved);
                self.add_edge(from_hash, to_hash);
            }
        }
    }

    /// Total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.imports.values().map(|v| v.len()).sum()
    }

    /// Total number of unique files tracked (either as importer or importee).
    pub fn file_count(&self) -> usize {
        let mut seen = HashSet::new();
        for &k in self.imports.keys() {
            seen.insert(k);
        }
        for &k in self.imported_by.keys() {
            seen.insert(k);
        }
        seen.len()
    }
}

// ---------------------------------------------------------------------------
// Import extraction
// ---------------------------------------------------------------------------

/// Extract raw import strings from a source file based on its language.
///
/// Returns unresolved import specifiers (e.g. `"./utils"`, `"crate::foo"`).
pub fn extract_imports(file_path: &str, content: &str) -> Vec<String> {
    let lang = Language::from_path(file_path);
    match lang {
        Language::JavaScript | Language::TypeScript => extract_js_imports(content),
        Language::Rust => extract_rust_imports(content),
        Language::Python => extract_python_imports(content),
        Language::Go => extract_go_imports(content),
        Language::C | Language::Cpp => extract_c_includes(content),
        Language::Unknown => Vec::new(),
    }
}

fn extract_js_imports(content: &str) -> Vec<String> {
    let mut results = Vec::new();
    for caps in JS_IMPORT_RE.captures_iter(content) {
        if let Some(m) = caps.get(1).or_else(|| caps.get(2)) {
            results.push(m.as_str().to_string());
        }
    }
    results
}

fn extract_rust_imports(content: &str) -> Vec<String> {
    let mut results = Vec::new();
    for caps in RUST_USE_RE.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            results.push(m.as_str().to_string());
        }
    }
    results
}

fn extract_python_imports(content: &str) -> Vec<String> {
    let mut results = Vec::new();
    for caps in PYTHON_IMPORT_RE.captures_iter(content) {
        if let Some(m) = caps.get(1).or_else(|| caps.get(2)) {
            results.push(m.as_str().to_string());
        }
    }
    results
}

fn extract_go_imports(content: &str) -> Vec<String> {
    let mut results = Vec::new();
    for caps in GO_IMPORT_RE.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            results.push(m.as_str().to_string());
        }
    }
    results
}

fn extract_c_includes(content: &str) -> Vec<String> {
    let mut results = Vec::new();
    for caps in C_INCLUDE_RE.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            results.push(m.as_str().to_string());
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Import resolution
// ---------------------------------------------------------------------------

/// Attempt to resolve a raw import string to a concrete file path.
///
/// This uses simple heuristics — not a full module resolver. It tries:
/// 1. Exact match against known file paths.
/// 2. Relative path resolution from the importing file's directory.
/// 3. Common extension appending (.ts, .js, .rs, .py, etc.).
/// 4. Index file resolution (index.ts, index.js, __init__.py, mod.rs).
///
/// Returns `None` if no known file matches.
/// Attempt to resolve a raw import string to a concrete file path.
///
/// This uses simple heuristics — not a full module resolver. It tries:
/// 1. Exact match against known file paths.
/// 2. Relative path resolution from the importing file's directory.
/// 3. Common extension appending (.ts, .js, .rs, .py, etc.).
/// 4. Index file resolution (index.ts, index.js, __init__.py, mod.rs).
///
/// Returns `None` if no known file matches.
fn resolve_import_path(
    importer_path: &str,
    raw_import: &str,
    known_files: &[(String, String)],
) -> Option<String> {
    let known_paths: HashSet<&str> = known_files.iter().map(|(p, _)| p.as_str()).collect();
    resolve_import_path_with_set(importer_path, raw_import, &known_paths)
}

/// Resolve with a pre-built known-paths set (used by `rebuild_from_files`
/// to avoid rebuilding the set for every import statement).
fn resolve_import_path_with_set(
    importer_path: &str,
    raw_import: &str,
    known_paths: &HashSet<&str>,
) -> Option<String> {

    // 1. Exact match
    if known_paths.contains(raw_import) {
        return Some(raw_import.to_string());
    }

    // 2. Relative path resolution
    let importer_dir = Path::new(importer_path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or("");

    let candidates = build_resolution_candidates(importer_dir, raw_import);

    for candidate in &candidates {
        // Normalize by removing leading "./"
        let normalized = candidate
            .strip_prefix("./")
            .unwrap_or(candidate);

        if known_paths.contains(normalized) {
            return Some(normalized.to_string());
        }

        // Also try with "./" prefix in case known_paths use that form
        let with_dot_slash = format!("./{normalized}");
        if known_paths.contains(with_dot_slash.as_str()) {
            return Some(with_dot_slash);
        }
    }

    None
}

/// Build a list of possible file paths that a raw import might refer to.
fn build_resolution_candidates(importer_dir: &str, raw_import: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // For relative imports (./foo, ../foo)
    let base = if raw_import.starts_with('.') {
        if importer_dir.is_empty() {
            raw_import.to_string()
        } else {
            format!("{importer_dir}/{raw_import}")
        }
    } else {
        raw_import.to_string()
    };

    // Normalize path components (remove ../ and ./)
    let normalized = normalize_path(&base);
    candidates.push(normalized.clone());

    // Try common extensions
    let extensions = &[
        ".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs",
        ".rs", ".py", ".go", ".c", ".h", ".cpp", ".hpp",
    ];
    for ext in extensions {
        candidates.push(format!("{normalized}{ext}"));
    }

    // Try index files
    let index_files = &[
        "index.ts", "index.tsx", "index.js", "index.jsx",
        "mod.rs", "__init__.py",
    ];
    for idx in index_files {
        candidates.push(format!("{normalized}/{idx}"));
    }

    candidates
}

/// Simplistic path normalization: collapse `.` and `..` components.
fn normalize_path(path: &str) -> String {
    let mut components: Vec<&str> = Vec::new();

    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            other => {
                components.push(other);
            }
        }
    }

    components.join("/")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::fnv_hash;

    // -- Language detection --

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("c"), Language::C);
        assert_eq!(Language::from_extension("cpp"), Language::Cpp);
        assert_eq!(Language::from_extension("xyz"), Language::Unknown);
    }

    #[test]
    fn test_language_from_path() {
        assert_eq!(Language::from_path("src/main.rs"), Language::Rust);
        assert_eq!(Language::from_path("lib/auth.ts"), Language::TypeScript);
        assert_eq!(Language::from_path("no_extension"), Language::Unknown);
    }

    #[test]
    fn test_same_family() {
        assert!(Language::same_family("a.ts", "b.js"));
        assert!(Language::same_family("a.c", "b.cpp"));
        assert!(Language::same_family("a.rs", "b.rs"));
        assert!(!Language::same_family("a.rs", "b.py"));
        assert!(!Language::same_family("a.rs", "b.unknown"));
    }

    // -- Import extraction --

    #[test]
    fn test_extract_js_imports_esm() {
        let content = r#"
import { foo } from "./foo";
import bar from "../bar";
import * as baz from "baz";
"#;
        let imports = extract_imports("test.ts", content);
        assert_eq!(imports, vec!["./foo", "../bar", "baz"]);
    }

    #[test]
    fn test_extract_js_imports_require() {
        let content = r#"
const x = require("./utils");
const y = require('lodash');
"#;
        let imports = extract_imports("test.js", content);
        assert_eq!(imports, vec!["./utils", "lodash"]);
    }

    #[test]
    fn test_extract_rust_imports() {
        let content = r#"
use crate::arena::Node;
use super::helpers;
use std::collections::HashMap;
"#;
        let imports = extract_imports("src/lib.rs", content);
        assert_eq!(imports.len(), 3);
        assert!(imports.contains(&"crate::arena::Node".to_string()));
        assert!(imports.contains(&"super::helpers".to_string()));
        assert!(imports.contains(&"std::collections::HashMap".to_string()));
    }

    #[test]
    fn test_extract_python_imports() {
        let content = r#"
import os
from pathlib import Path
from .utils import helper
import sys
"#;
        let imports = extract_imports("test.py", content);
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"pathlib".to_string()));
        assert!(imports.contains(&".utils".to_string()));
        assert!(imports.contains(&"sys".to_string()));
    }

    #[test]
    fn test_extract_go_imports() {
        let content = r#"
import (
    "fmt"
    "net/http"
    "github.com/foo/bar"
)
"#;
        let imports = extract_imports("main.go", content);
        assert!(imports.contains(&"fmt".to_string()));
        assert!(imports.contains(&"net/http".to_string()));
        assert!(imports.contains(&"github.com/foo/bar".to_string()));
    }

    #[test]
    fn test_extract_c_includes() {
        let content = r#"
#include <stdio.h>
#include "myheader.h"
#include <stdlib.h>
"#;
        let imports = extract_imports("main.c", content);
        assert!(imports.contains(&"stdio.h".to_string()));
        assert!(imports.contains(&"myheader.h".to_string()));
        assert!(imports.contains(&"stdlib.h".to_string()));
    }

    #[test]
    fn test_extract_unknown_language() {
        let imports = extract_imports("data.csv", "col1,col2,col3");
        assert!(imports.is_empty());
    }

    // -- Import graph --

    #[test]
    fn test_add_edge_and_query() {
        let mut g = ImportGraph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(4, 2);

        assert_eq!(g.imports_of(1), &[2, 3]);
        assert_eq!(g.importers_of(2), &[1, 4]);
        assert_eq!(g.imports_of(4), &[2]);
        assert!(g.imports_of(99).is_empty());
    }

    #[test]
    fn test_add_edge_dedup() {
        let mut g = ImportGraph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 2);
        g.add_edge(1, 2);

        assert_eq!(g.imports_of(1).len(), 1);
        assert_eq!(g.importers_of(2).len(), 1);
    }

    #[test]
    fn test_remove_outgoing() {
        let mut g = ImportGraph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(4, 2);

        g.remove_outgoing(1);

        assert!(g.imports_of(1).is_empty());
        // 2 should only be imported by 4 now
        assert_eq!(g.importers_of(2), &[4]);
        // 3 should have no importers
        assert!(g.importers_of(3).is_empty());
    }

    #[test]
    fn test_neighbors() {
        let mut g = ImportGraph::new();
        g.add_edge(1, 2); // 1 imports 2
        g.add_edge(3, 1); // 3 imports 1

        let n = g.neighbors(1);
        assert!(n.contains(&2)); // 1 imports 2
        assert!(n.contains(&3)); // 3 imports 1
        assert_eq!(n.len(), 2);
    }

    #[test]
    fn test_edge_count_and_file_count() {
        let mut g = ImportGraph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(4, 2);

        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.file_count(), 4); // files 1, 2, 3, 4
    }

    #[test]
    fn test_rebuild_from_files() {
        let files = vec![
            ("src/main.ts".to_string(), r#"import { foo } from "./utils";"#.to_string()),
            ("src/utils.ts".to_string(), "export function foo() {}".to_string()),
        ];

        let mut g = ImportGraph::new();
        g.rebuild_from_files(&files, fnv_hash);

        let main_hash = fnv_hash("src/main.ts");
        let utils_hash = fnv_hash("src/utils.ts");

        assert!(g.imports_of(main_hash).contains(&utils_hash));
        assert!(g.importers_of(utils_hash).contains(&main_hash));
    }

    #[test]
    fn test_update_file_incremental() {
        let files = vec![
            ("src/a.ts".to_string(), r#"import { b } from "./b";"#.to_string()),
            ("src/b.ts".to_string(), "export const b = 1;".to_string()),
            ("src/c.ts".to_string(), "export const c = 2;".to_string()),
        ];

        let mut g = ImportGraph::new();
        g.rebuild_from_files(&files, fnv_hash);

        let a_hash = fnv_hash("src/a.ts");
        let b_hash = fnv_hash("src/b.ts");
        let c_hash = fnv_hash("src/c.ts");

        assert!(g.imports_of(a_hash).contains(&b_hash));

        // Now a.ts changes to import c instead of b
        let new_content = r#"import { c } from "./c";"#;
        g.update_file("src/a.ts", new_content, &files, fnv_hash);

        assert!(!g.imports_of(a_hash).contains(&b_hash));
        assert!(g.imports_of(a_hash).contains(&c_hash));
        assert!(g.importers_of(b_hash).is_empty());
        assert!(g.importers_of(c_hash).contains(&a_hash));
    }

    // -- Path normalization --

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("src/../lib/foo"), "lib/foo");
        assert_eq!(normalize_path("./src/./utils"), "src/utils");
        assert_eq!(normalize_path("a/b/../c"), "a/c");
        assert_eq!(normalize_path("a/b/c"), "a/b/c");
    }

    // -- Resolution --

    #[test]
    fn test_resolve_relative_import() {
        let files = vec![
            ("src/main.ts".to_string(), String::new()),
            ("src/utils.ts".to_string(), String::new()),
        ];

        let resolved = resolve_import_path("src/main.ts", "./utils", &files);
        assert_eq!(resolved, Some("src/utils.ts".to_string()));
    }

    #[test]
    fn test_resolve_exact_match() {
        let files = vec![
            ("lib/helper.js".to_string(), String::new()),
        ];

        let resolved = resolve_import_path("src/main.js", "lib/helper.js", &files);
        assert_eq!(resolved, Some("lib/helper.js".to_string()));
    }

    #[test]
    fn test_resolve_not_found() {
        let files = vec![
            ("src/main.ts".to_string(), String::new()),
        ];

        let resolved = resolve_import_path("src/main.ts", "./nonexistent", &files);
        assert!(resolved.is_none());
    }
}
