//! Context engine subsystem (§9).
//!
//! Assembles context packages for every LLM call. The engine gathers candidate
//! context items from multiple sources (files, symbols, specs, import graph
//! neighbors), scores them with a 12-feature vector, and greedily packs the
//! highest-scoring candidates into a token budget.
//!
//! ## Submodules
//!
//! - [`candidates`]: `ContextCandidate`, `ContextFeatures`, and the
//!   `ContextSource` enum used by the scoring/packing pipeline.
//! - [`engine`]: `ContextEngine` — the main coordinator that gathers, scores,
//!   and packs candidates.
//! - [`packer`]: Greedy context packing with partial-inclusion and truncation.
//! - [`import_graph`]: Lightweight import/require/use graph built via regex.
//!
//! ## Legacy builder API
//!
//! The `ContextBuilder`, `ContextPackage`, and `ContextStrategy` types in this
//! file are the original context assembly interface used by the call assembler.
//! They remain available for callers that want direct, priority-based packing
//! without the full candidate-gathering pipeline.

pub mod candidates;
pub mod engine;
pub mod import_graph;
pub mod packer;

// ---------------------------------------------------------------------------
// Re-exports — new §9 pipeline types
// ---------------------------------------------------------------------------

pub use candidates::{ContextCandidate, ContextFeatures};
pub use candidates::ContextSource as CandidateSource;
pub use engine::{ContextEngine, ContextError};
pub use import_graph::ImportGraph;
pub use packer::{PackedContext, PackedItem};

// ---------------------------------------------------------------------------
// Imports for the legacy builder types defined in this file
// ---------------------------------------------------------------------------

use crate::fingerprint::Fingerprinter;
use kairo_llm::call::LLMCallType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// ContextSource (legacy) — what kinds of context the builder can gather
// ---------------------------------------------------------------------------

/// A single source of context information for an LLM call.
///
/// This is the legacy enum used by [`ContextBuilder`]. For the scoring/packing
/// pipeline, see [`candidates::ContextSource`] (re-exported as `CandidateSource`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextSource {
    /// Full file content.
    File {
        path: PathBuf,
        content: String,
    },
    /// A range of lines from a file.
    FileRange {
        path: PathBuf,
        content: String,
        start_line: u32,
        end_line: u32,
    },
    /// A symbol definition (function, struct, trait, class, etc.).
    Symbol {
        name: String,
        definition: String,
        file_path: PathBuf,
    },
    /// Project structure summary (directory tree, key files).
    ProjectStructure {
        summary: String,
    },
    /// Prior implementation output from another node (dependency context).
    DependencyOutput {
        node_title: String,
        output_summary: String,
    },
    /// Test output from a prior verification run.
    TestOutput {
        output: String,
        passed: bool,
    },
    /// Error or diagnostic message from a tool run.
    Diagnostic {
        tool: String,
        message: String,
    },
}

impl ContextSource {
    /// Estimated token count for this context source (rough: 1 token ~ 4 chars).
    pub fn estimated_tokens(&self) -> u32 {
        let chars = match self {
            Self::File { content, .. } => content.len(),
            Self::FileRange { content, .. } => content.len(),
            Self::Symbol { definition, .. } => definition.len(),
            Self::ProjectStructure { summary } => summary.len(),
            Self::DependencyOutput { output_summary, .. } => output_summary.len(),
            Self::TestOutput { output, .. } => output.len(),
            Self::Diagnostic { message, .. } => message.len(),
        };
        (chars as u32 / 4).max(1)
    }

    /// Human-readable label for this context source.
    pub fn label(&self) -> String {
        match self {
            Self::File { path, .. } => format!("file:{}", path.display()),
            Self::FileRange { path, start_line, end_line, .. } => {
                format!("file:{}:{}-{}", path.display(), start_line, end_line)
            }
            Self::Symbol { name, .. } => format!("symbol:{name}"),
            Self::ProjectStructure { .. } => "project-structure".to_string(),
            Self::DependencyOutput { node_title, .. } => format!("dep:{node_title}"),
            Self::TestOutput { passed, .. } => {
                format!("test-output:{}", if *passed { "pass" } else { "fail" })
            }
            Self::Diagnostic { tool, .. } => format!("diag:{tool}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ContextPackage (legacy)
// ---------------------------------------------------------------------------

/// A complete context package ready for rendering into an LLM prompt.
///
/// The package has been budget-trimmed and fingerprinted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackage {
    /// The context sources included in this package.
    pub sources: Vec<ContextSource>,
    /// Total estimated tokens for all included sources.
    pub total_estimated_tokens: u32,
    /// XXH3 fingerprint of the package contents (for change detection).
    pub fingerprint: u64,
    /// Whether any sources were trimmed to fit the budget.
    pub was_trimmed: bool,
}

impl ContextPackage {
    /// Render all sources into a single string for inclusion in an LLM prompt.
    pub fn render(&self) -> String {
        let mut parts = Vec::with_capacity(self.sources.len());

        for source in &self.sources {
            match source {
                ContextSource::File { path, content } => {
                    parts.push(format!(
                        "--- {} ---\n{}\n",
                        path.display(),
                        content
                    ));
                }
                ContextSource::FileRange {
                    path,
                    content,
                    start_line,
                    end_line,
                } => {
                    parts.push(format!(
                        "--- {} (lines {}-{}) ---\n{}\n",
                        path.display(),
                        start_line,
                        end_line,
                        content
                    ));
                }
                ContextSource::Symbol {
                    name,
                    definition,
                    file_path,
                } => {
                    parts.push(format!(
                        "--- symbol: {name} (from {}) ---\n{definition}\n",
                        file_path.display()
                    ));
                }
                ContextSource::ProjectStructure { summary } => {
                    parts.push(format!("--- Project Structure ---\n{summary}\n"));
                }
                ContextSource::DependencyOutput {
                    node_title,
                    output_summary,
                } => {
                    parts.push(format!(
                        "--- Dependency: {node_title} ---\n{output_summary}\n"
                    ));
                }
                ContextSource::TestOutput { output, passed } => {
                    let status = if *passed { "PASSED" } else { "FAILED" };
                    parts.push(format!("--- Test Output ({status}) ---\n{output}\n"));
                }
                ContextSource::Diagnostic { tool, message } => {
                    parts.push(format!("--- {tool} output ---\n{message}\n"));
                }
            }
        }

        parts.join("\n")
    }

    /// Render only sources that are new since a previous fingerprint.
    ///
    /// This is used for session continuation: only send new context to
    /// the LLM, not context it already has.
    pub fn render_new_only(&self, previous_fingerprint: u64) -> String {
        if self.fingerprint == previous_fingerprint {
            return String::new();
        }
        self.render()
    }
}

// ---------------------------------------------------------------------------
// ContextBuilder (legacy)
// ---------------------------------------------------------------------------

/// Builds a context package for an LLM call, respecting a token budget.
pub struct ContextBuilder {
    /// Maximum tokens to allocate for context.
    token_budget: u32,
    /// Collected context sources (not yet budget-trimmed).
    sources: Vec<(ContextSource, ContextPriority)>,
}

/// Priority of a context source — determines trim order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextPriority {
    /// Must be included (e.g., the file being edited).
    Required = 3,
    /// Strongly relevant (e.g., direct dependency output).
    High = 2,
    /// Useful but not critical (e.g., project structure).
    Medium = 1,
    /// Nice to have (e.g., similar files for style reference).
    Low = 0,
}

impl ContextBuilder {
    /// Create a new context builder with the given token budget.
    pub fn new(token_budget: u32) -> Self {
        Self {
            token_budget,
            sources: Vec::new(),
        }
    }

    /// Add a context source at the given priority.
    pub fn add(&mut self, source: ContextSource, priority: ContextPriority) -> &mut Self {
        self.sources.push((source, priority));
        self
    }

    /// Add a file as context.
    pub fn add_file(&mut self, path: PathBuf, content: String, priority: ContextPriority) -> &mut Self {
        self.add(ContextSource::File { path, content }, priority)
    }

    /// Add a symbol definition as context.
    pub fn add_symbol(
        &mut self,
        name: String,
        definition: String,
        file_path: PathBuf,
        priority: ContextPriority,
    ) -> &mut Self {
        self.add(
            ContextSource::Symbol {
                name,
                definition,
                file_path,
            },
            priority,
        )
    }

    /// Build the context package, trimming low-priority sources to fit the budget.
    pub fn build(mut self) -> ContextPackage {
        // Sort by priority (highest first)
        self.sources.sort_by(|a, b| b.1.cmp(&a.1));

        let mut included = Vec::new();
        let mut total_tokens = 0u32;
        let mut was_trimmed = false;

        for (source, _priority) in self.sources {
            let tokens = source.estimated_tokens();
            if total_tokens + tokens <= self.token_budget {
                total_tokens += tokens;
                included.push(source);
            } else {
                was_trimmed = true;
            }
        }

        // Compute fingerprint over all included source content
        let fingerprint = {
            let mut fp = Fingerprinter::new();
            for source in &included {
                fp.feed(source.label().as_bytes());
                match source {
                    ContextSource::File { content, .. }
                    | ContextSource::FileRange { content, .. } => {
                        fp.feed(content.as_bytes());
                    }
                    ContextSource::Symbol { definition, .. } => {
                        fp.feed(definition.as_bytes());
                    }
                    ContextSource::ProjectStructure { summary } => {
                        fp.feed(summary.as_bytes());
                    }
                    ContextSource::DependencyOutput { output_summary, .. } => {
                        fp.feed(output_summary.as_bytes());
                    }
                    ContextSource::TestOutput { output, .. } => {
                        fp.feed(output.as_bytes());
                    }
                    ContextSource::Diagnostic { message, .. } => {
                        fp.feed(message.as_bytes());
                    }
                }
            }
            fp.finish()
        };

        ContextPackage {
            sources: included,
            total_estimated_tokens: total_tokens,
            fingerprint,
            was_trimmed,
        }
    }
}

// ---------------------------------------------------------------------------
// ContextStrategy (legacy)
// ---------------------------------------------------------------------------

/// Determines the context gathering strategy for a given call type and priority.
pub struct ContextStrategy;

impl ContextStrategy {
    /// Get the recommended token budget fraction for context (of the total context window).
    ///
    /// The rest of the window is reserved for the system prompt, enforcement template,
    /// instruction, and LLM output.
    pub fn context_budget_fraction(call_type: LLMCallType) -> f32 {
        match call_type {
            LLMCallType::Implement => 0.50,
            LLMCallType::Verify | LLMCallType::Audit => 0.45,
            LLMCallType::Fix | LLMCallType::Debug => 0.40,
            LLMCallType::Plan | LLMCallType::Decompose => 0.35,
            LLMCallType::Explain => 0.40,
        }
    }

    /// Compute the absolute token budget for context.
    pub fn context_budget(call_type: LLMCallType, max_context_tokens: u32) -> u32 {
        (max_context_tokens as f32 * Self::context_budget_fraction(call_type)) as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_source_token_estimate() {
        let source = ContextSource::File {
            path: PathBuf::from("test.rs"),
            content: "fn main() { println!(\"hello\"); }".to_string(),
        };
        assert!(source.estimated_tokens() > 0);
    }

    #[test]
    fn test_context_source_label() {
        let source = ContextSource::File {
            path: PathBuf::from("src/main.rs"),
            content: String::new(),
        };
        assert_eq!(source.label(), "file:src/main.rs");

        let source = ContextSource::Symbol {
            name: "calculate_hash".to_string(),
            definition: String::new(),
            file_path: PathBuf::from("src/lib.rs"),
        };
        assert_eq!(source.label(), "symbol:calculate_hash");
    }

    #[test]
    fn test_context_builder_fits_budget() {
        let mut builder = ContextBuilder::new(100);

        builder.add_file(
            PathBuf::from("small.rs"),
            "fn a() {}".to_string(),
            ContextPriority::Required,
        );

        builder.add_file(
            PathBuf::from("large.rs"),
            "x".repeat(1000),
            ContextPriority::Low,
        );

        let package = builder.build();
        assert_eq!(package.sources.len(), 1);
        assert!(package.was_trimmed);
    }

    #[test]
    fn test_context_builder_priority_ordering() {
        let mut builder = ContextBuilder::new(10000);

        builder.add_file(
            PathBuf::from("low.rs"),
            "low priority".to_string(),
            ContextPriority::Low,
        );
        builder.add_file(
            PathBuf::from("high.rs"),
            "high priority".to_string(),
            ContextPriority::High,
        );
        builder.add_file(
            PathBuf::from("required.rs"),
            "required".to_string(),
            ContextPriority::Required,
        );

        let package = builder.build();
        assert_eq!(package.sources.len(), 3);
        assert!(!package.was_trimmed);
    }

    #[test]
    fn test_context_package_render() {
        let package = ContextPackage {
            sources: vec![
                ContextSource::File {
                    path: PathBuf::from("src/main.rs"),
                    content: "fn main() {}".to_string(),
                },
                ContextSource::Symbol {
                    name: "Handler".to_string(),
                    definition: "struct Handler { field: u32 }".to_string(),
                    file_path: PathBuf::from("src/lib.rs"),
                },
            ],
            total_estimated_tokens: 20,
            fingerprint: 0x1234,
            was_trimmed: false,
        };

        let rendered = package.render();
        assert!(rendered.contains("src/main.rs"));
        assert!(rendered.contains("fn main() {}"));
        assert!(rendered.contains("Handler"));
    }

    #[test]
    fn test_context_package_fingerprint_changes() {
        let mut builder1 = ContextBuilder::new(10000);
        builder1.add_file(
            PathBuf::from("a.rs"),
            "version 1".to_string(),
            ContextPriority::Required,
        );
        let pkg1 = builder1.build();

        let mut builder2 = ContextBuilder::new(10000);
        builder2.add_file(
            PathBuf::from("a.rs"),
            "version 2".to_string(),
            ContextPriority::Required,
        );
        let pkg2 = builder2.build();

        assert_ne!(pkg1.fingerprint, pkg2.fingerprint);
    }

    #[test]
    fn test_context_strategy_budgets() {
        let budget = ContextStrategy::context_budget(LLMCallType::Implement, 200_000);
        assert_eq!(budget, 100_000);

        let budget = ContextStrategy::context_budget(LLMCallType::Plan, 200_000);
        assert_eq!(budget, 70_000);
    }

    #[test]
    fn test_render_new_only_same_fingerprint() {
        let package = ContextPackage {
            sources: vec![ContextSource::File {
                path: PathBuf::from("a.rs"),
                content: "hello".to_string(),
            }],
            total_estimated_tokens: 2,
            fingerprint: 42,
            was_trimmed: false,
        };

        assert!(package.render_new_only(42).is_empty());
        assert!(!package.render_new_only(99).is_empty());
    }
}
