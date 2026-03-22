//! Context candidates and feature vectors (§9.2).
//!
//! A `ContextCandidate` is a piece of content (file, symbol, snippet, etc.)
//! that *might* be useful context for an LLM call. Each candidate carries a
//! feature vector (`ContextFeatures`) with 12 binary/categorical signals used
//! for scoring and ranking.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ContextSource — where the candidate came from
// ---------------------------------------------------------------------------

/// The origin of a context candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextSource {
    /// A full source file from the project tree.
    File,
    /// A single symbol definition (function, struct, type alias, etc.).
    Symbol,
    /// A snippet / region extracted from a larger file.
    Snippet,
    /// The natural-language specification attached to a graph node.
    NodeSpec,
    /// A test file associated with an implementation file.
    TestFile,
}

impl fmt::Display for ContextSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File => write!(f, "file"),
            Self::Symbol => write!(f, "symbol"),
            Self::Snippet => write!(f, "snippet"),
            Self::NodeSpec => write!(f, "node_spec"),
            Self::TestFile => write!(f, "test_file"),
        }
    }
}

// ---------------------------------------------------------------------------
// File-size bucket constants
// ---------------------------------------------------------------------------

/// File-size bucket: 0–50 tokens.
pub const BUCKET_TINY: u8 = 0;
/// File-size bucket: 51–300 tokens.
pub const BUCKET_SMALL: u8 = 1;
/// File-size bucket: 301–1500 tokens.
pub const BUCKET_MEDIUM: u8 = 2;
/// File-size bucket: >1500 tokens.
pub const BUCKET_LARGE: u8 = 3;

/// Map an estimated token count to a file-size bucket.
pub fn token_bucket(estimated_tokens: u32) -> u8 {
    match estimated_tokens {
        0..=50 => BUCKET_TINY,
        51..=300 => BUCKET_SMALL,
        301..=1500 => BUCKET_MEDIUM,
        _ => BUCKET_LARGE,
    }
}

// ---------------------------------------------------------------------------
// ContextFeatures — 12 binary/categorical signals
// ---------------------------------------------------------------------------

/// Feature vector describing the relationship of a candidate to the current
/// target node. These features are used by the scoring function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextFeatures {
    /// Candidate lives in the same directory as the primary target file.
    pub same_directory: bool,
    /// Candidate is imported by one of the target files.
    pub imported_by_target: bool,
    /// Candidate imports one of the target files.
    pub imports_target: bool,
    /// Candidate is written in the same programming language as the target.
    pub same_language: bool,
    /// Candidate was recently modified (within the current session window).
    pub recently_modified: bool,
    /// Candidate (or parts of it) is explicitly mentioned in the node spec.
    pub mentioned_in_spec: bool,
    /// Candidate is a test file related to the target implementation.
    pub related_test: bool,
    /// Candidate comes from a dependency node's implementation output.
    pub dependency_of_target: bool,
    /// Candidate filename is similar to the target filename.
    pub similar_name: bool,
    /// Candidate was touched / viewed in the current interactive session.
    pub in_current_session: bool,
    /// File-size bucket: 0=tiny, 1=small, 2=medium, 3=large.
    pub file_size_bucket: u8,
    /// Candidate contains type definitions (structs, interfaces, enums, etc.).
    pub type_definition: bool,
}

impl Default for ContextFeatures {
    fn default() -> Self {
        Self {
            same_directory: false,
            imported_by_target: false,
            imports_target: false,
            same_language: false,
            recently_modified: false,
            mentioned_in_spec: false,
            related_test: false,
            dependency_of_target: false,
            similar_name: false,
            in_current_session: false,
            file_size_bucket: BUCKET_MEDIUM,
            type_definition: false,
        }
    }
}

impl ContextFeatures {
    /// Compute a relevance score in the range `[0.0, 1.0]`.
    ///
    /// Weights are tuned so that direct import relationships and spec
    /// mentions dominate, while weaker signals like same-directory still
    /// contribute.
    pub fn score(&self) -> f64 {
        // Weights for each feature (hand-tuned per §9.2 recommendations).
        const W_IMPORTED_BY_TARGET: f64 = 0.20;
        const W_IMPORTS_TARGET: f64 = 0.15;
        const W_MENTIONED_IN_SPEC: f64 = 0.18;
        const W_DEPENDENCY_OF_TARGET: f64 = 0.12;
        const W_RELATED_TEST: f64 = 0.08;
        const W_TYPE_DEFINITION: f64 = 0.07;
        const W_SAME_DIRECTORY: f64 = 0.05;
        const W_SAME_LANGUAGE: f64 = 0.03;
        const W_RECENTLY_MODIFIED: f64 = 0.04;
        const W_SIMILAR_NAME: f64 = 0.04;
        const W_IN_CURRENT_SESSION: f64 = 0.03;
        // File-size bucket contributes a small bonus for smaller files
        // (they are cheaper to include).
        const W_SIZE_BONUS: f64 = 0.01;

        let mut s = 0.0;
        if self.imported_by_target {
            s += W_IMPORTED_BY_TARGET;
        }
        if self.imports_target {
            s += W_IMPORTS_TARGET;
        }
        if self.mentioned_in_spec {
            s += W_MENTIONED_IN_SPEC;
        }
        if self.dependency_of_target {
            s += W_DEPENDENCY_OF_TARGET;
        }
        if self.related_test {
            s += W_RELATED_TEST;
        }
        if self.type_definition {
            s += W_TYPE_DEFINITION;
        }
        if self.same_directory {
            s += W_SAME_DIRECTORY;
        }
        if self.same_language {
            s += W_SAME_LANGUAGE;
        }
        if self.recently_modified {
            s += W_RECENTLY_MODIFIED;
        }
        if self.similar_name {
            s += W_SIMILAR_NAME;
        }
        if self.in_current_session {
            s += W_IN_CURRENT_SESSION;
        }

        // Size bonus: tiny=3x, small=2x, medium=1x, large=0x
        let size_multiplier = match self.file_size_bucket {
            BUCKET_TINY => 3.0,
            BUCKET_SMALL => 2.0,
            BUCKET_MEDIUM => 1.0,
            _ => 0.0,
        };
        s += W_SIZE_BONUS * size_multiplier;

        s.clamp(0.0, 1.0)
    }

    /// Count how many boolean features are set to `true`.
    pub fn active_count(&self) -> u32 {
        let bools = [
            self.same_directory,
            self.imported_by_target,
            self.imports_target,
            self.same_language,
            self.recently_modified,
            self.mentioned_in_spec,
            self.related_test,
            self.dependency_of_target,
            self.similar_name,
            self.in_current_session,
            self.type_definition,
        ];
        bools.iter().filter(|&&b| b).count() as u32
    }
}

// ---------------------------------------------------------------------------
// ContextCandidate
// ---------------------------------------------------------------------------

/// A candidate piece of content that may be included in an LLM context package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCandidate {
    /// Where this candidate came from.
    pub source: ContextSource,
    /// XXH3 hash of the content for deduplication.
    pub content_hash: u64,
    /// Estimated number of tokens this content will consume.
    pub estimated_tokens: u32,
    /// Feature vector describing relevance signals.
    pub features: ContextFeatures,
    /// Human-readable label (file path, symbol name, etc.).
    pub label: String,
    /// The actual text content of this candidate.
    pub content: String,
}

impl ContextCandidate {
    /// Convenience constructor that computes the content hash automatically.
    pub fn new(
        source: ContextSource,
        label: impl Into<String>,
        content: impl Into<String>,
        features: ContextFeatures,
    ) -> Self {
        let content = content.into();
        let label = label.into();
        let content_hash = xxh3_hash(content.as_bytes());
        let estimated_tokens = estimate_tokens(&content);
        Self {
            source,
            content_hash,
            estimated_tokens,
            features: ContextFeatures {
                file_size_bucket: token_bucket(estimated_tokens),
                ..features
            },
            label,
            content,
        }
    }

    /// Relevance score derived from the feature vector.
    pub fn score(&self) -> f64 {
        self.features.score()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Hash bytes with XXH3-64.
pub(crate) fn xxh3_hash(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

/// Token estimate using char count for accurate estimation across all scripts.
///
/// Uses ~3.5 chars per token average, which better handles multi-byte UTF-8
/// content (CJK, emoji, etc.) compared to the old byte-based heuristic.
pub fn estimate_tokens(text: &str) -> u32 {
    let char_count = text.chars().count();
    ((char_count as f64 / 3.5) as u32).max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_source_display() {
        assert_eq!(ContextSource::File.to_string(), "file");
        assert_eq!(ContextSource::Symbol.to_string(), "symbol");
        assert_eq!(ContextSource::Snippet.to_string(), "snippet");
        assert_eq!(ContextSource::NodeSpec.to_string(), "node_spec");
        assert_eq!(ContextSource::TestFile.to_string(), "test_file");
    }

    #[test]
    fn test_token_bucket_mapping() {
        assert_eq!(token_bucket(0), BUCKET_TINY);
        assert_eq!(token_bucket(50), BUCKET_TINY);
        assert_eq!(token_bucket(51), BUCKET_SMALL);
        assert_eq!(token_bucket(300), BUCKET_SMALL);
        assert_eq!(token_bucket(301), BUCKET_MEDIUM);
        assert_eq!(token_bucket(1500), BUCKET_MEDIUM);
        assert_eq!(token_bucket(1501), BUCKET_LARGE);
        assert_eq!(token_bucket(100_000), BUCKET_LARGE);
    }

    #[test]
    fn test_default_features_score_is_low() {
        let feat = ContextFeatures::default();
        let score = feat.score();
        // Default features (all false, medium bucket) should give a minimal score.
        assert!(score < 0.05, "default score should be near zero, got {score}");
    }

    #[test]
    fn test_high_relevance_features_score_high() {
        let feat = ContextFeatures {
            imported_by_target: true,
            mentioned_in_spec: true,
            dependency_of_target: true,
            same_language: true,
            ..Default::default()
        };
        let score = feat.score();
        assert!(score > 0.4, "highly relevant features should score high, got {score}");
    }

    #[test]
    fn test_score_clamped_to_one() {
        let feat = ContextFeatures {
            same_directory: true,
            imported_by_target: true,
            imports_target: true,
            same_language: true,
            recently_modified: true,
            mentioned_in_spec: true,
            related_test: true,
            dependency_of_target: true,
            similar_name: true,
            in_current_session: true,
            file_size_bucket: BUCKET_TINY,
            type_definition: true,
        };
        let score = feat.score();
        assert!(score <= 1.0, "score must be clamped to 1.0, got {score}");
    }

    #[test]
    fn test_active_count() {
        let feat = ContextFeatures {
            same_directory: true,
            imported_by_target: true,
            mentioned_in_spec: true,
            ..Default::default()
        };
        assert_eq!(feat.active_count(), 3);
    }

    #[test]
    fn test_active_count_all_false() {
        assert_eq!(ContextFeatures::default().active_count(), 0);
    }

    #[test]
    fn test_candidate_new_computes_hash_and_tokens() {
        let c = ContextCandidate::new(
            ContextSource::File,
            "src/main.rs",
            "fn main() { println!(\"hello\"); }",
            ContextFeatures::default(),
        );
        assert_ne!(c.content_hash, 0);
        assert!(c.estimated_tokens > 0);
        assert_eq!(c.label, "src/main.rs");
        assert_eq!(c.source, ContextSource::File);
    }

    #[test]
    fn test_candidate_dedup_by_hash() {
        let c1 = ContextCandidate::new(
            ContextSource::File,
            "a.rs",
            "same content",
            ContextFeatures::default(),
        );
        let c2 = ContextCandidate::new(
            ContextSource::Symbol,
            "b.rs",
            "same content",
            ContextFeatures::default(),
        );
        assert_eq!(c1.content_hash, c2.content_hash);
    }

    #[test]
    fn test_candidate_different_content_different_hash() {
        let c1 = ContextCandidate::new(
            ContextSource::File,
            "a.rs",
            "content A",
            ContextFeatures::default(),
        );
        let c2 = ContextCandidate::new(
            ContextSource::File,
            "a.rs",
            "content B",
            ContextFeatures::default(),
        );
        assert_ne!(c1.content_hash, c2.content_hash);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 1); // min 1
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_xxh3_hash_deterministic() {
        let a = xxh3_hash(b"hello");
        let b = xxh3_hash(b"hello");
        assert_eq!(a, b);
        assert_ne!(xxh3_hash(b"hello"), xxh3_hash(b"world"));
    }

    #[test]
    fn test_file_size_bucket_set_on_construction() {
        let tiny = ContextCandidate::new(
            ContextSource::File,
            "t.rs",
            "x", // 1 token = tiny
            ContextFeatures::default(),
        );
        assert_eq!(tiny.features.file_size_bucket, BUCKET_TINY);

        // Generate ~400 tokens worth of content (1600 bytes)
        let medium_content = "a".repeat(1600);
        let medium = ContextCandidate::new(
            ContextSource::File,
            "m.rs",
            medium_content,
            ContextFeatures::default(),
        );
        assert_eq!(medium.features.file_size_bucket, BUCKET_MEDIUM);
    }

    #[test]
    fn test_size_bonus_in_score() {
        let feat_tiny = ContextFeatures {
            file_size_bucket: BUCKET_TINY,
            ..Default::default()
        };
        let feat_large = ContextFeatures {
            file_size_bucket: BUCKET_LARGE,
            ..Default::default()
        };
        // Tiny files get a small bonus over large files.
        assert!(feat_tiny.score() > feat_large.score());
    }
}
