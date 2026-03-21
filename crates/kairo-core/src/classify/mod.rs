//! Response classification subsystem (§Flaw 3).
//!
//! Classifies LLM responses into one of the [`ResponseClass`] categories
//! using a two-layer approach:
//!
//! **Layer 1 — Deterministic rules** (`rules.rs`):
//! Regex-based pattern matching against known PASS/FAIL keywords,
//! placeholder patterns, refusal patterns, and error detection.
//!
//! **Layer 2 — Heuristic fallback** (`fallback.rs`):
//! When deterministic rules are inconclusive, a heuristic-based classifier
//! uses response length, code-to-text ratio, and keyword presence to
//! classify ambiguous responses. A neural classifier stub is provided
//! for future integration.
//!
//! All regex patterns are compiled once via [`LazyLock`] in `patterns.rs`.

pub mod fallback;
pub mod patterns;
pub mod rules;

pub use fallback::heuristic_classify;
pub use patterns::Language;
pub use rules::classify_response;
