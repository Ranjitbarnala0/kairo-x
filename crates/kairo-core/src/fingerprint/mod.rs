//! Project fingerprinting subsystem (§14) and content fingerprinting utilities.
//!
//! ## Project Fingerprinting
//!
//! Automatically detects the language, package manager, build system,
//! framework, and project layout of a target repository. The resulting
//! [`ProjectFingerprint`] drives every downstream decision:
//!
//! - **Verification**: which shell commands to run for build/test/lint/typecheck
//! - **Enforcement**: language-aware placeholder patterns
//! - **Context**: which files constitute the "source root" vs test root
//!
//! Detection is fast and IO-bound (directory walking + file existence checks +
//! light JSON/TOML parsing). No network calls.
//!
//! ## Content Fingerprinting
//!
//! XXH3-based fingerprinting for change detection. Used by the context engine
//! and session manager to detect when content has changed between LLM calls.

pub mod detector;
pub mod frameworks;

pub use detector::{fingerprint_project, DetectedLanguage, Language, PackageManager, ProjectFingerprint};
pub use frameworks::FrameworkInfo;

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3Default;

// ---------------------------------------------------------------------------
// Content fingerprinting (XXH3)
// ---------------------------------------------------------------------------

/// Streaming fingerprint builder using XXH3.
///
/// Feeds chunks of data directly into the XXH3 streaming hasher with zero
/// intermediate allocation. Call [`finish`] to get the 64-bit fingerprint.
/// This is used to fingerprint context packages, file contents, and
/// other data that may arrive incrementally.
pub struct Fingerprinter {
    /// Streaming XXH3 hasher -- accumulates state incrementally without
    /// buffering all input data in a `Vec<u8>`.
    hasher: Xxh3Default,
    /// Number of bytes fed so far (tracked separately since `Xxh3Default`
    /// doesn't expose its `total_len`).
    fed_bytes: u64,
}

impl Fingerprinter {
    /// Create a new fingerprinter.
    pub fn new() -> Self {
        Self {
            hasher: Xxh3Default::new(),
            fed_bytes: 0,
        }
    }

    /// Create a new fingerprinter.
    ///
    /// The `_capacity` parameter is accepted for API compatibility but is
    /// ignored -- the streaming hasher uses a fixed-size internal buffer
    /// and requires no pre-allocation.
    pub fn with_capacity(_capacity: usize) -> Self {
        Self::new()
    }

    /// Feed data into the fingerprinter.
    pub fn feed(&mut self, data: &[u8]) {
        self.hasher.update(data);
        self.fed_bytes += data.len() as u64;
    }

    /// Feed a string into the fingerprinter.
    pub fn feed_str(&mut self, s: &str) {
        self.feed(s.as_bytes());
    }

    /// Feed a u32 into the fingerprinter.
    pub fn feed_u32(&mut self, v: u32) {
        self.feed(&v.to_le_bytes());
    }

    /// Feed a u64 into the fingerprinter.
    pub fn feed_u64(&mut self, v: u64) {
        self.feed(&v.to_le_bytes());
    }

    /// Finalize and return the 64-bit fingerprint.
    pub fn finish(&self) -> u64 {
        self.hasher.digest()
    }

    /// Reset the fingerprinter for reuse.
    pub fn reset(&mut self) {
        self.hasher.reset();
        self.fed_bytes = 0;
    }

    /// Number of bytes fed so far.
    pub fn len(&self) -> usize {
        self.fed_bytes as usize
    }

    /// Whether no data has been fed.
    pub fn is_empty(&self) -> bool {
        self.fed_bytes == 0
    }
}

impl Default for Fingerprinter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// One-shot convenience functions
// ---------------------------------------------------------------------------

/// Compute the XXH3 fingerprint of a byte slice.
pub fn fingerprint_bytes(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

/// Compute the XXH3 fingerprint of a string.
pub fn fingerprint_str(s: &str) -> u64 {
    fingerprint_bytes(s.as_bytes())
}

/// Compute the XXH3 fingerprint of multiple strings concatenated.
pub fn fingerprint_strings(strings: &[&str]) -> u64 {
    let mut fp = Fingerprinter::new();
    for s in strings {
        fp.feed_str(s);
    }
    fp.finish()
}

// ---------------------------------------------------------------------------
// File fingerprint — for tracking file changes
// ---------------------------------------------------------------------------

/// A file content fingerprint with metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFingerprint {
    /// XXH3 hash of the file content.
    pub content_hash: u64,
    /// File size in bytes.
    pub size: u64,
    /// FNV hash of the file path (for indexing).
    pub path_hash: u64,
}

impl FileFingerprint {
    /// Create a fingerprint for a file.
    pub fn new(path: &str, content: &[u8]) -> Self {
        Self {
            content_hash: fingerprint_bytes(content),
            size: content.len() as u64,
            path_hash: crate::arena::fnv_hash(path),
        }
    }

    /// Check if a file has changed by comparing fingerprints.
    pub fn has_changed(&self, other: &FileFingerprint) -> bool {
        self.content_hash != other.content_hash
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_deterministic() {
        let fp1 = fingerprint_str("hello world");
        let fp2 = fingerprint_str("hello world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_input() {
        let fp1 = fingerprint_str("hello");
        let fp2 = fingerprint_str("world");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprinter_streaming() {
        let mut fp = Fingerprinter::new();
        fp.feed_str("hello ");
        fp.feed_str("world");
        let streaming = fp.finish();

        let oneshot = fingerprint_str("hello world");
        assert_eq!(streaming, oneshot);
    }

    #[test]
    fn test_fingerprinter_reset() {
        let mut fp = Fingerprinter::new();
        fp.feed_str("hello");
        let first = fp.finish();

        fp.reset();
        fp.feed_str("world");
        let second = fp.finish();

        assert_ne!(first, second);
        assert_eq!(second, fingerprint_str("world"));
    }

    #[test]
    fn test_fingerprint_strings_multi() {
        let fp = fingerprint_strings(&["hello ", "world"]);
        assert_eq!(fp, fingerprint_str("hello world"));
    }

    #[test]
    fn test_file_fingerprint() {
        let fp1 = FileFingerprint::new("src/main.rs", b"fn main() {}");
        let fp2 = FileFingerprint::new("src/main.rs", b"fn main() { println!() }");

        assert!(fp1.has_changed(&fp2));
        assert!(!fp1.has_changed(&fp1));
    }

    #[test]
    fn test_file_fingerprint_size() {
        let fp = FileFingerprint::new("test.rs", b"hello");
        assert_eq!(fp.size, 5);
    }

    #[test]
    fn test_fingerprinter_numeric_types() {
        let mut fp = Fingerprinter::new();
        fp.feed_u32(42);
        fp.feed_u64(123456789);
        assert!(fp.finish() != 0);
    }

    #[test]
    fn test_fingerprinter_empty() {
        let fp = Fingerprinter::new();
        assert!(fp.is_empty());
        let hash = fp.finish();
        assert_eq!(hash, fingerprint_bytes(&[]));
    }
}
