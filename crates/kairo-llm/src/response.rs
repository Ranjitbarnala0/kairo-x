//! LLM response types and the raw response structure returned by providers.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stop reason — why the LLM stopped generating
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Natural end of response
    EndTurn,
    /// Hit a stop sequence
    StopSequence,
    /// Hit the max output token limit (likely truncated)
    MaxTokens,
    /// Unknown or provider-specific reason
    Unknown,
}

impl StopReason {
    /// Whether this stop reason indicates the response was likely truncated.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::MaxTokens)
    }
}

// ---------------------------------------------------------------------------
// Raw response from provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRawResponse {
    /// The generated text content
    pub content: String,
    /// Why the LLM stopped generating
    pub stop_reason: StopReason,
    /// Number of input tokens consumed (as reported by the API)
    pub input_tokens: u32,
    /// Number of output tokens generated (as reported by the API)
    pub output_tokens: u32,
    /// Model identifier that produced this response
    pub model: String,
    /// Provider-specific response ID (for debugging/logging)
    pub response_id: Option<String>,
}

impl LLMRawResponse {
    /// Total tokens used by this call (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// ---------------------------------------------------------------------------
// Response classification — the 9 categories from §Flaw 3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResponseClass {
    /// LLM produced a plan or decomposition
    Plan,
    /// LLM produced an implementation (code)
    Implementation,
    /// Verification passed
    VerificationPass,
    /// Verification failed (issues found)
    VerificationFail,
    /// LLM is asking a clarifying question
    Question,
    /// Response appears incomplete
    Incomplete,
    /// Placeholder/stub code detected
    PlaceholderDetected,
    /// LLM reported an error
    Error,
    /// LLM refused to comply
    Refusal,
}

impl ResponseClass {
    /// Whether this classification indicates a successful, usable response.
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            Self::Plan | Self::Implementation | Self::VerificationPass
        )
    }

    /// Whether this classification indicates a failure that needs handling.
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            Self::VerificationFail
                | Self::Incomplete
                | Self::PlaceholderDetected
                | Self::Error
                | Self::Refusal
        )
    }

    /// Whether this classification means the LLM needs more info before proceeding.
    pub fn needs_interaction(&self) -> bool {
        matches!(self, Self::Question | Self::Incomplete)
    }
}

// ---------------------------------------------------------------------------
// Parsed edit blocks from implementation responses
// ---------------------------------------------------------------------------

/// A single search-and-replace edit block parsed from an LLM response.
#[derive(Debug, Clone)]
pub struct SearchReplaceBlock {
    /// The exact text to search for in the target file
    pub search: String,
    /// The replacement text
    pub replace: String,
}

/// Represents the different kinds of file modifications an LLM can produce.
#[derive(Debug, Clone)]
pub enum FileModification {
    /// Create a new file with this content
    CreateFile {
        path: String,
        content: String,
    },
    /// Edit an existing file using search-and-replace blocks
    EditFile {
        path: String,
        edits: Vec<SearchReplaceBlock>,
    },
    /// Full rewrite of an existing file
    RewriteFile {
        path: String,
        content: String,
    },
}

/// Result of parsing an implementation response.
#[derive(Debug, Clone)]
pub struct ParsedImplementation {
    /// File modifications to apply
    pub modifications: Vec<FileModification>,
    /// Any explanatory text outside of code blocks
    pub explanation: String,
}

// ---------------------------------------------------------------------------
// Context request parsed from LLM response (§Flaw 11)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ContextRequestKind {
    /// Request for a file (optionally a line range)
    File {
        path: String,
        line_start: Option<u32>,
        line_end: Option<u32>,
    },
    /// Request for a specific symbol definition
    Symbol {
        name: String,
    },
}

#[derive(Debug, Clone)]
pub struct ContextRequest {
    pub kind: ContextRequestKind,
    pub reason: Option<String>,
}
