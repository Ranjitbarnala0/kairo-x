//! Node schema for the execution graph (§4.2).
//!
//! Each node represents an implementable component in the execution plan.
//! Edges (dependencies, dependents, children) are stored as inline SmallVecs
//! for cache locality — most nodes have <8 edges.

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Priority — determines execution order and verification depth
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Highest priority — foundational components. Full verification.
    Critical = 2,
    /// Normal priority — most components. Standard verification.
    Standard = 1,
    /// Lowest priority — boilerplate, config, simple wiring. Light verification.
    Mechanical = 0,
}

impl Priority {
    /// Base enforcement intensity for this priority level.
    pub fn base_intensity(&self) -> f32 {
        match self {
            Self::Critical => 0.9,
            Self::Standard => 0.6,
            Self::Mechanical => 0.3,
        }
    }

    /// Default maximum retries for nodes at this priority.
    pub fn default_max_retries(&self) -> u8 {
        match self {
            Self::Critical => 7,
            Self::Standard => 5,
            Self::Mechanical => 3,
        }
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "critical"),
            Self::Standard => write!(f, "standard"),
            Self::Mechanical => write!(f, "mechanical"),
        }
    }
}

impl std::str::FromStr for Priority {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "critical" => Ok(Self::Critical),
            "standard" => Ok(Self::Standard),
            "mechanical" => Ok(Self::Mechanical),
            other => Err(format!("Unknown priority: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Node status — lifecycle state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node created, waiting for dependencies to resolve.
    Pending,
    /// Dependencies resolved, ready for implementation.
    Ready,
    /// Currently being implemented by an LLM call.
    Implementing,
    /// Implementation exists, awaiting verification.
    AwaitingVerification,
    /// Deterministic verification (L1) in progress.
    VerifyingDeterministic,
    /// LLM audit (L2) in progress.
    VerifyingAudit,
    /// Verification failed, needs fix.
    FixNeeded,
    /// Fix in progress.
    Fixing,
    /// All verification passed. Terminal success state.
    Verified,
    /// Failed after max retries. Terminal failure state.
    Failed,
    /// Being decomposed into sub-nodes.
    Decomposing,
    /// Node has been deallocated and its slot is in the free list.
    Deallocated,
}

impl NodeStatus {
    /// Whether this node is in a terminal state (no more work to do).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Verified | Self::Failed | Self::Deallocated)
    }

    /// Whether this node is actively being worked on.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Implementing
                | Self::AwaitingVerification
                | Self::VerifyingDeterministic
                | Self::VerifyingAudit
                | Self::Fixing
                | Self::Decomposing
        )
    }
}

// ---------------------------------------------------------------------------
// Deterministic verdict (Layer 1 verification result)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterministicVerdict {
    /// Not yet run.
    NotRun,
    /// All deterministic checks passed.
    Pass,
    /// At least one check failed.
    Fail,
    /// No deterministic checks available for this project.
    Unavailable,
}

// ---------------------------------------------------------------------------
// LLM verdict (Layer 2 verification result)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMVerdict {
    /// Not yet run.
    NotRun,
    /// LLM audit passed.
    Pass,
    /// LLM audit found issues.
    Fail,
    /// Skipped (e.g., Efficient mode for Mechanical priority).
    Skipped,
    /// No LLM audit available (no provider configured, etc.).
    Unavailable,
}

// ---------------------------------------------------------------------------
// Node — the core graph element
// ---------------------------------------------------------------------------

/// A single node in the execution graph, representing one implementable component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    // -- Identity --

    /// Unique index in the arena. Set by `Arena::alloc`.
    pub id: u32,
    /// Parent node index. 0 = root (top-level component).
    pub parent: u32,

    // -- Content --

    /// Short descriptive title. Stack-allocated up to 128 bytes via CompactString.
    pub title: CompactString,
    /// Hash of the full specification (actual spec stored in arena side table).
    pub spec_hash: u64,
    /// Execution priority.
    pub priority: Priority,

    // -- State --

    /// Current lifecycle status.
    pub status: NodeStatus,
    /// Index into the dynamic itch register.
    pub itch_bit: u32,

    // -- Edges (inline SmallVec for cache locality) --

    /// Nodes this depends on (must be completed before this can start).
    pub dependencies: SmallVec<[u32; 4]>,
    /// Nodes that depend on this (blocked until this completes).
    pub dependents: SmallVec<[u32; 4]>,
    /// Child nodes (sub-components created by decomposition).
    pub children: SmallVec<[u32; 8]>,

    // -- Implementation --

    /// FNV hashes of file paths this node creates or modifies.
    pub impl_files: SmallVec<[u64; 4]>,

    // -- Verification --

    /// How many times this node has been retried after failure.
    pub retry_count: u8,
    /// Maximum retry attempts before marking as failed.
    pub max_retries: u8,
    /// Total LLM calls spent on this node.
    pub llm_calls_spent: u16,
    /// Result of deterministic verification (Layer 1).
    pub det_verdict: DeterministicVerdict,
    /// Result of LLM audit verification (Layer 2).
    pub llm_verdict: LLMVerdict,

    // -- Timing --

    /// Step at which this node was created.
    pub created_step: u32,
    /// Step at which this node was completed (0 = not yet).
    pub completed_step: u32,

    // -- Session --

    /// Bound LLM session ID (0 = no active session).
    pub session_id: u32,
}

impl Node {
    /// Create a new node with the given title and priority.
    pub fn new(title: String, priority: Priority) -> Self {
        let max_retries = priority.default_max_retries();
        Self {
            id: 0, // set by Arena::alloc
            parent: 0,
            title: CompactString::from(title),
            spec_hash: 0,
            priority,
            status: NodeStatus::Pending,
            itch_bit: 0,
            dependencies: SmallVec::new(),
            dependents: SmallVec::new(),
            children: SmallVec::new(),
            impl_files: SmallVec::new(),
            retry_count: 0,
            max_retries,
            llm_calls_spent: 0,
            det_verdict: DeterministicVerdict::NotRun,
            llm_verdict: LLMVerdict::NotRun,
            created_step: 0,
            completed_step: 0,
            session_id: 0,
        }
    }

    /// Create the sentinel root node (index 0).
    pub(crate) fn root() -> Self {
        Self {
            id: 0,
            parent: 0,
            title: CompactString::from("__root__"),
            spec_hash: 0,
            priority: Priority::Critical,
            status: NodeStatus::Verified, // Root is always "done"
            itch_bit: 0,
            dependencies: SmallVec::new(),
            dependents: SmallVec::new(),
            children: SmallVec::new(),
            impl_files: SmallVec::new(),
            retry_count: 0,
            max_retries: 0,
            llm_calls_spent: 0,
            det_verdict: DeterministicVerdict::NotRun,
            llm_verdict: LLMVerdict::NotRun,
            created_step: 0,
            completed_step: 0,
            session_id: 0,
        }
    }

    /// Whether this node can be retried (hasn't exceeded max retries).
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Whether this node has any unresolved children.
    pub fn has_unresolved_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Record an LLM call against this node.
    pub fn record_llm_call(&mut self) {
        self.llm_calls_spent = self.llm_calls_spent.saturating_add(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new("JWT token service".to_string(), Priority::Critical);
        assert_eq!(node.title.as_str(), "JWT token service");
        assert_eq!(node.priority, Priority::Critical);
        assert_eq!(node.status, NodeStatus::Pending);
        assert_eq!(node.max_retries, 7); // Critical default
        assert!(node.dependencies.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::Standard);
        assert!(Priority::Standard > Priority::Mechanical);
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!("critical".parse::<Priority>().unwrap(), Priority::Critical);
        assert_eq!("standard".parse::<Priority>().unwrap(), Priority::Standard);
        assert_eq!("mechanical".parse::<Priority>().unwrap(), Priority::Mechanical);
        assert!("unknown".parse::<Priority>().is_err());
    }

    #[test]
    fn test_node_status_terminal() {
        assert!(NodeStatus::Verified.is_terminal());
        assert!(NodeStatus::Failed.is_terminal());
        assert!(!NodeStatus::Pending.is_terminal());
        assert!(!NodeStatus::Implementing.is_terminal());
    }

    #[test]
    fn test_node_retry_limit() {
        let mut node = Node::new("test".to_string(), Priority::Mechanical);
        assert!(node.can_retry()); // max_retries = 3
        node.retry_count = 3;
        assert!(!node.can_retry());
    }

    #[test]
    fn test_compact_string_efficiency() {
        // CompactString stores short strings inline (no heap allocation)
        let node = Node::new("Short".to_string(), Priority::Standard);
        // This should be stack-allocated in CompactString
        assert_eq!(node.title.as_str(), "Short");
    }
}
