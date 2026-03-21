//! Session management subsystem (§5).
//!
//! Manages LLM conversation sessions that persist across multiple calls.
//! Sessions are bound to arena graph nodes and track message history, token
//! usage, and quality metrics.
//!
//! - **SessionManager** (`manager.rs`): Creates, retrieves, and closes sessions.
//!   Implements the session continuation decision table from §5.2.
//! - **TokenTracker** (`token_tracker.rs`): Tracks token usage and cost budgets.
//! - **Failover** (`failover.rs`): Session-level failover state tracking.

pub mod failover;
pub mod manager;
pub mod token_tracker;

pub use failover::SessionFailoverState;
pub use manager::{Session, SessionManager, SessionManagerSnapshot, SessionSummary};
pub use token_tracker::{CostMode, TokenTracker};
