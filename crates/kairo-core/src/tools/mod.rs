//! Tool subsystem for KAIRO-X agent operations.
//!
//! Provides the core tool implementations that the agent uses to interact
//! with the filesystem, execute commands, apply edits, manage git state,
//! and coordinate parallel execution through file locks and snapshots.

pub mod filesystem;
pub mod search_replace;
pub mod executor;
pub mod git;
pub mod snapshots;
pub mod file_locks;
