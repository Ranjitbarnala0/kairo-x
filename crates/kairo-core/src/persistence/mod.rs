//! Persistence layer — checkpoint and resume (§15).
//!
//! Provides fast binary checkpointing of all agent state and full resume
//! with change detection. Checkpoint format:
//!
//! ```text
//! .kairo/checkpoints/step_00420/
//! ├── graph.bin                // Arena nodes + edges + side tables
//! ├── itch.bin                 // Dynamic BitVec
//! ├── controller_state.bin     // 768 × fp32 = 3 KB
//! ├── sessions_summary.bin     // Session metadata
//! ├── token_accounting.bin     // Cost tracking
//! ├── compliance.bin           // Compliance tracker state
//! ├── import_graph.bin         // Import graph
//! ├── snapshots/               // File snapshots for rollback
//! │   ├── index.bin
//! │   └── files/
//! └── meta.json                // Step, timestamp, XXH3 hash
//! ```

use crate::arena::Arena;
use crate::controller::Controller;
use crate::enforcement::compliance::ComplianceTracker;
use crate::session::manager::{SessionManager, SessionManagerSnapshot};
use crate::session::token_tracker::TokenTracker;
use crate::tools::snapshots::SnapshotStore;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Integrity check failed for {file}: expected {expected:#018x}, got {actual:#018x}")]
    IntegrityFailed {
        file: String,
        expected: u64,
        actual: u64,
    },

    #[error("Checkpoint not found at {0}")]
    NotFound(PathBuf),

    #[error("Corrupted checkpoint: {0}")]
    Corrupted(String),
}

// ---------------------------------------------------------------------------
// Checkpoint metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Step number at which this checkpoint was taken.
    pub step: u32,
    /// Timestamp of checkpoint creation.
    pub timestamp: DateTime<Utc>,
    /// XXH3 hash of the combined checkpoint data for integrity.
    pub combined_hash: u64,
    /// Number of live nodes in the arena at checkpoint time.
    pub node_count: usize,
    /// Number of completed nodes at checkpoint time.
    pub completed_count: usize,
    /// Total LLM calls made at checkpoint time.
    pub total_llm_calls: u32,
    /// Total tokens spent at checkpoint time.
    pub total_tokens: u64,
}

/// Serializable index of which files had snapshots at checkpoint time.
///
/// We only persist the file paths (not the content, which is ephemeral
/// and may be very large). On restore, this tells the runtime which
/// files were being tracked.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SnapshotIndex {
    pub paths: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Checkpoint manager
// ---------------------------------------------------------------------------

/// Manages checkpoint creation, pruning, and restoration.
pub struct CheckpointManager {
    /// Base directory for checkpoints (e.g., `.kairo/checkpoints`).
    base_dir: PathBuf,
    /// How often to checkpoint (every N steps).
    interval: u32,
    /// Maximum number of checkpoints to retain.
    max_checkpoints: usize,
    /// Step of the last checkpoint.
    last_checkpoint_step: u32,
}

impl CheckpointManager {
    pub fn new(base_dir: PathBuf, interval: u32, max_checkpoints: usize) -> Self {
        Self {
            base_dir,
            interval,
            max_checkpoints,
            last_checkpoint_step: 0,
        }
    }

    /// Whether it's time to create a checkpoint at the given step.
    pub fn should_checkpoint(&self, current_step: u32) -> bool {
        current_step >= self.last_checkpoint_step + self.interval
    }

    /// Create a checkpoint of the current agent state.
    ///
    /// Saves arena, itch, token tracker, compliance, controller state,
    /// session summaries, and snapshot index.
    #[allow(clippy::too_many_arguments)]
    pub fn create_checkpoint(
        &mut self,
        step: u32,
        arena: &Arena,
        token_tracker: &TokenTracker,
        compliance: &ComplianceTracker,
        controller: &Controller,
        session_manager: &SessionManager,
        snapshots: &SnapshotStore,
    ) -> Result<PathBuf, PersistenceError> {
        let checkpoint_dir = self.base_dir.join(format!("step_{step:05}"));
        std::fs::create_dir_all(&checkpoint_dir)?;

        // Serialize arena graph
        let graph_bytes = arena.serialize_to_bytes().map_err(|e| {
            PersistenceError::Serialization(format!("Arena serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("graph.bin"), &graph_bytes)?;

        // Serialize itch register separately (for fast independent reads)
        let itch_data = bincode::serialize(&arena.itch).map_err(|e| {
            PersistenceError::Serialization(format!("Itch serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("itch.bin"), &itch_data)?;

        // Serialize token accounting
        let token_data = bincode::serialize(token_tracker).map_err(|e| {
            PersistenceError::Serialization(format!("Token tracker serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("token_accounting.bin"), &token_data)?;

        // Serialize compliance tracker
        let compliance_data = bincode::serialize(compliance).map_err(|e| {
            PersistenceError::Serialization(format!("Compliance serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("compliance.bin"), &compliance_data)?;

        // Serialize controller recurrent state
        let controller_state = controller.serialize_state();
        let controller_bytes = bincode::serialize(&controller_state).map_err(|e| {
            PersistenceError::Serialization(format!("Controller state serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("controller_state.bin"), &controller_bytes)?;

        // Serialize session summaries (metadata only, not message history)
        let session_snapshot = session_manager.snapshot();
        let sessions_bytes = bincode::serialize(&session_snapshot).map_err(|e| {
            PersistenceError::Serialization(format!("Sessions serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("sessions_summary.bin"), &sessions_bytes)?;

        // Serialize snapshot index (which files have snapshots)
        let snapshot_index = SnapshotIndex {
            paths: snapshots.snapshot_paths().into_iter().cloned().collect(),
        };
        let snapshots_bytes = bincode::serialize(&snapshot_index).map_err(|e| {
            PersistenceError::Serialization(format!("Snapshot index serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("snapshots_index.bin"), &snapshots_bytes)?;

        // Combined hash for integrity (over the core files that existed before)
        let mut combined = Vec::new();
        combined.extend_from_slice(&graph_bytes);
        combined.extend_from_slice(&itch_data);
        combined.extend_from_slice(&token_data);
        combined.extend_from_slice(&compliance_data);
        let combined_hash = xxh3_64(&combined);

        // Write metadata
        let (_active_itch, _) = arena.itch_stats();
        let summary = arena.status_summary();
        let meta = CheckpointMeta {
            step,
            timestamp: Utc::now(),
            combined_hash,
            node_count: arena.live_count(),
            completed_count: summary.completed,
            total_llm_calls: arena.total_llm_calls(),
            total_tokens: token_tracker.total_input + token_tracker.total_output,
        };

        let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| {
            PersistenceError::Serialization(format!("Meta serialization failed: {e}"))
        })?;
        std::fs::write(checkpoint_dir.join("meta.json"), meta_json)?;

        self.last_checkpoint_step = step;

        // Prune old checkpoints
        self.prune_old_checkpoints()?;

        tracing::info!(
            step,
            nodes = meta.node_count,
            completed = meta.completed_count,
            "Checkpoint created"
        );

        Ok(checkpoint_dir)
    }

    /// Restore agent state from the latest (or specified) checkpoint.
    pub fn restore_latest(&self) -> Result<RestoredState, PersistenceError> {
        let latest = self.find_latest_checkpoint()?;
        self.restore_from(&latest)
    }

    /// Restore from a specific checkpoint directory.
    pub fn restore_from(&self, checkpoint_dir: &Path) -> Result<RestoredState, PersistenceError> {
        if !checkpoint_dir.exists() {
            return Err(PersistenceError::NotFound(checkpoint_dir.to_path_buf()));
        }

        // Read metadata
        let meta_path = checkpoint_dir.join("meta.json");
        let meta_json = std::fs::read_to_string(&meta_path)?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_json).map_err(|e| {
            PersistenceError::Corrupted(format!("Invalid meta.json: {e}"))
        })?;

        // Read and verify all data
        let graph_bytes = std::fs::read(checkpoint_dir.join("graph.bin"))?;
        let itch_data = std::fs::read(checkpoint_dir.join("itch.bin"))?;
        let token_data = std::fs::read(checkpoint_dir.join("token_accounting.bin"))?;
        let compliance_data = std::fs::read(checkpoint_dir.join("compliance.bin"))?;

        // Verify integrity
        let mut combined = Vec::new();
        combined.extend_from_slice(&graph_bytes);
        combined.extend_from_slice(&itch_data);
        combined.extend_from_slice(&token_data);
        combined.extend_from_slice(&compliance_data);
        let actual_hash = xxh3_64(&combined);

        if actual_hash != meta.combined_hash {
            return Err(PersistenceError::IntegrityFailed {
                file: "combined".to_string(),
                expected: meta.combined_hash,
                actual: actual_hash,
            });
        }

        // Deserialize arena
        let arena = Arena::deserialize_from_bytes(&graph_bytes).map_err(|e| {
            PersistenceError::Corrupted(format!("Arena deserialization failed: {e}"))
        })?;

        // Deserialize token tracker
        let token_tracker: TokenTracker = bincode::deserialize(&token_data).map_err(|e| {
            PersistenceError::Corrupted(format!("Token tracker deserialization failed: {e}"))
        })?;

        // Deserialize compliance tracker
        let compliance: ComplianceTracker =
            bincode::deserialize(&compliance_data).map_err(|e| {
                PersistenceError::Corrupted(format!(
                    "Compliance tracker deserialization failed: {e}"
                ))
            })?;

        // Load additional files gracefully (backward compat with old checkpoints)
        let controller_state = {
            let path = checkpoint_dir.join("controller_state.bin");
            if path.exists() {
                let data = std::fs::read(&path)?;
                bincode::deserialize::<Vec<f32>>(&data).unwrap_or_else(|e| {
                    tracing::warn!("Failed to deserialize controller_state.bin: {e}");
                    Vec::new()
                })
            } else {
                tracing::debug!("controller_state.bin not found (old checkpoint format)");
                Vec::new()
            }
        };

        let session_snapshot = {
            let path = checkpoint_dir.join("sessions_summary.bin");
            if path.exists() {
                let data = std::fs::read(&path)?;
                match bincode::deserialize::<SessionManagerSnapshot>(&data) {
                    Ok(snap) => Some(snap),
                    Err(e) => {
                        tracing::warn!("Failed to deserialize sessions_summary.bin: {e}");
                        None
                    }
                }
            } else {
                tracing::debug!("sessions_summary.bin not found (old checkpoint format)");
                None
            }
        };

        let snapshot_index = {
            let path = checkpoint_dir.join("snapshots_index.bin");
            if path.exists() {
                let data = std::fs::read(&path)?;
                match bincode::deserialize::<SnapshotIndex>(&data) {
                    Ok(idx) => Some(idx),
                    Err(e) => {
                        tracing::warn!("Failed to deserialize snapshots_index.bin: {e}");
                        None
                    }
                }
            } else {
                tracing::debug!("snapshots_index.bin not found (old checkpoint format)");
                None
            }
        };

        tracing::info!(
            step = meta.step,
            nodes = meta.node_count,
            completed = meta.completed_count,
            "Checkpoint restored"
        );

        Ok(RestoredState {
            meta,
            arena,
            token_tracker,
            compliance,
            controller_state,
            session_snapshot,
            snapshot_index,
        })
    }

    /// Find the latest checkpoint directory.
    fn find_latest_checkpoint(&self) -> Result<PathBuf, PersistenceError> {
        if !self.base_dir.exists() {
            return Err(PersistenceError::NotFound(self.base_dir.clone()));
        }

        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.base_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
            })
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("step_"))
            })
            .map(|e| e.path())
            .collect();

        entries.sort();

        entries
            .last()
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound(self.base_dir.clone()))
    }

    /// Prune old checkpoints, keeping only the most recent `max_checkpoints`.
    fn prune_old_checkpoints(&self) -> Result<(), PersistenceError> {
        if !self.base_dir.exists() {
            return Ok(());
        }

        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.base_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
            })
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("step_"))
            })
            .map(|e| e.path())
            .collect();

        entries.sort();

        // Remove oldest checkpoints beyond the max
        while entries.len() > self.max_checkpoints {
            if let Some(oldest) = entries.first() {
                tracing::debug!(path = %oldest.display(), "Pruning old checkpoint");
                std::fs::remove_dir_all(oldest)?;
                entries.remove(0);
            }
        }

        Ok(())
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointMeta>, PersistenceError> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut metas = Vec::new();

        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.base_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        entries.sort();

        for entry in entries {
            let meta_path = entry.join("meta.json");
            if meta_path.exists() {
                if let Ok(json) = std::fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<CheckpointMeta>(&json) {
                        metas.push(meta);
                    }
                }
            }
        }

        Ok(metas)
    }
}

// ---------------------------------------------------------------------------
// Restored state
// ---------------------------------------------------------------------------

/// Complete agent state restored from a checkpoint.
pub struct RestoredState {
    pub meta: CheckpointMeta,
    pub arena: Arena,
    pub token_tracker: TokenTracker,
    pub compliance: ComplianceTracker,
    /// Controller recurrent state vector (may be empty for old checkpoints).
    pub controller_state: Vec<f32>,
    /// Session manager snapshot (may be None for old checkpoints).
    pub session_snapshot: Option<SessionManagerSnapshot>,
    /// Snapshot index (may be None for old checkpoints).
    pub snapshot_index: Option<SnapshotIndex>,
}

// ---------------------------------------------------------------------------
// Resume protocol (§15.2)
// ---------------------------------------------------------------------------

/// Execute the resume protocol after restoring from a checkpoint.
///
/// 1. Load checkpoint (already done by restore).
/// 2. Scan project for file changes since checkpoint.
/// 3. Flag affected nodes for re-verification.
pub fn resume_with_change_detection(
    restored: &mut RestoredState,
    project_root: &Path,
) -> Result<Vec<u32>, PersistenceError> {
    let checkpoint_time = restored.meta.timestamp;
    let mut affected_nodes = Vec::new();

    // Scan for files modified since checkpoint
    let modified_files = scan_modified_files(project_root, checkpoint_time)?;

    // Find graph nodes that touch modified files and flag them for re-verification.
    // We collect the touching node indices first to avoid holding an immutable
    // borrow on `restored.arena` while we mutate it.
    for file_path in &modified_files {
        let touching: Vec<u32> = restored
            .arena
            .nodes_touching_file(file_path)
            .to_vec();
        for node_idx in touching {
            let node = restored.arena.get(node_idx);
            // Only re-verify nodes that were previously verified
            if node.status == crate::arena::node::NodeStatus::Verified {
                restored.arena.get_mut(node_idx).status =
                    crate::arena::node::NodeStatus::AwaitingVerification;
                // Re-set itch bit so this node blocks termination
                restored.arena.itch.set(node_idx as usize);
                if !affected_nodes.contains(&node_idx) {
                    affected_nodes.push(node_idx);
                }
            }
        }
    }

    if !affected_nodes.is_empty() {
        tracing::warn!(
            count = affected_nodes.len(),
            "Nodes flagged for re-verification due to external file changes"
        );
    }

    Ok(affected_nodes)
}

/// Scan a directory tree for files modified after the given timestamp.
fn scan_modified_files(
    root: &Path,
    since: DateTime<Utc>,
) -> Result<Vec<String>, PersistenceError> {
    let mut modified = Vec::new();
    let since_system = std::time::SystemTime::from(since);

    if !root.exists() {
        return Ok(modified);
    }

    scan_dir_recursive(root, root, &since_system, &mut modified)?;
    Ok(modified)
}

fn scan_dir_recursive(
    current: &Path,
    root: &Path,
    since: &std::time::SystemTime,
    modified: &mut Vec<String>,
) -> Result<(), PersistenceError> {
    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return Ok(()), // Skip unreadable directories
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden dirs, node_modules, target, .git, .kairo
        if name_str.starts_with('.')
            || name_str == "node_modules"
            || name_str == "target"
            || name_str == "__pycache__"
            || name_str == ".kairo"
        {
            continue;
        }

        if path.is_dir() {
            scan_dir_recursive(&path, root, since, modified)?;
        } else if path.is_file() {
            if let Ok(metadata) = path.metadata() {
                if let Ok(mod_time) = metadata.modified() {
                    if mod_time > *since {
                        // Convert to relative path string
                        if let Ok(rel) = path.strip_prefix(root) {
                            modified.push(rel.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::{Node, Priority};
    use crate::controller::Controller;
    use crate::session::manager::SessionManager;
    use crate::session::token_tracker::CostMode;
    use crate::tools::snapshots::SnapshotStore;
    use std::fs;
    use tempfile::tempdir;

    fn make_test_state() -> (Arena, TokenTracker, ComplianceTracker, Controller, SessionManager, SnapshotStore) {
        let mut arena = Arena::new();
        let n1 = arena.alloc(Node::new("Auth Service".to_string(), Priority::Critical));
        arena.set_spec(n1, "Implement JWT auth".to_string());
        arena.mark_complete(n1);

        let n2 = arena.alloc(Node::new("Rate Limiter".to_string(), Priority::Standard));
        arena.set_spec(n2, "Sliding window rate limiter".to_string());

        let token_tracker = TokenTracker::new(0, 0, 0, 0, CostMode::Balanced);
        let compliance = ComplianceTracker::new();
        let controller = Controller::zeros();
        let session_manager = SessionManager::new();
        let snapshots = SnapshotStore::new();

        (arena, token_tracker, compliance, controller, session_manager, snapshots)
    }

    #[test]
    fn test_checkpoint_round_trip() {
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 10, 5);

        // Create checkpoint
        let cp_path = manager
            .create_checkpoint(42, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
            .unwrap();

        assert!(cp_path.exists());
        assert!(cp_path.join("graph.bin").exists());
        assert!(cp_path.join("meta.json").exists());
        assert!(cp_path.join("controller_state.bin").exists());
        assert!(cp_path.join("sessions_summary.bin").exists());
        assert!(cp_path.join("snapshots_index.bin").exists());

        // Restore
        let restored = manager.restore_from(&cp_path).unwrap();
        assert_eq!(restored.meta.step, 42);
        assert_eq!(restored.arena.live_count(), 2);
        assert!(!restored.controller_state.is_empty());
        assert!(restored.session_snapshot.is_some());
        assert!(restored.snapshot_index.is_some());
    }

    #[test]
    fn test_checkpoint_pruning() {
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 1, 3);

        // Create 5 checkpoints
        for step in 1..=5 {
            manager
                .create_checkpoint(step, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
                .unwrap();
            manager.last_checkpoint_step = 0; // Reset to allow next
        }

        // Should only have 3 remaining (max_checkpoints = 3)
        let metas = manager.list_checkpoints().unwrap();
        assert_eq!(metas.len(), 3);
        assert_eq!(metas[0].step, 3); // Oldest remaining
        assert_eq!(metas[2].step, 5); // Newest
    }

    #[test]
    fn test_integrity_check() {
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 10, 5);
        let cp_path = manager
            .create_checkpoint(1, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
            .unwrap();

        // Corrupt the graph.bin file
        let graph_path = cp_path.join("graph.bin");
        let mut data = fs::read(&graph_path).unwrap();
        let mid = data.len() / 2;
        data[mid] ^= 0xFF;
        fs::write(&graph_path, &data).unwrap();

        // Restore should fail with integrity error
        let result = manager.restore_from(&cp_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_should_checkpoint() {
        let manager = CheckpointManager::new(PathBuf::from("/tmp/test"), 10, 5);
        assert!(!manager.should_checkpoint(5));
        assert!(manager.should_checkpoint(10));
        assert!(manager.should_checkpoint(15));
    }

    #[test]
    fn test_backward_compat_restore() {
        // Simulate restoring from an old checkpoint without the new files
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 10, 5);
        let cp_path = manager
            .create_checkpoint(1, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
            .unwrap();

        // Delete the new files to simulate an old checkpoint
        fs::remove_file(cp_path.join("controller_state.bin")).unwrap();
        fs::remove_file(cp_path.join("sessions_summary.bin")).unwrap();
        fs::remove_file(cp_path.join("snapshots_index.bin")).unwrap();

        // Restore should still succeed
        let restored = manager.restore_from(&cp_path).unwrap();
        assert_eq!(restored.meta.step, 1);
        assert!(restored.controller_state.is_empty());
        assert!(restored.session_snapshot.is_none());
        assert!(restored.snapshot_index.is_none());
    }
}
