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
use bincode::Options;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

/// Maximum size (in bytes) for bincode deserialization to prevent OOM from
/// corrupted or malicious checkpoint data. 256 MB is generous enough for any
/// legitimate KAIRO-X arena while still capping memory exposure.
const MAX_DESERIALIZE_SIZE: u64 = 256 * 1024 * 1024;

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

    #[error("Unsupported checkpoint format version {found} (expected {expected})")]
    FormatVersionMismatch { expected: u32, found: u32 },

    #[error("Symlink detected at {0}, refusing to delete")]
    SymlinkRefused(PathBuf),
}

// ---------------------------------------------------------------------------
// Checkpoint metadata
// ---------------------------------------------------------------------------

/// Current checkpoint format version. Bump this when the on-disk layout changes.
const CHECKPOINT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Checkpoint format version for forward-compatibility checks.
    #[serde(default = "default_format_version")]
    pub format_version: u32,
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

/// Default for `format_version` when deserializing old meta.json files that
/// lack the field (pre-v1 checkpoints are implicitly version 0).
fn default_format_version() -> u32 {
    0
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
// Atomic file writes
// ---------------------------------------------------------------------------

/// Write `data` to `path` atomically via a temporary file + rename.
///
/// This guarantees that readers never see a half-written file: either the old
/// content is present or the new content is, never a truncated intermediate.
fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
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

        // Serialize itch register separately (for fast independent reads)
        let itch_data = bincode::serialize(&arena.itch).map_err(|e| {
            PersistenceError::Serialization(format!("Itch serialization failed: {e}"))
        })?;

        // Serialize token accounting
        let token_data = bincode::serialize(token_tracker).map_err(|e| {
            PersistenceError::Serialization(format!("Token tracker serialization failed: {e}"))
        })?;

        // Serialize compliance tracker
        let compliance_data = bincode::serialize(compliance).map_err(|e| {
            PersistenceError::Serialization(format!("Compliance serialization failed: {e}"))
        })?;

        // Serialize controller recurrent state
        let controller_state = controller.serialize_state();
        let controller_bytes = bincode::serialize(&controller_state).map_err(|e| {
            PersistenceError::Serialization(format!("Controller state serialization failed: {e}"))
        })?;

        // Serialize session summaries (metadata only, not message history)
        let session_snapshot = session_manager.snapshot();
        let sessions_bytes = bincode::serialize(&session_snapshot).map_err(|e| {
            PersistenceError::Serialization(format!("Sessions serialization failed: {e}"))
        })?;

        // Serialize snapshot index (which files have snapshots)
        let snapshot_index = SnapshotIndex {
            paths: snapshots.snapshot_paths().into_iter().cloned().collect(),
        };
        let snapshots_bytes = bincode::serialize(&snapshot_index).map_err(|e| {
            PersistenceError::Serialization(format!("Snapshot index serialization failed: {e}"))
        })?;

        // Combined hash for integrity: covers ALL serialized data.
        let mut combined = Vec::new();
        combined.extend_from_slice(&graph_bytes);
        combined.extend_from_slice(&itch_data);
        combined.extend_from_slice(&token_data);
        combined.extend_from_slice(&compliance_data);
        combined.extend_from_slice(&controller_bytes);
        combined.extend_from_slice(&sessions_bytes);
        combined.extend_from_slice(&snapshots_bytes);
        let combined_hash = xxh3_64(&combined);

        // Write data files atomically (tmp + rename). Meta is written LAST so
        // its presence serves as the commit marker for a complete checkpoint.
        atomic_write(&checkpoint_dir.join("graph.bin"), &graph_bytes)?;
        atomic_write(&checkpoint_dir.join("itch.bin"), &itch_data)?;
        atomic_write(&checkpoint_dir.join("token_accounting.bin"), &token_data)?;
        atomic_write(&checkpoint_dir.join("compliance.bin"), &compliance_data)?;
        atomic_write(&checkpoint_dir.join("controller_state.bin"), &controller_bytes)?;
        atomic_write(&checkpoint_dir.join("sessions_summary.bin"), &sessions_bytes)?;
        atomic_write(&checkpoint_dir.join("snapshots_index.bin"), &snapshots_bytes)?;

        // Write metadata LAST — its presence signals a complete checkpoint.
        let (_active_itch, _) = arena.itch_stats();
        let summary = arena.status_summary();
        let meta = CheckpointMeta {
            format_version: CHECKPOINT_FORMAT_VERSION,
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
        atomic_write(&checkpoint_dir.join("meta.json"), meta_json.as_bytes())?;

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

        // Check format version — reject unknown future versions.
        if meta.format_version > CHECKPOINT_FORMAT_VERSION {
            return Err(PersistenceError::FormatVersionMismatch {
                expected: CHECKPOINT_FORMAT_VERSION,
                found: meta.format_version,
            });
        }

        let deserializer = bincode::DefaultOptions::new()
            .with_limit(MAX_DESERIALIZE_SIZE);

        // Read core data files
        let graph_bytes = std::fs::read(checkpoint_dir.join("graph.bin"))?;
        let itch_data = std::fs::read(checkpoint_dir.join("itch.bin"))?;
        let token_data = std::fs::read(checkpoint_dir.join("token_accounting.bin"))?;
        let compliance_data = std::fs::read(checkpoint_dir.join("compliance.bin"))?;

        // Read additional files (backward-compat: may be absent in v0 checkpoints)
        let controller_data = {
            let path = checkpoint_dir.join("controller_state.bin");
            if path.exists() {
                Some(std::fs::read(&path)?)
            } else {
                tracing::debug!("controller_state.bin not found (old checkpoint format)");
                None
            }
        };

        let sessions_data = {
            let path = checkpoint_dir.join("sessions_summary.bin");
            if path.exists() {
                Some(std::fs::read(&path)?)
            } else {
                tracing::debug!("sessions_summary.bin not found (old checkpoint format)");
                None
            }
        };

        let snapshots_data = {
            let path = checkpoint_dir.join("snapshots_index.bin");
            if path.exists() {
                Some(std::fs::read(&path)?)
            } else {
                tracing::debug!("snapshots_index.bin not found (old checkpoint format)");
                None
            }
        };

        // Verify integrity: hash covers ALL serialized data.
        // For v0 checkpoints (format_version == 0) the hash only covered the
        // four core files, so we replicate that behaviour for backward compat.
        let mut combined = Vec::new();
        combined.extend_from_slice(&graph_bytes);
        combined.extend_from_slice(&itch_data);
        combined.extend_from_slice(&token_data);
        combined.extend_from_slice(&compliance_data);
        if meta.format_version >= 1 {
            if let Some(ref cd) = controller_data {
                combined.extend_from_slice(cd);
            }
            if let Some(ref sd) = sessions_data {
                combined.extend_from_slice(sd);
            }
            if let Some(ref sn) = snapshots_data {
                combined.extend_from_slice(sn);
            }
        }
        let actual_hash = xxh3_64(&combined);

        if actual_hash != meta.combined_hash {
            return Err(PersistenceError::IntegrityFailed {
                file: "combined".to_string(),
                expected: meta.combined_hash,
                actual: actual_hash,
            });
        }

        // Deserialize arena (has its own internal integrity check)
        let arena = Arena::deserialize_from_bytes(&graph_bytes).map_err(|e| {
            PersistenceError::Corrupted(format!("Arena deserialization failed: {e}"))
        })?;

        // Deserialize token tracker (size-limited)
        let token_tracker: TokenTracker = deserializer
            .deserialize(&token_data)
            .map_err(|e| {
                PersistenceError::Corrupted(format!("Token tracker deserialization failed: {e}"))
            })?;

        // Deserialize compliance tracker (size-limited)
        let compliance: ComplianceTracker = deserializer
            .deserialize(&compliance_data)
            .map_err(|e| {
                PersistenceError::Corrupted(format!(
                    "Compliance tracker deserialization failed: {e}"
                ))
            })?;

        // Deserialize additional files with size-limited bincode
        let controller_state = match controller_data {
            Some(data) => {
                deserializer.deserialize::<Vec<f32>>(&data).unwrap_or_else(|e| {
                    tracing::warn!("Failed to deserialize controller_state.bin: {e}");
                    Vec::new()
                })
            }
            None => Vec::new(),
        };

        let session_snapshot = match sessions_data {
            Some(data) => {
                match deserializer.deserialize::<SessionManagerSnapshot>(&data) {
                    Ok(snap) => Some(snap),
                    Err(e) => {
                        tracing::warn!("Failed to deserialize sessions_summary.bin: {e}");
                        None
                    }
                }
            }
            None => None,
        };

        let snapshot_index = match snapshots_data {
            Some(data) => {
                match deserializer.deserialize::<SnapshotIndex>(&data) {
                    Ok(idx) => Some(idx),
                    Err(e) => {
                        tracing::warn!("Failed to deserialize snapshots_index.bin: {e}");
                        None
                    }
                }
            }
            None => None,
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

        // Remove oldest checkpoints beyond the max.
        // Use drain(..excess) instead of repeated remove(0) to avoid O(n^2).
        let excess_count = entries.len().saturating_sub(self.max_checkpoints);
        if excess_count > 0 {
            for oldest in entries.drain(..excess_count) {
                // Refuse to follow symlinks — a symlink in the checkpoints dir
                // could point anywhere and remove_dir_all would nuke the target.
                let sym_meta = std::fs::symlink_metadata(&oldest)?;
                if sym_meta.file_type().is_symlink() {
                    return Err(PersistenceError::SymlinkRefused(oldest));
                }
                tracing::debug!(path = %oldest.display(), "Pruning old checkpoint");
                std::fs::remove_dir_all(&oldest)?;
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
            let status = match restored.arena.get(node_idx) {
                Some(node) => node.status,
                None => continue,
            };
            // Only re-verify nodes that were previously verified
            if status == crate::arena::node::NodeStatus::Verified {
                if let Some(node) = restored.arena.get_mut(node_idx) {
                    node.status = crate::arena::node::NodeStatus::AwaitingVerification;
                }
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

    for dir_result in entries {
        // Log skipped entries at debug level instead of silently swallowing.
        let entry = match dir_result {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!(
                    dir = %current.display(),
                    error = %e,
                    "Skipping unreadable directory entry"
                );
                continue;
            }
        };

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

        // Use entry.file_type() which does NOT follow symlinks (unlike
        // path.is_dir() / path.is_file() which call stat and follow them).
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %e,
                    "Skipping entry: could not read file type"
                );
                continue;
            }
        };

        // Skip symlinks entirely to avoid escaping the project tree.
        if ft.is_symlink() {
            tracing::debug!(path = %path.display(), "Skipping symlink");
            continue;
        }

        if ft.is_dir() {
            scan_dir_recursive(&path, root, since, modified)?;
        } else if ft.is_file() {
            if let Ok(metadata) = entry.metadata() {
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
        assert_eq!(restored.meta.format_version, CHECKPOINT_FORMAT_VERSION);
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
        // Simulate restoring from an old (v0) checkpoint without the new files.
        // v0 checkpoints only hash the 4 core files and lack format_version.
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 10, 5);
        let cp_path = manager
            .create_checkpoint(1, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
            .unwrap();

        // Delete the extra files to simulate an old checkpoint
        fs::remove_file(cp_path.join("controller_state.bin")).unwrap();
        fs::remove_file(cp_path.join("sessions_summary.bin")).unwrap();
        fs::remove_file(cp_path.join("snapshots_index.bin")).unwrap();

        // Rewrite meta.json with format_version=0 and a hash covering only
        // the 4 core files (which is what the v0 code produced).
        let graph_bytes = fs::read(cp_path.join("graph.bin")).unwrap();
        let itch_data = fs::read(cp_path.join("itch.bin")).unwrap();
        let token_data = fs::read(cp_path.join("token_accounting.bin")).unwrap();
        let compliance_data = fs::read(cp_path.join("compliance.bin")).unwrap();

        let mut combined = Vec::new();
        combined.extend_from_slice(&graph_bytes);
        combined.extend_from_slice(&itch_data);
        combined.extend_from_slice(&token_data);
        combined.extend_from_slice(&compliance_data);
        let v0_hash = xxh3_64(&combined);

        let meta_json = fs::read_to_string(cp_path.join("meta.json")).unwrap();
        let mut meta: CheckpointMeta = serde_json::from_str(&meta_json).unwrap();
        meta.format_version = 0;
        meta.combined_hash = v0_hash;
        let patched_json = serde_json::to_string_pretty(&meta).unwrap();
        fs::write(cp_path.join("meta.json"), patched_json).unwrap();

        // Restore should still succeed with v0 backward-compat path
        let restored = manager.restore_from(&cp_path).unwrap();
        assert_eq!(restored.meta.step, 1);
        assert_eq!(restored.meta.format_version, 0);
        assert!(restored.controller_state.is_empty());
        assert!(restored.session_snapshot.is_none());
        assert!(restored.snapshot_index.is_none());
    }

    #[test]
    fn test_format_version_check() {
        // A checkpoint with an unknown future format_version should be rejected.
        let dir = tempdir().unwrap();
        let checkpoint_base = dir.path().join("checkpoints");

        let (arena, token_tracker, compliance, controller, session_manager, snapshots) = make_test_state();

        let mut manager = CheckpointManager::new(checkpoint_base.clone(), 10, 5);
        let cp_path = manager
            .create_checkpoint(1, &arena, &token_tracker, &compliance, &controller, &session_manager, &snapshots)
            .unwrap();

        // Patch meta.json to a future version
        let meta_json = fs::read_to_string(cp_path.join("meta.json")).unwrap();
        let mut meta: CheckpointMeta = serde_json::from_str(&meta_json).unwrap();
        meta.format_version = 999;
        let patched_json = serde_json::to_string_pretty(&meta).unwrap();
        fs::write(cp_path.join("meta.json"), patched_json).unwrap();

        let result = manager.restore_from(&cp_path);
        assert!(matches!(
            result.unwrap_err(),
            PersistenceError::FormatVersionMismatch { expected: 1, found: 999 }
        ));
    }
}
