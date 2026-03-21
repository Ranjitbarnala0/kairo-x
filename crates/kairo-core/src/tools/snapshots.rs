//! File snapshot system for pre-edit rollback.
//!
//! Before modifying a file, the agent snapshots its current content into the
//! `SnapshotStore`. If verification fails or the edit needs to be undone, the
//! original content can be restored instantly without touching git.
//!
//! This provides a fast, in-memory rollback mechanism that works independently
//! of the git history. Snapshots are ephemeral — they live only for the
//! duration of the current session.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the snapshot system.
#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("no snapshot exists for: {0}")]
    NoSnapshot(PathBuf),

    #[error("failed to read file for snapshotting: {path}: {source}")]
    ReadFailed {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to write file during restore: {path}: {source}")]
    WriteFailed {
        path: PathBuf,
        source: std::io::Error,
    },
}

// ---------------------------------------------------------------------------
// Snapshot entry
// ---------------------------------------------------------------------------

/// A single file snapshot.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The original file content at the time of snapshotting.
    pub content: Vec<u8>,
    /// Whether the file existed when the snapshot was taken.
    /// If false, restoring means deleting the file.
    pub existed: bool,
    /// Timestamp when the snapshot was taken.
    pub taken_at: std::time::Instant,
}

// ---------------------------------------------------------------------------
// SnapshotStore
// ---------------------------------------------------------------------------

/// In-memory store mapping file paths to their pre-edit content.
///
/// Thread-safety: `SnapshotStore` is designed for single-threaded use within
/// one track. For multi-track scenarios, each track should have its own store,
/// or wrap this in an `Arc<Mutex<_>>`.
#[derive(Debug)]
pub struct SnapshotStore {
    /// Map of canonical file paths to their snapshots.
    snapshots: HashMap<PathBuf, Snapshot>,
}

impl SnapshotStore {
    /// Create a new empty snapshot store.
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
        }
    }

    /// Create a snapshot store with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            snapshots: HashMap::with_capacity(capacity),
        }
    }

    /// Snapshot a file's current content.
    ///
    /// If a snapshot already exists for this path, it is **not** overwritten
    /// — we preserve the earliest (original) state for correct rollback.
    /// Use `force_snapshot` to overwrite an existing snapshot.
    pub fn snapshot_file(&mut self, path: &Path) -> Result<(), SnapshotError> {
        // Canonicalize the path for consistent keys.
        let canonical = normalize_path(path);

        if self.snapshots.contains_key(&canonical) {
            debug!(path = %canonical.display(), "snapshot already exists, preserving original");
            return Ok(());
        }

        let (content, existed) = if path.exists() {
            let bytes = std::fs::read(path).map_err(|e| SnapshotError::ReadFailed {
                path: path.to_path_buf(),
                source: e,
            })?;
            (bytes, true)
        } else {
            (Vec::new(), false)
        };

        debug!(
            path = %canonical.display(),
            size = content.len(),
            existed,
            "file snapshotted"
        );

        self.snapshots.insert(
            canonical,
            Snapshot {
                content,
                existed,
                taken_at: std::time::Instant::now(),
            },
        );

        Ok(())
    }

    /// Snapshot a file, overwriting any existing snapshot.
    pub fn force_snapshot(&mut self, path: &Path) -> Result<(), SnapshotError> {
        let canonical = normalize_path(path);
        self.snapshots.remove(&canonical);
        self.snapshot_file(path)
    }

    /// Restore a file to its snapshotted state.
    ///
    /// If the file did not exist at snapshot time, it is deleted.
    /// If the file existed, its content is overwritten with the snapshot.
    pub fn restore_file(&mut self, path: &Path) -> Result<(), SnapshotError> {
        let canonical = normalize_path(path);

        let snapshot = self
            .snapshots
            .get(&canonical)
            .ok_or_else(|| SnapshotError::NoSnapshot(canonical.clone()))?;

        if snapshot.existed {
            // Ensure parent directory exists.
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| SnapshotError::WriteFailed {
                        path: path.to_path_buf(),
                        source: e,
                    })?;
                }
            }

            std::fs::write(path, &snapshot.content).map_err(|e| SnapshotError::WriteFailed {
                path: path.to_path_buf(),
                source: e,
            })?;
            debug!(path = %canonical.display(), "file restored from snapshot");
        } else {
            // File didn't exist before — delete it if it exists now.
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| SnapshotError::WriteFailed {
                    path: path.to_path_buf(),
                    source: e,
                })?;
                debug!(path = %canonical.display(), "file deleted (did not exist at snapshot time)");
            }
        }

        Ok(())
    }

    /// Restore all snapshotted files to their original state.
    ///
    /// Continues restoring remaining files even if one fails, collecting
    /// all errors.
    pub fn restore_all(&mut self) -> Vec<SnapshotError> {
        let paths: Vec<PathBuf> = self.snapshots.keys().cloned().collect();
        let mut errors = Vec::new();

        for canonical in &paths {
            if let Err(e) = self.restore_single(canonical) {
                warn!(path = %canonical.display(), error = %e, "failed to restore snapshot");
                errors.push(e);
            }
        }

        if errors.is_empty() {
            debug!(count = paths.len(), "all snapshots restored successfully");
        } else {
            warn!(
                total = paths.len(),
                failed = errors.len(),
                "some snapshots failed to restore"
            );
        }

        errors
    }

    /// Internal: restore a single file by its canonical path.
    fn restore_single(&self, canonical: &Path) -> Result<(), SnapshotError> {
        let snapshot = self
            .snapshots
            .get(canonical)
            .ok_or_else(|| SnapshotError::NoSnapshot(canonical.to_path_buf()))?;

        if snapshot.existed {
            if let Some(parent) = canonical.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| SnapshotError::WriteFailed {
                        path: canonical.to_path_buf(),
                        source: e,
                    })?;
                }
            }

            std::fs::write(canonical, &snapshot.content).map_err(|e| {
                SnapshotError::WriteFailed {
                    path: canonical.to_path_buf(),
                    source: e,
                }
            })?;
        } else if canonical.exists() {
            std::fs::remove_file(canonical).map_err(|e| SnapshotError::WriteFailed {
                path: canonical.to_path_buf(),
                source: e,
            })?;
        }

        Ok(())
    }

    /// Check whether a snapshot exists for the given path.
    pub fn has_snapshot(&self, path: &Path) -> bool {
        let canonical = normalize_path(path);
        self.snapshots.contains_key(&canonical)
    }

    /// Discard the snapshot for a path (e.g., after successful verification).
    ///
    /// Returns `true` if a snapshot was discarded, `false` if none existed.
    pub fn discard(&mut self, path: &Path) -> bool {
        let canonical = normalize_path(path);
        let removed = self.snapshots.remove(&canonical).is_some();
        if removed {
            debug!(path = %canonical.display(), "snapshot discarded");
        }
        removed
    }

    /// Discard all snapshots.
    pub fn discard_all(&mut self) {
        let count = self.snapshots.len();
        self.snapshots.clear();
        debug!(count, "all snapshots discarded");
    }

    /// Number of active snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Total memory used by snapshot content (bytes).
    pub fn memory_usage(&self) -> usize {
        self.snapshots.values().map(|s| s.content.len()).sum()
    }

    /// List all paths that have snapshots.
    pub fn snapshot_paths(&self) -> Vec<&PathBuf> {
        self.snapshots.keys().collect()
    }

    /// Get the snapshot content for a path (for inspection, not restoration).
    pub fn get_snapshot_content(&self, path: &Path) -> Option<&[u8]> {
        let canonical = normalize_path(path);
        self.snapshots.get(&canonical).map(|s| s.content.as_slice())
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalize a path for use as a consistent map key.
///
/// Attempts canonicalization; falls back to the path as-is if the file
/// doesn't exist yet (e.g., for new files that will be created).
fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kairo_snapshot_test_{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_snapshot_and_restore() {
        let dir = temp_dir();
        let file = dir.join("test.txt");
        fs::write(&file, "original content").unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        // Modify the file.
        fs::write(&file, "modified content").unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "modified content");

        // Restore from snapshot.
        store.restore_file(&file).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "original content");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_preserves_original() {
        let dir = temp_dir();
        let file = dir.join("preserve.txt");
        fs::write(&file, "first").unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        // Modify and snapshot again — should keep the first snapshot.
        fs::write(&file, "second").unwrap();
        store.snapshot_file(&file).unwrap();

        // Restore should give us "first", not "second".
        store.restore_file(&file).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "first");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_force_snapshot() {
        let dir = temp_dir();
        let file = dir.join("force.txt");
        fs::write(&file, "first").unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        fs::write(&file, "second").unwrap();
        store.force_snapshot(&file).unwrap();

        // Restore should give "second" since we force-updated.
        store.restore_file(&file).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "second");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_nonexistent_file() {
        let dir = temp_dir();
        let file = dir.join("nonexistent.txt");

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        // Create the file.
        fs::write(&file, "new content").unwrap();
        assert!(file.exists());

        // Restore should delete the file.
        store.restore_file(&file).unwrap();
        assert!(!file.exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_no_snapshot() {
        let mut store = SnapshotStore::new();
        let result = store.restore_file(Path::new("/nonexistent/path"));
        assert!(matches!(result, Err(SnapshotError::NoSnapshot(_))));
    }

    #[test]
    fn test_has_snapshot() {
        let dir = temp_dir();
        let file = dir.join("check.txt");
        fs::write(&file, "content").unwrap();

        let mut store = SnapshotStore::new();
        assert!(!store.has_snapshot(&file));

        store.snapshot_file(&file).unwrap();
        assert!(store.has_snapshot(&file));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_discard() {
        let dir = temp_dir();
        let file = dir.join("discard.txt");
        fs::write(&file, "content").unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();
        assert!(store.has_snapshot(&file));

        let removed = store.discard(&file);
        assert!(removed);
        assert!(!store.has_snapshot(&file));

        // Discard again — should return false.
        let removed_again = store.discard(&file);
        assert!(!removed_again);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_discard_all() {
        let dir = temp_dir();
        for i in 0..5 {
            let file = dir.join(format!("file_{i}.txt"));
            fs::write(&file, format!("content {i}")).unwrap();

            let mut store = SnapshotStore::new();
            store.snapshot_file(&file).unwrap();
        }

        let mut store = SnapshotStore::new();
        for i in 0..5 {
            let file = dir.join(format!("file_{i}.txt"));
            store.snapshot_file(&file).unwrap();
        }

        assert_eq!(store.count(), 5);
        store.discard_all();
        assert_eq!(store.count(), 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_all() {
        let dir = temp_dir();
        let mut store = SnapshotStore::new();

        // Create and snapshot 3 files.
        for i in 0..3 {
            let file = dir.join(format!("file_{i}.txt"));
            fs::write(&file, format!("original_{i}")).unwrap();
            store.snapshot_file(&file).unwrap();
        }

        // Modify all files.
        for i in 0..3 {
            let file = dir.join(format!("file_{i}.txt"));
            fs::write(&file, format!("modified_{i}")).unwrap();
        }

        // Restore all.
        let errors = store.restore_all();
        assert!(errors.is_empty());

        // Verify all restored.
        for i in 0..3 {
            let file = dir.join(format!("file_{i}.txt"));
            assert_eq!(
                fs::read_to_string(&file).unwrap(),
                format!("original_{i}")
            );
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_memory_usage() {
        let dir = temp_dir();
        let mut store = SnapshotStore::new();

        let file = dir.join("big.txt");
        let content = "x".repeat(1024);
        fs::write(&file, &content).unwrap();

        store.snapshot_file(&file).unwrap();
        assert_eq!(store.memory_usage(), 1024);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_count() {
        let dir = temp_dir();
        let mut store = SnapshotStore::new();

        assert_eq!(store.count(), 0);

        for i in 0..3 {
            let file = dir.join(format!("f{i}.txt"));
            fs::write(&file, "x").unwrap();
            store.snapshot_file(&file).unwrap();
        }

        assert_eq!(store.count(), 3);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_get_snapshot_content() {
        let dir = temp_dir();
        let file = dir.join("peek.txt");
        fs::write(&file, "peek content").unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        let content = store.get_snapshot_content(&file);
        assert!(content.is_some());
        assert_eq!(content.unwrap(), b"peek content");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_binary_file() {
        let dir = temp_dir();
        let file = dir.join("binary.dat");
        let data: Vec<u8> = (0..=255).collect();
        fs::write(&file, &data).unwrap();

        let mut store = SnapshotStore::new();
        store.snapshot_file(&file).unwrap();

        // Overwrite with different data.
        fs::write(&file, [0u8; 10]).unwrap();

        // Restore.
        store.restore_file(&file).unwrap();
        assert_eq!(fs::read(&file).unwrap(), data);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_paths() {
        let dir = temp_dir();
        let mut store = SnapshotStore::new();

        let file_a = dir.join("a.txt");
        let file_b = dir.join("b.txt");
        fs::write(&file_a, "a").unwrap();
        fs::write(&file_b, "b").unwrap();

        store.snapshot_file(&file_a).unwrap();
        store.snapshot_file(&file_b).unwrap();

        let paths = store.snapshot_paths();
        assert_eq!(paths.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }
}
