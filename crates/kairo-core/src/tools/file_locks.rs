//! Per-file exclusive lock system for parallel track execution (§7).
//!
//! When multiple tracks execute in parallel, they may attempt to modify the
//! same file simultaneously. The `FileLockTable` ensures mutual exclusion:
//! only one track can hold a write lock on a given file path at a time.
//!
//! Uses `DashMap` for lock-free concurrent access to the lock table itself,
//! while providing exclusive per-file semantics to tracks.

use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Track identifier — u8 supports up to 256 parallel tracks.
pub type TrackId = u8;

/// Result of attempting to acquire a file lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockResult {
    /// Lock successfully acquired for this track.
    Acquired,
    /// This track already holds the lock — no-op success.
    AlreadyOwned,
    /// Another track holds the lock.
    Blocked(TrackId),
}

/// Result of a write-with-lock operation.
#[derive(Debug)]
pub enum WriteResult<T> {
    /// Write completed successfully.
    Ok(T),
    /// The file was locked by another track that may have modified it.
    /// The caller should re-read the file before retrying.
    ReReadNeeded {
        blocking_track: TrackId,
    },
}

/// Errors from the file lock system.
#[derive(Debug, Error)]
pub enum FileLockError {
    #[error("lock acquisition timed out after {elapsed_ms}ms, blocked by track {blocking_track}")]
    Timeout {
        elapsed_ms: u64,
        blocking_track: TrackId,
    },
    #[error("I/O error during locked write: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Path canonicalization
// ---------------------------------------------------------------------------

/// Canonicalize a path for use as a DashMap key, ensuring that different
/// string representations of the same filesystem location resolve to the
/// same key. For paths that do not yet exist on disk, canonicalizes the
/// parent directory and appends the file name.
fn canonical_key(path: &Path) -> PathBuf {
    match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // Non-existent file: canonicalize parent + append name
            if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
                parent
                    .canonicalize()
                    .map(|p| p.join(name))
                    .unwrap_or_else(|_| path.to_path_buf())
            } else {
                path.to_path_buf()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FileLockTable
// ---------------------------------------------------------------------------

/// Concurrent per-file exclusive lock table.
///
/// Each entry maps a canonical file path to the track that currently holds
/// the lock. DashMap provides the outer concurrency safety; the semantics
/// are strictly exclusive (one track per file).
#[derive(Debug, Clone)]
pub struct FileLockTable {
    locks: Arc<DashMap<PathBuf, TrackId>>,
}

impl FileLockTable {
    /// Create a new empty lock table.
    pub fn new() -> Self {
        Self {
            locks: Arc::new(DashMap::new()),
        }
    }

    /// Attempt to acquire the lock on `path` for `track`.
    ///
    /// Returns:
    /// - `Acquired` if the lock was free and is now held by `track`.
    /// - `AlreadyOwned` if `track` already holds this lock.
    /// - `Blocked(other)` if another track holds the lock.
    pub fn acquire(&self, path: &Path, track: TrackId) -> LockResult {
        let key = canonical_key(path);
        // Try to insert atomically. DashMap::entry provides the CAS semantics.
        match self.locks.entry(key) {
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(track);
                debug!(path = %path.display(), track, "file lock acquired");
                LockResult::Acquired
            }
            dashmap::mapref::entry::Entry::Occupied(occupied) => {
                let holder = *occupied.get();
                if holder == track {
                    LockResult::AlreadyOwned
                } else {
                    debug!(
                        path = %path.display(),
                        track,
                        holder,
                        "file lock blocked"
                    );
                    LockResult::Blocked(holder)
                }
            }
        }
    }

    /// Release the lock on `path` if held by `track`.
    ///
    /// No-op if the lock is not held or held by a different track.
    pub fn release(&self, path: &Path, track: TrackId) {
        let key = canonical_key(path);
        self.locks.remove_if(&key, |_, holder| {
            if *holder == track {
                debug!(path = %path.display(), track, "file lock released");
                true
            } else {
                false
            }
        });
    }

    /// Release all locks held by `track`.
    ///
    /// Uses collect-then-remove instead of `retain` to avoid potential
    /// deadlocks from holding DashMap shard locks during iteration.
    pub fn release_all(&self, track: TrackId) {
        let keys_to_remove: Vec<PathBuf> = self
            .locks
            .iter()
            .filter(|entry| *entry.value() == track)
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys_to_remove {
            debug!(path = %key.display(), track, "file lock released (release_all)");
            self.locks.remove_if(&key, |_, v| *v == track);
        }
    }

    /// Query which track (if any) holds the lock on `path`.
    pub fn held_by(&self, path: &Path) -> Option<TrackId> {
        let key = canonical_key(path);
        self.locks.get(&key).map(|entry| *entry.value())
    }

    /// List all file paths currently locked by `track`.
    pub fn locks_held_by(&self, track: TrackId) -> Vec<PathBuf> {
        self.locks
            .iter()
            .filter(|entry| *entry.value() == track)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Total number of active locks.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }

    /// Clear all locks. Use only during shutdown or reset.
    pub fn clear(&self) {
        self.locks.clear();
    }
}

impl Default for FileLockTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RAII lock guard
// ---------------------------------------------------------------------------

/// RAII guard that automatically releases a file lock when dropped.
///
/// Obtained via [`FileLockTable::acquire_guard`]. The lock is released
/// in the `Drop` implementation, so callers cannot accidentally forget
/// to release it — even in the presence of early returns or panics.
pub struct FileLockGuard<'a> {
    table: &'a FileLockTable,
    path: PathBuf,
    track: TrackId,
}

impl FileLockGuard<'_> {
    /// The canonical path this guard protects.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The track that owns this lock.
    pub fn track(&self) -> TrackId {
        self.track
    }
}

impl Drop for FileLockGuard<'_> {
    fn drop(&mut self) {
        self.table.release(&self.path, self.track);
    }
}

impl FileLockTable {
    /// Acquire a file lock and return an RAII guard that releases it on drop.
    ///
    /// Returns `Ok(FileLockGuard)` if the lock was acquired (or already owned),
    /// or `Err(LockResult::Blocked(other))` if another track holds the lock.
    pub fn acquire_guard(&self, path: &Path, track: TrackId) -> Result<FileLockGuard<'_>, LockResult> {
        match self.acquire(path, track) {
            LockResult::Acquired | LockResult::AlreadyOwned => Ok(FileLockGuard {
                table: self,
                path: canonical_key(path),
                track,
            }),
            other => Err(other),
        }
    }
}

// ---------------------------------------------------------------------------
// write_with_lock — async lock-acquire-and-write with retry
// ---------------------------------------------------------------------------

/// Retry interval when blocked by another track.
const RETRY_INTERVAL: Duration = Duration::from_millis(500);

/// Maximum time to wait before giving up and returning `ReReadNeeded`.
const MAX_WAIT: Duration = Duration::from_secs(5);

/// Acquire a file lock and execute a write operation.
///
/// If the lock is held by another track, retries every 500ms for up to
/// 5 seconds. If still blocked after 5 seconds, returns `ReReadNeeded`
/// — the caller must re-read the file (the other track may have changed it)
/// before retrying the edit.
///
/// # Arguments
///
/// * `table` — the shared lock table.
/// * `path` — the file path to lock.
/// * `track` — the track requesting the lock.
/// * `write_fn` — the async closure that performs the write while the lock is held.
///
/// The lock is released after `write_fn` completes (or on error).
pub async fn write_with_lock<F, Fut, T>(
    table: &FileLockTable,
    path: &Path,
    track: TrackId,
    write_fn: F,
) -> Result<WriteResult<T>, FileLockError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, std::io::Error>>,
{
    let start = tokio::time::Instant::now();

    loop {
        match table.acquire(path, track) {
            LockResult::Acquired | LockResult::AlreadyOwned => {
                // We hold the lock — execute the write.
                let result = write_fn().await;
                // Release the lock regardless of write outcome.
                table.release(path, track);
                match result {
                    Ok(value) => return Ok(WriteResult::Ok(value)),
                    Err(io_err) => return Err(FileLockError::Io(io_err)),
                }
            }
            LockResult::Blocked(blocker) => {
                let elapsed = start.elapsed();
                if elapsed >= MAX_WAIT {
                    warn!(
                        path = %path.display(),
                        track,
                        blocker,
                        elapsed_ms = elapsed.as_millis() as u64,
                        "lock wait timeout, returning ReReadNeeded"
                    );
                    return Ok(WriteResult::ReReadNeeded {
                        blocking_track: blocker,
                    });
                }
                // Wait and retry.
                let remaining = MAX_WAIT.saturating_sub(elapsed);
                let sleep_time = RETRY_INTERVAL.min(remaining);
                tokio::time::sleep(sleep_time).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_and_release() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/test.rs");

        assert_eq!(table.acquire(&path, 0), LockResult::Acquired);
        assert_eq!(table.held_by(&path), Some(0));

        table.release(&path, 0);
        assert_eq!(table.held_by(&path), None);
    }

    #[test]
    fn test_already_owned() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/test.rs");

        assert_eq!(table.acquire(&path, 1), LockResult::Acquired);
        assert_eq!(table.acquire(&path, 1), LockResult::AlreadyOwned);

        table.release(&path, 1);
    }

    #[test]
    fn test_blocked() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/test.rs");

        assert_eq!(table.acquire(&path, 0), LockResult::Acquired);
        assert_eq!(table.acquire(&path, 1), LockResult::Blocked(0));
        assert_eq!(table.acquire(&path, 2), LockResult::Blocked(0));

        table.release(&path, 0);
        assert_eq!(table.acquire(&path, 1), LockResult::Acquired);
    }

    #[test]
    fn test_release_wrong_track() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/test.rs");

        table.acquire(&path, 0);
        table.release(&path, 1); // Wrong track — should be no-op.
        assert_eq!(table.held_by(&path), Some(0));

        table.release(&path, 0);
        assert_eq!(table.held_by(&path), None);
    }

    #[test]
    fn test_release_all() {
        let table = FileLockTable::new();

        table.acquire(Path::new("/a.rs"), 0);
        table.acquire(Path::new("/b.rs"), 0);
        table.acquire(Path::new("/c.rs"), 1);

        table.release_all(0);

        assert_eq!(table.held_by(Path::new("/a.rs")), None);
        assert_eq!(table.held_by(Path::new("/b.rs")), None);
        assert_eq!(table.held_by(Path::new("/c.rs")), Some(1));
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_locks_held_by() {
        let table = FileLockTable::new();

        table.acquire(Path::new("/a.rs"), 0);
        table.acquire(Path::new("/b.rs"), 0);
        table.acquire(Path::new("/c.rs"), 1);

        let mut held = table.locks_held_by(0);
        held.sort();
        assert_eq!(held, vec![PathBuf::from("/a.rs"), PathBuf::from("/b.rs")]);

        let held_1 = table.locks_held_by(1);
        assert_eq!(held_1, vec![PathBuf::from("/c.rs")]);
    }

    #[test]
    fn test_multiple_files_same_track() {
        let table = FileLockTable::new();

        assert_eq!(table.acquire(Path::new("/a.rs"), 0), LockResult::Acquired);
        assert_eq!(table.acquire(Path::new("/b.rs"), 0), LockResult::Acquired);
        assert_eq!(table.acquire(Path::new("/c.rs"), 0), LockResult::Acquired);

        assert_eq!(table.lock_count(), 3);

        table.release_all(0);
        assert_eq!(table.lock_count(), 0);
    }

    #[test]
    fn test_clear() {
        let table = FileLockTable::new();

        table.acquire(Path::new("/a.rs"), 0);
        table.acquire(Path::new("/b.rs"), 1);
        assert_eq!(table.lock_count(), 2);

        table.clear();
        assert_eq!(table.lock_count(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        // Verify DashMap-based table is safe for concurrent access.
        let table = Arc::new(FileLockTable::new());
        let mut handles = vec![];

        for track in 0..8u8 {
            let table = Arc::clone(&table);
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let path = PathBuf::from(format!("/file_{track}_{i}.rs"));
                    let result = table.acquire(&path, track);
                    assert_eq!(result, LockResult::Acquired);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(table.lock_count(), 800);
    }

    #[test]
    fn test_contention() {
        // Multiple tracks trying to lock the same file.
        let table = FileLockTable::new();
        let path = PathBuf::from("/shared.rs");

        // Track 0 gets it first.
        assert_eq!(table.acquire(&path, 0), LockResult::Acquired);

        // Tracks 1-7 are all blocked by track 0.
        for track in 1..8u8 {
            assert_eq!(table.acquire(&path, track), LockResult::Blocked(0));
        }

        // Release and track 1 can now acquire.
        table.release(&path, 0);
        assert_eq!(table.acquire(&path, 1), LockResult::Acquired);

        // Tracks 2-7 now blocked by track 1.
        for track in 2..8u8 {
            assert_eq!(table.acquire(&path, track), LockResult::Blocked(1));
        }
    }

    #[tokio::test]
    async fn test_write_with_lock_success() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/write_test.rs");

        let result = write_with_lock(&table, &path, 0, || async {
            Ok::<_, std::io::Error>(42)
        })
        .await
        .unwrap();

        match result {
            WriteResult::Ok(val) => assert_eq!(val, 42),
            _ => panic!("expected WriteResult::Ok"),
        }

        // Lock should be released after write_fn completes.
        assert_eq!(table.held_by(&path), None);
    }

    #[tokio::test]
    async fn test_write_with_lock_io_error() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/write_test_err.rs");

        let result = write_with_lock(&table, &path, 0, || async {
            Err::<(), _>(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "test error",
            ))
        })
        .await;

        assert!(result.is_err());
        // Lock should still be released on error.
        assert_eq!(table.held_by(&path), None);
    }

    #[tokio::test]
    async fn test_write_with_lock_blocked_then_released() {
        let table = Arc::new(FileLockTable::new());
        let path = PathBuf::from("/tmp/contended.rs");

        // Track 0 holds the lock.
        table.acquire(&path, 0);

        let table_clone = Arc::clone(&table);
        let path_clone = path.clone();

        // Spawn a task that releases track 0's lock after 600ms.
        let release_handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(600)).await;
            table_clone.release(&path_clone, 0);
        });

        // Track 1 tries to write — should block, then succeed after release.
        let result = write_with_lock(&table, &path, 1, || async {
            Ok::<_, std::io::Error>("written")
        })
        .await
        .unwrap();

        release_handle.await.unwrap();

        match result {
            WriteResult::Ok(val) => assert_eq!(val, "written"),
            _ => panic!("expected WriteResult::Ok after retry"),
        }
    }

    #[tokio::test]
    async fn test_write_with_lock_timeout_reread_needed() {
        let table = FileLockTable::new();
        let path = PathBuf::from("/tmp/stuck.rs");

        // Track 0 holds the lock and never releases it.
        table.acquire(&path, 0);

        let start = tokio::time::Instant::now();
        let result = write_with_lock(&table, &path, 1, || async {
            Ok::<_, std::io::Error>(())
        })
        .await
        .unwrap();

        let elapsed = start.elapsed();

        match result {
            WriteResult::ReReadNeeded { blocking_track } => {
                assert_eq!(blocking_track, 0);
                // Should have waited approximately 5 seconds.
                assert!(elapsed >= Duration::from_secs(4));
            }
            _ => panic!("expected WriteResult::ReReadNeeded"),
        }
    }
}
