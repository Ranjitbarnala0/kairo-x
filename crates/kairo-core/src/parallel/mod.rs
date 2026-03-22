//! Parallel execution scheduler (§11).
//!
//! Manages multiple execution tracks that can work on independent graph nodes
//! simultaneously. Each track has its own LLM session. File-level write locks
//! (from `tools::file_locks`) prevent conflicts when tracks touch the same files.
//!
//! Verification (deterministic checks) is serialized through a VerificationQueue
//! to avoid interference (e.g., two tracks running `cargo test` simultaneously).
//! LLM audit calls run in parallel since they're just API calls.

use crate::arena::Arena;
use crate::arena::node::NodeStatus;
use crate::tools::file_locks::FileLockTable;
use std::collections::VecDeque;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Track state
// ---------------------------------------------------------------------------

/// State of a single parallel execution track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackState {
    /// No node assigned — track is available for work.
    Idle,
    /// Track is implementing a node (waiting for LLM response).
    Implementing,
    /// Track is running deterministic verification or LLM audit.
    Verifying,
    /// Track is blocked waiting for a file lock held by another track.
    Blocked(PathBuf),
    /// Track is running a fix cycle (LLM fix after verification failure).
    Fixing,
}

/// A single parallel execution track.
#[derive(Debug)]
pub struct Track {
    /// Track identifier (0-based).
    pub id: u8,
    /// Currently assigned node index (0 = none).
    pub node_id: u32,
    /// Current session ID bound to this track (0 = none).
    pub session_id: u32,
    /// Current state.
    pub state: TrackState,
    /// Number of LLM calls made in current assignment.
    pub calls_in_assignment: u16,
}

impl Track {
    pub fn new(id: u8) -> Self {
        Self {
            id,
            node_id: 0,
            session_id: 0,
            state: TrackState::Idle,
            calls_in_assignment: 0,
        }
    }

    /// Whether this track is available for a new node assignment.
    pub fn is_idle(&self) -> bool {
        self.state == TrackState::Idle
    }

    /// Reset the track to idle state.
    pub fn reset(&mut self) {
        self.node_id = 0;
        self.session_id = 0;
        self.state = TrackState::Idle;
        self.calls_in_assignment = 0;
    }

    /// Assign a node to this track.
    pub fn assign(&mut self, node_id: u32, session_id: u32) {
        self.node_id = node_id;
        self.session_id = session_id;
        self.state = TrackState::Implementing;
        self.calls_in_assignment = 0;
    }
}

// ---------------------------------------------------------------------------
// Verification queue — serializes deterministic checks across tracks
// ---------------------------------------------------------------------------

/// Serializes deterministic verification (build, test, etc.) across tracks
/// so that only one track runs these at a time, avoiding interference.
#[derive(Debug)]
pub struct VerificationQueue {
    /// Queue of (track_id, node_id) waiting for deterministic verification.
    queue: VecDeque<(u8, u32)>,
    /// Which track is currently running deterministic checks (None = queue idle).
    active: Option<u8>,
}

impl VerificationQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            active: None,
        }
    }

    /// Enqueue a track for deterministic verification.
    pub fn enqueue(&mut self, track_id: u8, node_id: u32) {
        self.queue.push_back((track_id, node_id));
    }

    /// Try to start the next verification. Returns `Some((track_id, node_id))`
    /// if a track can start verifying now, `None` if the queue is empty or
    /// another track is already verifying.
    pub fn try_start_next(&mut self) -> Option<(u8, u32)> {
        if self.active.is_some() {
            return None; // Another track is verifying
        }
        if let Some((track_id, node_id)) = self.queue.pop_front() {
            self.active = Some(track_id);
            Some((track_id, node_id))
        } else {
            None
        }
    }

    /// Mark the current verification as done. Allows the next queued track to start.
    pub fn finish_current(&mut self) {
        self.active = None;
    }

    /// Whether a specific track is currently the active verifier.
    pub fn is_active_verifier(&self, track_id: u8) -> bool {
        self.active == Some(track_id)
    }

    /// Number of tracks waiting in the queue (not including the active one).
    pub fn waiting_count(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is idle (no active verification and no waiters).
    pub fn is_idle(&self) -> bool {
        self.active.is_none() && self.queue.is_empty()
    }
}

impl Default for VerificationQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Parallel scheduler
// ---------------------------------------------------------------------------

/// Parallel execution scheduler that manages multiple tracks.
///
/// Each track works on an independent graph node. The scheduler assigns
/// nodes from the priority queue to idle tracks, handles file lock
/// contention, and serializes deterministic verification.
#[derive(Debug)]
pub struct ParallelScheduler {
    /// Maximum number of concurrent tracks.
    pub max_tracks: u8,
    /// The execution tracks.
    pub tracks: Vec<Track>,
    /// Shared file lock table (reference held externally, passed to methods).
    /// Serialization queue for deterministic verification.
    pub verification_queue: VerificationQueue,
}

impl ParallelScheduler {
    /// Create a new scheduler with the specified number of tracks.
    pub fn new(max_tracks: u8) -> Self {
        let tracks = (0..max_tracks).map(Track::new).collect();
        Self {
            max_tracks,
            tracks,
            verification_queue: VerificationQueue::new(),
        }
    }

    /// Get the number of idle tracks available for assignment.
    pub fn idle_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_idle()).count()
    }

    /// Get the number of active (non-idle) tracks.
    pub fn active_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| !t.is_idle()).count()
    }

    /// Find the first idle track and return its ID.
    pub fn find_idle_track(&self) -> Option<u8> {
        self.tracks.iter().find(|t| t.is_idle()).map(|t| t.id)
    }

    /// Assign a node from the pending queue to the next idle track.
    ///
    /// Returns `Some((track_id, node_idx))` if assignment was made, `None` if
    /// no idle tracks or no pending nodes.
    ///
    /// Uses a bounded loop (max 10 000 iterations) to skip stale or conflicting
    /// entries without recursion.  A `tracing::error!` fires if the bound is
    /// ever reached — that indicates a bug in queue maintenance.
    pub fn assign_next_pending(
        &mut self,
        arena: &mut Arena,
        _file_locks: &FileLockTable,
    ) -> Option<(u8, u32)> {
        const MAX_ITERATIONS: u32 = 10_000;

        let track_id = self.find_idle_track()?;

        let mut iterations: u32 = 0;
        loop {
            if iterations >= MAX_ITERATIONS {
                tracing::error!(
                    "assign_next_pending: hit {} iteration limit — \
                     pending_queue may contain only stale/conflicting entries",
                    MAX_ITERATIONS,
                );
                return None;
            }
            iterations += 1;

            // Pop the highest-priority pending node; empty queue → nothing to assign.
            let entry = arena.pending_queue.pop()?;
            let node_idx = entry.node_idx;

            // Bounds-checked access: if the index is out of range (shouldn't
            // happen, but defend against arena corruption) treat it as stale.
            let node = match arena.nodes.get(node_idx as usize) {
                Some(n) => n,
                None => {
                    tracing::error!(
                        node_idx,
                        "assign_next_pending: node_idx out of arena bounds, dropping entry"
                    );
                    continue; // stale / corrupt entry, skip
                }
            };

            // Verify the node is still in a state that needs execution.
            if node.status != NodeStatus::Pending && node.status != NodeStatus::Ready {
                continue; // already claimed or completed — skip
            }

            // Check if this node's files conflict with any active track's files.
            let has_conflict = self.tracks.iter().any(|track| {
                !track.is_idle() && arena.nodes_share_files(node_idx, track.node_id)
            });

            if has_conflict {
                // Conflict — put the node back and try the next one.
                arena.pending_queue.push(entry);
                continue;
            }

            // All checks passed — assign the node to the track.
            arena.get_mut(node_idx).expect("node validated above").status = NodeStatus::Implementing;
            self.tracks[track_id as usize].assign(node_idx, 0); // session_id set later

            return Some((track_id, node_idx));
        }
    }

    /// Release a track (node completed, failed, or being reassigned).
    ///
    /// Releases all file locks held by the track.
    pub fn release_track(&mut self, track_id: u8, file_locks: &FileLockTable) {
        file_locks.release_all(track_id);
        self.tracks[track_id as usize].reset();
    }

    /// Get a reference to a specific track.
    pub fn track(&self, track_id: u8) -> &Track {
        &self.tracks[track_id as usize]
    }

    /// Get a mutable reference to a specific track.
    pub fn track_mut(&mut self, track_id: u8) -> &mut Track {
        &mut self.tracks[track_id as usize]
    }

    /// Move a track to verification state and enqueue for deterministic checks.
    pub fn enqueue_verification(&mut self, track_id: u8) {
        let node_id = self.tracks[track_id as usize].node_id;
        self.tracks[track_id as usize].state = TrackState::Verifying;
        self.verification_queue.enqueue(track_id, node_id);
    }

    /// Whether parallelization should be used at all.
    ///
    /// Returns false if conditions from §11.4 are met (single track
    /// configured, only 1 pending node, etc.)
    pub fn should_parallelize(&self, pending_count: usize) -> bool {
        if self.max_tracks <= 1 {
            return false;
        }
        if pending_count <= 1 && self.active_track_count() == 0 {
            return false;
        }
        true
    }

    /// Summary of all track states for display.
    pub fn status_summary(&self) -> Vec<TrackSummary> {
        self.tracks
            .iter()
            .map(|t| TrackSummary {
                id: t.id,
                node_id: t.node_id,
                state: format!("{:?}", t.state),
                calls: t.calls_in_assignment,
            })
            .collect()
    }
}

/// Summary of a single track for display purposes.
#[derive(Debug, Clone)]
pub struct TrackSummary {
    pub id: u8,
    pub node_id: u32,
    pub state: String,
    pub calls: u16,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::{Node, Priority};
    use crate::arena::priority_queue::PendingEntry;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = ParallelScheduler::new(3);
        assert_eq!(scheduler.tracks.len(), 3);
        assert_eq!(scheduler.idle_track_count(), 3);
        assert_eq!(scheduler.active_track_count(), 0);
    }

    #[test]
    fn test_track_lifecycle() {
        let mut track = Track::new(0);
        assert!(track.is_idle());

        track.assign(42, 1);
        assert!(!track.is_idle());
        assert_eq!(track.node_id, 42);
        assert_eq!(track.state, TrackState::Implementing);

        track.reset();
        assert!(track.is_idle());
        assert_eq!(track.node_id, 0);
    }

    #[test]
    fn test_verification_queue() {
        let mut vq = VerificationQueue::new();
        assert!(vq.is_idle());

        vq.enqueue(0, 10);
        vq.enqueue(1, 20);

        let first = vq.try_start_next();
        assert_eq!(first, Some((0, 10)));
        assert!(vq.is_active_verifier(0));

        // Can't start another while one is active
        assert!(vq.try_start_next().is_none());

        vq.finish_current();
        let second = vq.try_start_next();
        assert_eq!(second, Some((1, 20)));
    }

    #[test]
    fn test_assign_next_pending() {
        let mut arena = Arena::new();
        let n1 = arena.alloc(Node::new("JWT Service".to_string(), Priority::Critical));
        let n2 = arena.alloc(Node::new("Auth Middleware".to_string(), Priority::Standard));

        // Add to pending queue (simulating what graph construction does)
        arena.pending_queue.push(PendingEntry {
            node_idx: n1,
            priority: Priority::Critical,
        });
        arena.pending_queue.push(PendingEntry {
            node_idx: n2,
            priority: Priority::Standard,
        });

        let file_locks = FileLockTable::new();
        let mut scheduler = ParallelScheduler::new(3);

        // First assignment should get the Critical node
        let result = scheduler.assign_next_pending(&mut arena, &file_locks);
        assert!(result.is_some());
        let (track_id, node_idx) = result.unwrap();
        assert_eq!(node_idx, n1); // Critical first
        assert_eq!(track_id, 0);
        assert_eq!(scheduler.active_track_count(), 1);
    }

    #[test]
    fn test_should_parallelize() {
        let scheduler = ParallelScheduler::new(1);
        assert!(!scheduler.should_parallelize(5));

        let scheduler = ParallelScheduler::new(3);
        assert!(scheduler.should_parallelize(3));
        assert!(!scheduler.should_parallelize(1)); // only 1 pending, no active
    }
}
