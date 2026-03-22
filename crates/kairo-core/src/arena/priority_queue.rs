//! Priority queue for pending nodes in the execution graph.
//!
//! Orders nodes by priority (Critical > Standard > Mechanical),
//! with tiebreaking by node ID (lower ID first = FIFO within priority).

use crate::arena::node::Priority;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// ---------------------------------------------------------------------------
// Pending entry — wrapper for the priority queue
// ---------------------------------------------------------------------------

/// An entry in the pending queue representing a node ready for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingEntry {
    /// Index of the node in the arena.
    pub node_idx: u32,
    /// Priority of the node (determines execution order).
    pub priority: Priority,
}

/// Ordering: higher priority first, then lower node_idx first (FIFO tiebreak).
///
/// BinaryHeap is a max-heap, so we implement Ord to give higher Priority
/// and lower node_idx greater ordering.
impl Eq for PendingEntry {}

impl PartialEq for PendingEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.node_idx == other.node_idx
    }
}

impl PartialOrd for PendingEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // First: higher priority wins
        self.priority
            .cmp(&other.priority)
            // Then: lower node_idx wins (FIFO among same priority)
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

// ---------------------------------------------------------------------------
// Priority queue
// ---------------------------------------------------------------------------

/// Priority queue for scheduling pending nodes.
///
/// Nodes are dequeued in order: Critical first, then Standard, then Mechanical.
/// Within the same priority, earlier-created nodes (lower ID) go first.
///
/// Uses lazy deletion: `remove_node` marks an entry as removed in O(1), and
/// `pop` skips lazily-removed entries. This avoids the O(n) drain+rebuild that
/// the previous eager approach required.
#[derive(Debug, Clone)]
pub struct PriorityQueue {
    heap: BinaryHeap<PendingEntry>,
    /// Node indices that have been lazily removed. `pop` skips these.
    removed: HashSet<u32>,
}

impl PriorityQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            removed: HashSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            removed: HashSet::new(),
        }
    }

    /// Push a node onto the queue.
    ///
    /// If the node was previously lazily removed, the removal mark is cleared
    /// so this new entry will be returned by `pop`.
    pub fn push(&mut self, entry: PendingEntry) {
        self.removed.remove(&entry.node_idx);
        self.heap.push(entry);
    }

    /// Pop the highest-priority node from the queue, skipping lazily removed entries.
    pub fn pop(&mut self) -> Option<PendingEntry> {
        while let Some(entry) = self.heap.pop() {
            if self.removed.remove(&entry.node_idx) {
                // This entry was lazily removed -- skip it.
                continue;
            }
            return Some(entry);
        }
        None
    }

    /// Peek at the highest-priority live node without removing it.
    ///
    /// Note: this may return a lazily-removed entry if the top of the heap
    /// has not been cleaned yet. For authoritative checks, use `pop`.
    pub fn peek(&self) -> Option<&PendingEntry> {
        // Return the first non-removed entry at the top.
        // Since we can't mutate in peek, we check the top only.
        self.heap.peek().filter(|e| !self.removed.contains(&e.node_idx))
    }

    /// Number of entries in the queue (includes lazily removed entries).
    /// For an exact count, subtract `removed.len()`.
    pub fn len(&self) -> usize {
        self.heap.len().saturating_sub(self.removed.len())
    }

    /// Whether the queue has no live entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lazily remove all entries for a specific node. O(1).
    ///
    /// The entry remains in the heap but will be skipped by `pop`.
    pub fn remove_node(&mut self, node_idx: u32) {
        self.removed.insert(node_idx);
    }

    /// Drain all live entries. Used during re-planning.
    pub fn drain(&mut self) -> Vec<PendingEntry> {
        let removed = &self.removed;
        let entries: Vec<PendingEntry> = self
            .heap
            .drain()
            .filter(|e| !removed.contains(&e.node_idx))
            .collect();
        self.removed.clear();
        entries
    }

    /// Rebuild the queue from a list of entries. Used during checkpoint restore.
    pub fn rebuild(&mut self, entries: Vec<PendingEntry>) {
        self.heap.clear();
        self.removed.clear();
        for entry in entries {
            self.heap.push(entry);
        }
    }

    /// Count live entries by priority.
    pub fn count_by_priority(&self) -> (usize, usize, usize) {
        let mut critical = 0;
        let mut standard = 0;
        let mut mechanical = 0;

        for entry in self.heap.iter() {
            if self.removed.contains(&entry.node_idx) {
                continue;
            }
            match entry.priority {
                Priority::Critical => critical += 1,
                Priority::Standard => standard += 1,
                Priority::Mechanical => mechanical += 1,
            }
        }

        (critical, standard, mechanical)
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let mut pq = PriorityQueue::new();

        pq.push(PendingEntry {
            node_idx: 3,
            priority: Priority::Mechanical,
        });
        pq.push(PendingEntry {
            node_idx: 1,
            priority: Priority::Critical,
        });
        pq.push(PendingEntry {
            node_idx: 2,
            priority: Priority::Standard,
        });

        // Should come out: Critical, Standard, Mechanical
        assert_eq!(pq.pop().unwrap().node_idx, 1);
        assert_eq!(pq.pop().unwrap().node_idx, 2);
        assert_eq!(pq.pop().unwrap().node_idx, 3);
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut pq = PriorityQueue::new();

        pq.push(PendingEntry {
            node_idx: 5,
            priority: Priority::Standard,
        });
        pq.push(PendingEntry {
            node_idx: 2,
            priority: Priority::Standard,
        });
        pq.push(PendingEntry {
            node_idx: 8,
            priority: Priority::Standard,
        });

        // Lower node_idx first (FIFO by creation order)
        assert_eq!(pq.pop().unwrap().node_idx, 2);
        assert_eq!(pq.pop().unwrap().node_idx, 5);
        assert_eq!(pq.pop().unwrap().node_idx, 8);
    }

    #[test]
    fn test_remove_node() {
        let mut pq = PriorityQueue::new();

        pq.push(PendingEntry {
            node_idx: 1,
            priority: Priority::Critical,
        });
        pq.push(PendingEntry {
            node_idx: 2,
            priority: Priority::Standard,
        });
        pq.push(PendingEntry {
            node_idx: 3,
            priority: Priority::Mechanical,
        });

        pq.remove_node(2);
        assert_eq!(pq.len(), 2);
        assert_eq!(pq.pop().unwrap().node_idx, 1);
        assert_eq!(pq.pop().unwrap().node_idx, 3);
    }

    #[test]
    fn test_count_by_priority() {
        let mut pq = PriorityQueue::new();

        pq.push(PendingEntry { node_idx: 1, priority: Priority::Critical });
        pq.push(PendingEntry { node_idx: 2, priority: Priority::Critical });
        pq.push(PendingEntry { node_idx: 3, priority: Priority::Standard });
        pq.push(PendingEntry { node_idx: 4, priority: Priority::Standard });
        pq.push(PendingEntry { node_idx: 5, priority: Priority::Standard });
        pq.push(PendingEntry { node_idx: 6, priority: Priority::Mechanical });

        let (c, s, m) = pq.count_by_priority();
        assert_eq!(c, 2);
        assert_eq!(s, 3);
        assert_eq!(m, 1);
    }
}
