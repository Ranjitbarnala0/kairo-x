//! Priority queue for pending nodes in the execution graph.
//!
//! Orders nodes by priority (Critical > Standard > Mechanical),
//! with tiebreaking by node ID (lower ID first = FIFO within priority).

use crate::arena::node::Priority;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
#[derive(Debug, Clone)]
pub struct PriorityQueue {
    heap: BinaryHeap<PendingEntry>,
}

impl PriorityQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Push a node onto the queue.
    pub fn push(&mut self, entry: PendingEntry) {
        self.heap.push(entry);
    }

    /// Pop the highest-priority node from the queue.
    pub fn pop(&mut self) -> Option<PendingEntry> {
        self.heap.pop()
    }

    /// Peek at the highest-priority node without removing it.
    pub fn peek(&self) -> Option<&PendingEntry> {
        self.heap.peek()
    }

    /// Number of entries in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Remove all entries for a specific node (e.g., if it got claimed by a track).
    ///
    /// This is O(n) but happens rarely — only when a node is claimed.
    pub fn remove_node(&mut self, node_idx: u32) {
        let entries: Vec<PendingEntry> = self.heap.drain().collect();
        self.heap = entries
            .into_iter()
            .filter(|e| e.node_idx != node_idx)
            .collect();
    }

    /// Drain all entries. Used during re-planning.
    pub fn drain(&mut self) -> Vec<PendingEntry> {
        self.heap.drain().collect()
    }

    /// Rebuild the queue from a list of entries. Used during checkpoint restore.
    pub fn rebuild(&mut self, entries: Vec<PendingEntry>) {
        self.heap.clear();
        for entry in entries {
            self.heap.push(entry);
        }
    }

    /// Count entries by priority.
    pub fn count_by_priority(&self) -> (usize, usize, usize) {
        let mut critical = 0;
        let mut standard = 0;
        let mut mechanical = 0;

        for entry in self.heap.iter() {
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
