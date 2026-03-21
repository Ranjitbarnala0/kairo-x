pub mod node;
pub mod query;
pub mod priority_queue;
pub mod serialize;

use crate::itch::ItchRegister;
use fnv::FnvHashMap;
use node::{Node, NodeStatus};
use priority_queue::{PendingEntry, PriorityQueue};
use smallvec::SmallVec;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Arena — the core execution graph data structure
// ---------------------------------------------------------------------------

/// Arena-allocated DAG for the execution graph.
///
/// All nodes live in a contiguous `Vec<Node>`. Edges are inline `SmallVec`s
/// per node (no separate edge array). Free list enables O(1) node recycling.
///
/// Indices provide fast lookup:
/// - `title_index`: FNV hash of title → node index
/// - `path_index`: FNV hash of file path → node indices touching that file
#[derive(Debug)]
pub struct Arena {
    /// All nodes. Index 0 is reserved (sentinel/root).
    pub(crate) nodes: Vec<Node>,
    /// Free list of deallocated node indices for reuse.
    pub(crate) free_list: Vec<u32>,
    /// FNV hash of title → node index for fast title lookup.
    pub(crate) title_index: FnvHashMap<u64, u32>,
    /// FNV hash of file path → node indices that touch that file.
    pub(crate) path_index: FnvHashMap<u64, Vec<u32>>,
    /// Priority queue for pending nodes with resolved dependencies.
    pub(crate) pending_queue: PriorityQueue,
    /// Dynamic itch register — one bit per node.
    pub(crate) itch: ItchRegister,
    /// Side table: node index → full specification text.
    pub(crate) specs: HashMap<u32, String>,
    /// Side table: file hash → file path string.
    pub(crate) file_paths: HashMap<u64, String>,
    /// Monotonic step counter for timing.
    pub(crate) current_step: u32,
}

impl Arena {
    /// Create a new empty arena.
    pub fn new() -> Self {
        let mut arena = Self {
            nodes: Vec::with_capacity(256),
            free_list: Vec::new(),
            title_index: FnvHashMap::default(),
            path_index: FnvHashMap::default(),
            pending_queue: PriorityQueue::new(),
            itch: ItchRegister::new(),
            specs: HashMap::new(),
            file_paths: HashMap::new(),
            current_step: 0,
        };

        // Index 0 is the sentinel root node
        arena.nodes.push(Node::root());
        arena
    }

    /// Allocate a new node in the arena. Returns the node index.
    ///
    /// Reuses freed slots when available; otherwise appends.
    pub fn alloc(&mut self, mut node: Node) -> u32 {
        let idx = if let Some(free_idx) = self.free_list.pop() {
            node.id = free_idx;
            self.nodes[free_idx as usize] = node;
            free_idx
        } else {
            let idx = self.nodes.len() as u32;
            node.id = idx;
            self.nodes.push(node);
            idx
        };

        // Set itch bit — this node needs to be resolved
        self.itch.set(idx as usize);

        // Update title index
        let title_hash = fnv_hash(self.nodes[idx as usize].title.as_str());
        self.title_index.insert(title_hash, idx);

        // Update path index for impl_files
        for &file_hash in &self.nodes[idx as usize].impl_files {
            self.path_index
                .entry(file_hash)
                .or_default()
                .push(idx);
        }

        // Set created_step
        self.nodes[idx as usize].created_step = self.current_step;

        idx
    }

    /// Deallocate a node, returning it to the free list.
    pub fn dealloc(&mut self, idx: u32) {
        if idx == 0 || idx as usize >= self.nodes.len() {
            return;
        }

        // Clear itch bit
        self.itch.clear(idx as usize);

        // Remove from title index
        let title_hash = fnv_hash(self.nodes[idx as usize].title.as_str());
        self.title_index.remove(&title_hash);

        // Remove from path index
        for &file_hash in &self.nodes[idx as usize].impl_files {
            if let Some(entries) = self.path_index.get_mut(&file_hash) {
                entries.retain(|&i| i != idx);
            }
        }

        // Mark node as deallocated and add to free list
        self.nodes[idx as usize].status = NodeStatus::Deallocated;
        self.free_list.push(idx);
    }

    /// Get an immutable reference to a node.
    pub fn get(&self, idx: u32) -> &Node {
        &self.nodes[idx as usize]
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, idx: u32) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    /// Total number of live (non-deallocated) nodes, excluding root.
    pub fn live_count(&self) -> usize {
        self.nodes.len() - 1 - self.free_list.len()
    }

    /// Store the full specification text for a node.
    pub fn set_spec(&mut self, node_idx: u32, spec: String) {
        let hash = fnv_hash(&spec);
        self.nodes[node_idx as usize].spec_hash = hash;
        self.specs.insert(node_idx, spec);
    }

    /// Get the full specification text for a node.
    pub fn get_spec(&self, node_idx: u32) -> Option<&str> {
        self.specs.get(&node_idx).map(|s| s.as_str())
    }

    /// Register a file path and return its hash.
    pub fn register_file_path(&mut self, path: &str) -> u64 {
        let hash = fnv_hash(path);
        self.file_paths.entry(hash).or_insert_with(|| path.to_string());
        hash
    }

    /// Look up a file path by its hash.
    pub fn resolve_file_path(&self, hash: u64) -> Option<&str> {
        self.file_paths.get(&hash).map(|s| s.as_str())
    }

    /// Advance the step counter and return the new step number.
    pub fn advance_step(&mut self) -> u32 {
        self.current_step += 1;
        self.current_step
    }

    /// Mark a node as complete. Clears its itch bit.
    pub fn mark_complete(&mut self, idx: u32) {
        let node = &mut self.nodes[idx as usize];
        node.status = NodeStatus::Verified;
        node.completed_step = self.current_step;
        self.itch.clear(idx as usize);

        // Check if any dependents are now unblocked
        let dependents: SmallVec<[u32; 4]> = node.dependents.clone();
        for dep_idx in dependents {
            if self.are_dependencies_resolved(dep_idx) {
                // Add to pending queue if it's still pending
                let dep_node = &self.nodes[dep_idx as usize];
                if dep_node.status == NodeStatus::Pending {
                    self.pending_queue.push(PendingEntry {
                        node_idx: dep_idx,
                        priority: dep_node.priority,
                    });
                }
            }
        }
    }

    /// Mark a node as failed. Clears its itch bit so the system can
    /// eventually terminate even if some nodes fail.
    pub fn mark_failed(&mut self, idx: u32) {
        let node = &mut self.nodes[idx as usize];
        node.status = NodeStatus::Failed;
        node.retry_count += 1;
        self.itch.clear(idx as usize);
    }

    /// Check whether all dependencies of a node are resolved (Verified).
    pub fn are_dependencies_resolved(&self, idx: u32) -> bool {
        let node = &self.nodes[idx as usize];
        node.dependencies
            .iter()
            .all(|&dep_idx| self.nodes[dep_idx as usize].status == NodeStatus::Verified)
    }

    /// Add a dependency edge: `node_idx` depends on `dep_idx`.
    pub fn add_dependency(&mut self, node_idx: u32, dep_idx: u32) {
        let node = &mut self.nodes[node_idx as usize];
        if !node.dependencies.contains(&dep_idx) {
            node.dependencies.push(dep_idx);
        }

        let dep_node = &mut self.nodes[dep_idx as usize];
        if !dep_node.dependents.contains(&node_idx) {
            dep_node.dependents.push(node_idx);
        }
    }

    /// Add a parent-child relationship: `child_idx` is a child of `parent_idx`.
    pub fn add_child(&mut self, parent_idx: u32, child_idx: u32) {
        self.nodes[parent_idx as usize].children.push(child_idx);
        self.nodes[child_idx as usize].parent = parent_idx;
    }

    /// Whether the itch gate allows termination (no active nodes remain).
    pub fn can_terminate(&self) -> bool {
        !self.itch.any_active()
    }

    /// Get the itch register statistics.
    pub fn itch_stats(&self) -> (u32, usize) {
        (self.itch.active_count(), self.nodes.len())
    }

    /// Iterate over all live nodes (excluding root and deallocated).
    pub fn live_nodes(&self) -> impl Iterator<Item = (u32, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1) // skip root
            .filter(|(_, n)| n.status != NodeStatus::Deallocated)
            .map(|(i, n)| (i as u32, n))
    }

    /// Find a node by title.
    pub fn find_by_title(&self, title: &str) -> Option<u32> {
        let hash = fnv_hash(title);
        self.title_index.get(&hash).copied()
    }

    /// Find all nodes that touch a given file path.
    pub fn nodes_touching_file(&self, file_path: &str) -> &[u32] {
        let hash = fnv_hash(file_path);
        self.path_index
            .get(&hash)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hashing utility
// ---------------------------------------------------------------------------

/// Compute FNV-1a hash for a string.
pub fn fnv_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::Priority;

    #[test]
    fn test_arena_alloc_and_get() {
        let mut arena = Arena::new();
        let node = Node::new("Test node".to_string(), Priority::Standard);
        let idx = arena.alloc(node);

        assert_eq!(arena.get(idx).title.as_str(), "Test node");
        assert_eq!(arena.live_count(), 1);
        assert!(!arena.can_terminate()); // itch bit is set
    }

    #[test]
    fn test_arena_mark_complete() {
        let mut arena = Arena::new();
        let node = Node::new("Test".to_string(), Priority::Standard);
        let idx = arena.alloc(node);

        assert!(!arena.can_terminate());
        arena.mark_complete(idx);
        assert!(arena.can_terminate());
    }

    #[test]
    fn test_arena_free_list_reuse() {
        let mut arena = Arena::new();
        let n1 = arena.alloc(Node::new("A".to_string(), Priority::Standard));
        let _n2 = arena.alloc(Node::new("B".to_string(), Priority::Standard));

        arena.mark_complete(n1);
        arena.dealloc(n1);

        // Allocating again should reuse the freed slot
        let n3 = arena.alloc(Node::new("C".to_string(), Priority::Standard));
        assert_eq!(n3, n1); // reused index
        assert_eq!(arena.get(n3).title.as_str(), "C");
    }

    #[test]
    fn test_arena_dependencies() {
        let mut arena = Arena::new();
        let n1 = arena.alloc(Node::new("Dep".to_string(), Priority::Critical));
        let n2 = arena.alloc(Node::new("Dependent".to_string(), Priority::Standard));

        arena.add_dependency(n2, n1);

        assert!(!arena.are_dependencies_resolved(n2));
        arena.mark_complete(n1);
        assert!(arena.are_dependencies_resolved(n2));
    }

    #[test]
    fn test_arena_title_lookup() {
        let mut arena = Arena::new();
        let _idx = arena.alloc(Node::new("Auth middleware".to_string(), Priority::Standard));

        let found = arena.find_by_title("Auth middleware");
        assert!(found.is_some());
        assert!(arena.find_by_title("Nonexistent").is_none());
    }

    #[test]
    fn test_arena_file_path_index() {
        let mut arena = Arena::new();
        let file_hash = arena.register_file_path("src/auth.rs");

        let mut node = Node::new("Auth".to_string(), Priority::Standard);
        node.impl_files.push(file_hash);
        let idx = arena.alloc(node);

        let touching = arena.nodes_touching_file("src/auth.rs");
        assert!(touching.contains(&idx));
    }

    #[test]
    fn test_mark_failed_clears_itch_bit() {
        let mut arena = Arena::new();
        let n1 = arena.alloc(Node::new("Will Fail".to_string(), Priority::Standard));
        let n2 = arena.alloc(Node::new("Will Succeed".to_string(), Priority::Standard));

        assert!(!arena.can_terminate()); // both itch bits set

        arena.mark_complete(n2);
        assert!(!arena.can_terminate()); // n1 still active

        arena.mark_failed(n1);
        assert!(arena.can_terminate()); // both cleared: system can terminate
    }
}
