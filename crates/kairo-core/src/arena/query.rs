//! Graph query methods for the arena.
//!
//! Provides efficient traversal, status queries, and graph analysis
//! operations used by the runtime loop and parallel scheduler.

use crate::arena::Arena;
use crate::arena::node::NodeStatus;

// ---------------------------------------------------------------------------
// Status-based queries
// ---------------------------------------------------------------------------

impl Arena {
    /// Get all nodes with a specific status.
    pub fn nodes_by_status(&self, status: NodeStatus) -> Vec<u32> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1) // skip root
            .filter(|(_, n)| n.status == status)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Get all nodes that are ready for execution:
    /// status == Pending and all dependencies resolved.
    pub fn nodes_ready(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(i, n)| {
                n.status == NodeStatus::Pending && self.are_dependencies_resolved(*i as u32)
            })
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Get all nodes that are currently being worked on (active).
    pub fn active_nodes(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, n)| n.status.is_active())
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Get all nodes that need fixing (FixNeeded status).
    pub fn nodes_needing_fix(&self) -> Vec<u32> {
        self.nodes_by_status(NodeStatus::FixNeeded)
    }

    /// Get all failed nodes.
    pub fn failed_nodes(&self) -> Vec<u32> {
        self.nodes_by_status(NodeStatus::Failed)
    }

    /// Get all verified (completed) nodes.
    pub fn completed_nodes(&self) -> Vec<u32> {
        self.nodes_by_status(NodeStatus::Verified)
    }

    /// Get counts of nodes by status category.
    pub fn status_summary(&self) -> StatusSummary {
        let mut summary = StatusSummary::default();

        for node in self.nodes.iter().skip(1) {
            match node.status {
                NodeStatus::Pending | NodeStatus::Ready => summary.pending += 1,
                NodeStatus::Implementing
                | NodeStatus::AwaitingVerification
                | NodeStatus::VerifyingDeterministic
                | NodeStatus::VerifyingAudit
                | NodeStatus::Fixing
                | NodeStatus::Decomposing => summary.active += 1,
                NodeStatus::FixNeeded => summary.fix_needed += 1,
                NodeStatus::Verified => summary.completed += 1,
                NodeStatus::Failed => summary.failed += 1,
                NodeStatus::Deallocated => {} // not counted
            }
        }

        summary.total = summary.pending + summary.active + summary.fix_needed
            + summary.completed + summary.failed;
        summary
    }
}

// ---------------------------------------------------------------------------
// Dependency chain queries
// ---------------------------------------------------------------------------

impl Arena {
    /// Get the full dependency chain (transitive) for a node.
    /// Returns all ancestors that must complete before this node can start.
    ///
    /// Uses a `HashSet` for O(1) visited/membership checks instead of
    /// `Vec::contains` which would be O(n) per check, making the whole
    /// traversal O(n^2) for deep chains.
    pub fn dependency_chain(&self, node_idx: u32) -> Vec<u32> {
        let mut chain = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_dependencies(node_idx, &mut chain, &mut visited);
        chain
    }

    fn collect_dependencies(&self, node_idx: u32, chain: &mut Vec<u32>, visited: &mut std::collections::HashSet<u32>) {
        if !visited.insert(node_idx) {
            return; // Already visited — cycle protection
        }

        for &dep_idx in &self.nodes[node_idx as usize].dependencies {
            if visited.contains(&dep_idx) {
                continue; // Already in the chain
            }
            chain.push(dep_idx);
            self.collect_dependencies(dep_idx, chain, visited);
        }
    }

    /// Get all downstream nodes (transitive dependents) that are blocked
    /// until this node completes.
    pub fn downstream_nodes(&self, node_idx: u32) -> Vec<u32> {
        let mut downstream = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_dependents(node_idx, &mut downstream, &mut visited);
        downstream
    }

    fn collect_dependents(
        &self,
        node_idx: u32,
        downstream: &mut Vec<u32>,
        visited: &mut std::collections::HashSet<u32>,
    ) {
        if !visited.insert(node_idx) {
            return;
        }

        for &dep_idx in &self.nodes[node_idx as usize].dependents {
            if visited.contains(&dep_idx) {
                continue;
            }
            downstream.push(dep_idx);
            self.collect_dependents(dep_idx, downstream, visited);
        }
    }

    /// Get the critical path -- the longest weighted path through the
    /// non-terminal portion of the dependency DAG. Returns an ordered
    /// path from root dependency to leaf (inclusive).
    ///
    /// Uses Kahn's algorithm to produce a topological order, then
    /// relaxes edges to find the longest path (standard DAG longest-path
    /// in O(V + E)).
    pub fn critical_path(&self) -> Vec<u32> {
        use std::collections::VecDeque;

        // Collect live (non-terminal) node indices.
        let live: Vec<u32> = self
            .nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, n)| !n.status.is_terminal())
            .map(|(i, _)| i as u32)
            .collect();

        if live.is_empty() {
            return Vec::new();
        }

        let max_idx = self.nodes.len();
        // in_degree restricted to the live subgraph
        let mut in_degree = vec![0u32; max_idx];
        let mut is_live = vec![false; max_idx];
        for &idx in &live {
            is_live[idx as usize] = true;
        }
        for &idx in &live {
            for &dep in &self.nodes[idx as usize].dependencies {
                if is_live[dep as usize] {
                    in_degree[idx as usize] += 1;
                }
            }
        }

        // Kahn's topological sort over the live subgraph
        let mut queue = VecDeque::new();
        for &idx in &live {
            if in_degree[idx as usize] == 0 {
                queue.push_back(idx);
            }
        }

        let mut topo_order = Vec::with_capacity(live.len());
        while let Some(idx) = queue.pop_front() {
            topo_order.push(idx);
            for &dep_idx in &self.nodes[idx as usize].dependents {
                if !is_live[dep_idx as usize] {
                    continue;
                }
                in_degree[dep_idx as usize] -= 1;
                if in_degree[dep_idx as usize] == 0 {
                    queue.push_back(dep_idx);
                }
            }
        }

        // Longest-path relaxation (each node has weight 1).
        let mut dist = vec![0u32; max_idx];
        let mut predecessor: Vec<Option<u32>> = vec![None; max_idx];

        for &idx in &topo_order {
            for &dep_idx in &self.nodes[idx as usize].dependents {
                if !is_live[dep_idx as usize] {
                    continue;
                }
                let new_dist = dist[idx as usize] + 1;
                if new_dist > dist[dep_idx as usize] {
                    dist[dep_idx as usize] = new_dist;
                    predecessor[dep_idx as usize] = Some(idx);
                }
            }
        }

        // Find the node with the maximum distance (end of critical path).
        let end_node = live
            .iter()
            .copied()
            .max_by_key(|&idx| dist[idx as usize]);

        let end_node = match end_node {
            Some(n) => n,
            None => return Vec::new(),
        };

        // Trace back from end to start via predecessors.
        let mut path = Vec::new();
        let mut current = Some(end_node);
        while let Some(idx) = current {
            path.push(idx);
            current = predecessor[idx as usize];
        }
        path.reverse();
        path
    }
}

// ---------------------------------------------------------------------------
// Child/parent queries
// ---------------------------------------------------------------------------

impl Arena {
    /// Get all children of a node (direct, not transitive).
    pub fn children_of(&self, node_idx: u32) -> &[u32] {
        // SmallVec derefs to slice
        &self.nodes[node_idx as usize].children
    }

    /// Get all descendant nodes (transitive children).
    ///
    /// Uses a visited set to protect against cycles in the child graph
    /// (which should not occur in a well-formed tree but guards against corruption).
    pub fn descendants(&self, node_idx: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(node_idx);
        let mut stack = vec![node_idx];

        while let Some(idx) = stack.pop() {
            for &child in &self.nodes[idx as usize].children {
                if visited.insert(child) {
                    result.push(child);
                    stack.push(child);
                }
            }
        }

        result
    }

    /// Get the depth of a node in the tree (root = 0).
    ///
    /// Bounded to a maximum depth of 1000 to prevent infinite loops if the
    /// parent chain contains a cycle due to corruption.
    pub fn depth(&self, node_idx: u32) -> u32 {
        const MAX_DEPTH: u32 = 1000;
        let mut depth = 0u32;
        let mut current = node_idx;
        let mut visited = std::collections::HashSet::new();
        visited.insert(current);

        while current != 0 && depth < MAX_DEPTH {
            current = self.nodes[current as usize].parent;
            if !visited.insert(current) {
                // Cycle detected in parent chain -- break to prevent infinite loop
                break;
            }
            depth += 1;
        }

        depth
    }

    /// Get all sibling nodes (same parent, excluding self).
    pub fn siblings(&self, node_idx: u32) -> Vec<u32> {
        let parent = self.nodes[node_idx as usize].parent;
        self.nodes[parent as usize]
            .children
            .iter()
            .copied()
            .filter(|&c| c != node_idx)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// File-related queries
// ---------------------------------------------------------------------------

impl Arena {
    /// Get all unique file hashes touched by pending/active nodes.
    pub fn files_in_flight(&self) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for node in self.nodes.iter().skip(1) {
            if node.status.is_active() {
                for &fh in &node.impl_files {
                    if seen.insert(fh) {
                        files.push(fh);
                    }
                }
            }
        }
        files
    }

    /// Check if two nodes share any impl_files (potential conflict).
    pub fn nodes_share_files(&self, a: u32, b: u32) -> bool {
        let a_files = &self.nodes[a as usize].impl_files;
        let b_files = &self.nodes[b as usize].impl_files;

        for af in a_files.iter() {
            if b_files.contains(af) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Session queries
// ---------------------------------------------------------------------------

impl Arena {
    /// Find the node currently bound to a given session.
    pub fn node_for_session(&self, session_id: u32) -> Option<u32> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, n)| n.session_id == session_id && n.status != NodeStatus::Deallocated)
            .map(|(i, _)| i as u32)
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary of node statuses.
#[derive(Debug, Default, Clone)]
pub struct StatusSummary {
    pub total: usize,
    pub pending: usize,
    pub active: usize,
    pub fix_needed: usize,
    pub completed: usize,
    pub failed: usize,
}

impl StatusSummary {
    /// Progress as a percentage (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.completed as f64 / self.total as f64
    }
}

impl std::fmt::Display for StatusSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} completed ({} pending, {} active, {} fix needed, {} failed)",
            self.completed,
            self.total,
            self.pending,
            self.active,
            self.fix_needed,
            self.failed
        )
    }
}

// ---------------------------------------------------------------------------
// Total LLM cost tracking across all nodes
// ---------------------------------------------------------------------------

impl Arena {
    /// Sum of all LLM calls spent across all nodes.
    pub fn total_llm_calls(&self) -> u32 {
        self.nodes
            .iter()
            .skip(1)
            .map(|n| n.llm_calls_spent as u32)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::{Node, Priority};

    fn make_arena_with_chain() -> Arena {
        // Create: A -> B -> C (A must complete before B, B before C)
        let mut arena = Arena::new();

        let a = arena.alloc(Node::new("A".to_string(), Priority::Critical));
        let b = arena.alloc(Node::new("B".to_string(), Priority::Standard));
        let c = arena.alloc(Node::new("C".to_string(), Priority::Mechanical));

        arena.add_dependency(b, a).expect("no cycle");
        arena.add_dependency(c, b).expect("no cycle");

        arena
    }

    #[test]
    fn test_nodes_ready() {
        let arena = make_arena_with_chain();

        let ready = arena.nodes_ready();
        // Only A should be ready (no dependencies)
        assert_eq!(ready.len(), 1);
        assert_eq!(arena.get(ready[0]).expect("node must exist").title.as_str(), "A");
    }

    #[test]
    fn test_dependency_chain() {
        let arena = make_arena_with_chain();
        let c_idx = arena.find_by_title("C").unwrap();

        let chain = arena.dependency_chain(c_idx);
        assert_eq!(chain.len(), 2); // B and A
    }

    #[test]
    fn test_downstream_nodes() {
        let arena = make_arena_with_chain();
        let a_idx = arena.find_by_title("A").unwrap();

        let downstream = arena.downstream_nodes(a_idx);
        assert_eq!(downstream.len(), 2); // B and C
    }

    #[test]
    fn test_status_summary() {
        let mut arena = make_arena_with_chain();
        let summary = arena.status_summary();

        assert_eq!(summary.total, 3);
        assert_eq!(summary.pending, 3);
        assert_eq!(summary.completed, 0);

        let a_idx = arena.find_by_title("A").unwrap();
        arena.mark_complete(a_idx);

        let summary = arena.status_summary();
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.pending, 2);
    }

    #[test]
    fn test_siblings() {
        let mut arena = Arena::new();
        let parent = arena.alloc(Node::new("Parent".to_string(), Priority::Standard));
        let c1 = arena.alloc(Node::new("Child1".to_string(), Priority::Standard));
        let c2 = arena.alloc(Node::new("Child2".to_string(), Priority::Standard));
        let c3 = arena.alloc(Node::new("Child3".to_string(), Priority::Standard));

        arena.add_child(parent, c1);
        arena.add_child(parent, c2);
        arena.add_child(parent, c3);

        let sibs = arena.siblings(c2);
        assert_eq!(sibs.len(), 2);
        assert!(sibs.contains(&c1));
        assert!(sibs.contains(&c3));
    }
}
