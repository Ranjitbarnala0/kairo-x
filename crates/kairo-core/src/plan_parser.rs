//! Two-pass plan parser (§Flaw 4, §4.5).
//!
//! Converts LLM-generated plans into arena graph nodes.
//!
//! **Pass 1:** LLM generates a natural language plan (whatever format it wants).
//! **Pass 2:** LLM converts its own plan to JSON (same session, ~200 tokens).
//! **Parse:** `serde_json::from_str`. If parse fails → retry. Still fails → numbered list fallback.
//!
//! This two-pass approach costs one extra API call but eliminates an entire
//! class of parsing failures. JSON parsing is a solved problem.

use crate::arena::Arena;
use crate::arena::node::{Node, Priority};
use crate::arena::priority_queue::PendingEntry;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PlanParseError {
    #[error("JSON parse error: {0}")]
    JsonParse(String),

    #[error("Invalid plan structure: {0}")]
    InvalidStructure(String),

    #[error("Empty plan — no components generated")]
    EmptyPlan,

    #[error("Circular dependency detected: node {node_id} depends on itself")]
    CircularDependency { node_id: u32 },
}

// ---------------------------------------------------------------------------
// Plan node — the JSON structure from the LLM
// ---------------------------------------------------------------------------

/// A single component in the LLM-generated plan JSON.
///
/// The LLM outputs an array of these in Pass 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanComponent {
    /// LLM-assigned ID (1-based, used for dependency references).
    pub id: u32,
    /// Short title for the component.
    pub title: String,
    /// What to implement — the specification.
    pub spec: String,
    /// Priority level: "critical", "standard", or "mechanical".
    pub priority: String,
    /// IDs of components this one depends on.
    #[serde(default)]
    pub depends_on: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Plan summary (returned to caller)
// ---------------------------------------------------------------------------

/// Summary of a parsed plan after graph construction.
#[derive(Debug)]
pub struct PlanSummary {
    /// Total number of components in the plan.
    pub total_components: usize,
    /// Breakdown by priority.
    pub critical_count: usize,
    pub standard_count: usize,
    pub mechanical_count: usize,
    /// Arena node indices for the created nodes.
    pub node_indices: Vec<u32>,
}

// ---------------------------------------------------------------------------
// JSON parsing (primary path)
// ---------------------------------------------------------------------------

/// Parse the LLM's JSON plan output into a vector of PlanComponents.
///
/// This is the primary parsing path. The LLM was asked to output only valid JSON
/// in the format: `[{"id": 1, "title": "...", "spec": "...", "priority": "...", "depends_on": [...]}]`
pub fn parse_plan_json(json_text: &str) -> Result<Vec<PlanComponent>, PlanParseError> {
    // Strip any text before/after the JSON array
    let trimmed = extract_json_array(json_text);

    let components: Vec<PlanComponent> = serde_json::from_str(trimmed).map_err(|e| {
        PlanParseError::JsonParse(format!(
            "Failed to parse plan JSON: {e}. Input starts with: {}",
            &trimmed[..trimmed.len().min(200)]
        ))
    })?;

    if components.is_empty() {
        return Err(PlanParseError::EmptyPlan);
    }

    // Validate: no self-dependencies, all dependency references are valid
    let ids: Vec<u32> = components.iter().map(|c| c.id).collect();
    for component in &components {
        for &dep in &component.depends_on {
            if dep == component.id {
                return Err(PlanParseError::CircularDependency {
                    node_id: component.id,
                });
            }
            if !ids.contains(&dep) {
                return Err(PlanParseError::InvalidStructure(format!(
                    "Component {} depends on {}, which doesn't exist in the plan",
                    component.id, dep
                )));
            }
        }
    }

    // Validate: no multi-node cycles (e.g., A->B->A). Use topological sort
    // to detect any cycle in the dependency graph.
    {
        // Compute in-degree for each node (number of dependencies it has)
        let mut in_degree: std::collections::HashMap<u32, usize> =
            ids.iter().map(|&id| (id, 0)).collect();
        for c in &components {
            *in_degree.get_mut(&c.id).unwrap() = c.depends_on.len();
        }

        let mut queue: std::collections::VecDeque<u32> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut visited_count = 0usize;

        while let Some(node) = queue.pop_front() {
            visited_count += 1;
            // Find all components that depend on this node
            for c in &components {
                if c.depends_on.contains(&node) {
                    let deg = in_degree.get_mut(&c.id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(c.id);
                    }
                }
            }
        }

        if visited_count != ids.len() {
            // Find a node still in the cycle for the error message
            let cycle_node = in_degree
                .iter()
                .find(|(_, deg)| **deg > 0)
                .map(|(id, _)| *id)
                .unwrap_or(0);
            return Err(PlanParseError::CircularDependency { node_id: cycle_node });
        }
    }

    Ok(components)
}

/// Extract a JSON array from text that may contain surrounding prose.
///
/// The LLM might output: "Here is the plan:\n[{...}]\nDone."
/// We need just the "[{...}]" part.
fn extract_json_array(text: &str) -> &str {
    let trimmed = text.trim();

    // If it already starts with '[', use it directly
    if trimmed.starts_with('[') {
        // Find the matching closing bracket
        if let Some(end) = find_matching_bracket(trimmed) {
            return &trimmed[..=end];
        }
        return trimmed;
    }

    // Search for the first '[' and extract from there
    if let Some(start) = trimmed.find('[') {
        let rest = &trimmed[start..];
        if let Some(end) = find_matching_bracket(rest) {
            return &rest[..=end];
        }
        return rest;
    }

    trimmed
}

/// Find the position of the matching closing bracket for a string starting with '['.
fn find_matching_bracket(text: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Numbered list fallback (last resort)
// ---------------------------------------------------------------------------

/// Parse a numbered list plan as a last resort when JSON parsing fails twice.
///
/// Expected format:
/// ```text
/// 1. Title of first component
///    Description of what to implement
/// 2. Title of second component (depends on 1)
///    Description...
/// ```
pub fn parse_plan_numbered_list(text: &str) -> Result<Vec<PlanComponent>, PlanParseError> {
    let mut components = Vec::new();
    let mut current_id = 0u32;
    let mut current_title = String::new();
    let mut current_spec = String::new();
    let mut current_deps = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Check for numbered line: "1. Title..."
        if let Some(rest) = try_parse_numbered_line(trimmed) {
            // Save previous component if any
            if current_id > 0 && !current_title.is_empty() {
                components.push(PlanComponent {
                    id: current_id,
                    title: current_title.clone(),
                    spec: current_spec.trim().to_string(),
                    priority: "standard".to_string(), // Default; no priority info in list format
                    depends_on: current_deps.clone(),
                });
            }

            current_id = components.len() as u32 + 1;
            current_title = rest.to_string();
            current_spec = String::new();
            current_deps = Vec::new();

            // Check for "(depends on X, Y)" in the title
            if let Some(dep_start) = current_title.find("(depends on") {
                let dep_text = &current_title[dep_start..];
                current_deps = extract_dependency_ids(dep_text);
                current_title = current_title[..dep_start].trim().to_string();
            }
        } else if current_id > 0 && !trimmed.is_empty() {
            // Continuation line — add to spec
            if !current_spec.is_empty() {
                current_spec.push('\n');
            }
            current_spec.push_str(trimmed);
        }
    }

    // Don't forget the last component
    if current_id > 0 && !current_title.is_empty() {
        components.push(PlanComponent {
            id: current_id,
            title: current_title,
            spec: current_spec.trim().to_string(),
            priority: "standard".to_string(),
            depends_on: current_deps,
        });
    }

    if components.is_empty() {
        return Err(PlanParseError::EmptyPlan);
    }

    Ok(components)
}

/// Try to parse a line as "N. rest..." and return the rest.
fn try_parse_numbered_line(line: &str) -> Option<&str> {
    let mut chars = line.chars();
    let first = chars.next()?;
    if !first.is_ascii_digit() {
        return None;
    }

    // Consume remaining digits
    let rest_str = chars.as_str();
    let dot_pos = rest_str.find(". ")?;

    // Verify everything before the dot is also digits
    if rest_str[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
        Some(rest_str[dot_pos + 2..].trim())
    } else {
        None
    }
}

/// Extract dependency IDs from text like "(depends on 1, 3, 5)".
fn extract_dependency_ids(text: &str) -> Vec<u32> {
    text.chars()
        .filter(|c| c.is_ascii_digit() || *c == ' ')
        .collect::<String>()
        .split_whitespace()
        .filter_map(|s| s.parse::<u32>().ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build arena graph nodes from parsed plan components.
///
/// Returns a PlanSummary with the created node indices.
pub fn build_graph_from_plan(
    arena: &mut Arena,
    components: &[PlanComponent],
    parent_node: u32,
) -> Result<PlanSummary, PlanParseError> {
    // Map from LLM-assigned IDs to arena indices
    let mut id_to_arena_idx: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut node_indices = Vec::with_capacity(components.len());

    let mut critical_count = 0;
    let mut standard_count = 0;
    let mut mechanical_count = 0;

    // First pass: create all nodes
    for component in components {
        let priority = component
            .priority
            .parse::<Priority>()
            .unwrap_or(Priority::Standard);

        match priority {
            Priority::Critical => critical_count += 1,
            Priority::Standard => standard_count += 1,
            Priority::Mechanical => mechanical_count += 1,
        }

        let node = Node::new(component.title.clone(), priority);
        let arena_idx = arena.alloc(node);
        arena.set_spec(arena_idx, component.spec.clone());
        arena.add_child(parent_node, arena_idx);

        id_to_arena_idx.insert(component.id, arena_idx);
        node_indices.push(arena_idx);
    }

    // Second pass: wire up dependencies
    for component in components {
        let arena_idx = id_to_arena_idx[&component.id];
        for &dep_id in &component.depends_on {
            if let Some(&dep_arena_idx) = id_to_arena_idx.get(&dep_id) {
                arena.add_dependency(arena_idx, dep_arena_idx);
            }
        }
    }

    // Third pass: add nodes with resolved dependencies to the pending queue
    for component in components {
        let arena_idx = id_to_arena_idx[&component.id];
        if arena.are_dependencies_resolved(arena_idx) {
            let priority = arena.get(arena_idx).priority;
            arena.pending_queue.push(PendingEntry {
                node_idx: arena_idx,
                priority,
            });
        }
    }

    Ok(PlanSummary {
        total_components: components.len(),
        critical_count,
        standard_count,
        mechanical_count,
        node_indices,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_json_plan() {
        let json = r#"[
            {"id": 1, "title": "JWT Service", "spec": "Implement JWT with RS256", "priority": "critical", "depends_on": []},
            {"id": 2, "title": "Auth Middleware", "spec": "Token validation middleware", "priority": "standard", "depends_on": [1]},
            {"id": 3, "title": "Login Endpoint", "spec": "POST /auth/login", "priority": "standard", "depends_on": [1, 2]}
        ]"#;

        let components = parse_plan_json(json).unwrap();
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].title, "JWT Service");
        assert_eq!(components[1].depends_on, vec![1]);
        assert_eq!(components[2].depends_on, vec![1, 2]);
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let json = r#"Here is the plan:
        [
            {"id": 1, "title": "Setup", "spec": "Set up the project", "priority": "mechanical", "depends_on": []}
        ]
        That's my plan."#;

        let components = parse_plan_json(json).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].title, "Setup");
    }

    #[test]
    fn test_parse_json_self_dependency_error() {
        let json = r#"[{"id": 1, "title": "A", "spec": "do A", "priority": "standard", "depends_on": [1]}]"#;
        let result = parse_plan_json(json);
        assert!(matches!(result, Err(PlanParseError::CircularDependency { .. })));
    }

    #[test]
    fn test_parse_json_missing_dependency_error() {
        let json = r#"[{"id": 1, "title": "A", "spec": "do A", "priority": "standard", "depends_on": [99]}]"#;
        let result = parse_plan_json(json);
        assert!(matches!(result, Err(PlanParseError::InvalidStructure(_))));
    }

    #[test]
    fn test_parse_json_multi_node_cycle_error() {
        // A depends on B, B depends on A -- not a self-dependency, but a cycle
        let json = r#"[
            {"id": 1, "title": "A", "spec": "do A", "priority": "standard", "depends_on": [2]},
            {"id": 2, "title": "B", "spec": "do B", "priority": "standard", "depends_on": [1]}
        ]"#;
        let result = parse_plan_json(json);
        assert!(matches!(result, Err(PlanParseError::CircularDependency { .. })));
    }

    #[test]
    fn test_parse_json_three_node_cycle_error() {
        // A->B->C->A
        let json = r#"[
            {"id": 1, "title": "A", "spec": "do A", "priority": "standard", "depends_on": [3]},
            {"id": 2, "title": "B", "spec": "do B", "priority": "standard", "depends_on": [1]},
            {"id": 3, "title": "C", "spec": "do C", "priority": "standard", "depends_on": [2]}
        ]"#;
        let result = parse_plan_json(json);
        assert!(matches!(result, Err(PlanParseError::CircularDependency { .. })));
    }

    #[test]
    fn test_parse_json_single_node_no_deps() {
        let json = r#"[{"id": 1, "title": "Solo", "spec": "single task", "priority": "critical", "depends_on": []}]"#;
        let components = parse_plan_json(json).unwrap();
        assert_eq!(components.len(), 1);
        assert!(components[0].depends_on.is_empty());
    }

    #[test]
    fn test_parse_numbered_list() {
        let text = "1. JWT token service\n   Implement JWT with RS256 signing\n2. Auth middleware (depends on 1)\n   Token validation and extraction\n3. Login endpoint (depends on 1, 2)\n   POST /auth/login handler\n";

        let components = parse_plan_numbered_list(text).unwrap();
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].title, "JWT token service");
        assert_eq!(components[1].depends_on, vec![1]);
        assert_eq!(components[2].depends_on, vec![1, 2]);
    }

    #[test]
    fn test_build_graph() {
        let components = vec![
            PlanComponent {
                id: 1,
                title: "Auth".to_string(),
                spec: "Auth service".to_string(),
                priority: "critical".to_string(),
                depends_on: vec![],
            },
            PlanComponent {
                id: 2,
                title: "Middleware".to_string(),
                spec: "Auth middleware".to_string(),
                priority: "standard".to_string(),
                depends_on: vec![1],
            },
        ];

        let mut arena = Arena::new();
        let summary = build_graph_from_plan(&mut arena, &components, 0).unwrap();

        assert_eq!(summary.total_components, 2);
        assert_eq!(summary.critical_count, 1);
        assert_eq!(summary.standard_count, 1);
        assert_eq!(arena.live_count(), 2);

        // First node should be in the pending queue (no deps)
        // Second node should NOT be in pending queue (depends on first)
        let ready = arena.nodes_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(arena.get(ready[0]).title.as_str(), "Auth");
    }

    #[test]
    fn test_extract_json_array() {
        assert_eq!(extract_json_array("[1, 2, 3]"), "[1, 2, 3]");
        assert_eq!(extract_json_array("  [1]  "), "[1]");
        assert_eq!(extract_json_array("text [1] more"), "[1]");
    }
}
