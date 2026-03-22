//! Token and cost tracking (§5.3).
//!
//! Tracks cumulative token usage, cost in microcents, and budget limits.
//! Supports per-node tracking and three cost modes that influence verification
//! depth and parallel execution decisions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cost mode
// ---------------------------------------------------------------------------

/// Cost mode controls the trade-off between quality and spending.
///
/// Selected by the user at invocation time; affects verification depth,
/// parallel track count, and audit frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CostMode {
    /// Maximum quality: full verification for all priorities, deep audits,
    /// more parallel tracks. ~2-3x cost of Balanced.
    Thorough,
    /// Default mode: full verification for Critical/Standard, light for
    /// Mechanical. Standard parallelism.
    #[default]
    Balanced,
    /// Minimum spending: skip LLM audits for Standard/Mechanical, fewer
    /// parallel tracks, aggressive context trimming.
    Efficient,
}

impl CostMode {
    /// Whether LLM audit (Layer 2) should be performed for the given priority.
    pub fn should_audit(&self, priority: crate::arena::node::Priority) -> bool {
        use crate::arena::node::Priority;
        match (self, priority) {
            (Self::Thorough, _) => true,
            (Self::Balanced, Priority::Critical | Priority::Standard) => true,
            (Self::Balanced, Priority::Mechanical) => false,
            (Self::Efficient, Priority::Critical) => true,
            (Self::Efficient, _) => false,
        }
    }

    /// Maximum number of parallel execution tracks.
    pub fn max_parallel_tracks(&self) -> usize {
        match self {
            Self::Thorough => 5,
            Self::Balanced => 3,
            Self::Efficient => 2,
        }
    }
}


impl std::fmt::Display for CostMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Thorough => write!(f, "Thorough"),
            Self::Balanced => write!(f, "Balanced"),
            Self::Efficient => write!(f, "Efficient"),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-node usage
// ---------------------------------------------------------------------------

/// Token and cost usage for a single node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeUsage {
    /// Total input tokens consumed by this node.
    pub input_tokens: u32,
    /// Total output tokens generated for this node.
    pub output_tokens: u32,
    /// Total cost for this node in microdollars (1/1_000_000 of a dollar).
    pub cost_microdollars: u64,
    /// Number of LLM calls made for this node.
    pub call_count: u32,
}

impl NodeUsage {
    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ---------------------------------------------------------------------------
// TokenTracker
// ---------------------------------------------------------------------------

/// Tracks token usage, costs, and budgets across the entire execution.
///
/// Maintains both global totals and per-node breakdowns. Supports both a
/// **token budget** (total input+output tokens) and a **cost budget**
/// (microdollars). The runtime can check either limit before making
/// additional LLM calls.
///
/// # Units
///
/// - Cost rates come from the provider config as **millidollars per million
///   tokens** (e.g., 3000 = $3.00/Mtok). Internally these are converted to
///   **microdollars** (1/1,000,000 of a dollar) for precision.
/// - `total_cost_microdollars` and `cost_limit_microdollars` are in
///   microdollars.
/// - `token_budget` is in **tokens** (input + output combined).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTracker {
    /// Total input tokens consumed across all calls.
    pub total_input: u64,
    /// Total output tokens generated across all calls.
    pub total_output: u64,
    /// Total cost in microdollars (1/1_000_000 of a dollar).
    pub total_cost_microdollars: u64,
    /// Total number of LLM calls made.
    pub total_calls: u64,
    /// Per-node usage breakdown.
    pub per_node: HashMap<u32, NodeUsage>,
    /// Cost budget limit in microdollars. 0 = unlimited.
    pub cost_limit_microdollars: u64,
    /// Token budget limit (total input+output tokens). 0 = unlimited.
    pub token_budget: u64,
    /// The cost mode governing spend decisions.
    pub cost_mode: CostMode,
    /// Cost per million input tokens in microdollars (converted from config).
    cost_per_input_mtok: u64,
    /// Cost per million output tokens in microdollars (converted from config).
    cost_per_output_mtok: u64,
}

impl TokenTracker {
    /// Create a new tracker with the given cost rates and budget limits.
    ///
    /// `cost_per_input_mtok` and `cost_per_output_mtok` are in **millidollars
    /// per million tokens** — the same unit used in the config file and
    /// `ProviderSpec` (e.g., 3000 = $3.00/Mtok). Internally these are
    /// multiplied by 1000 to convert to microdollars for precision.
    ///
    /// `cost_limit_microdollars`: cost budget in microdollars (0 = unlimited).
    /// `token_budget`: token budget in tokens (0 = unlimited).
    pub fn new(
        cost_per_input_mtok: u64,
        cost_per_output_mtok: u64,
        cost_limit_microdollars: u64,
        token_budget: u64,
        cost_mode: CostMode,
    ) -> Self {
        Self {
            total_input: 0,
            total_output: 0,
            total_cost_microdollars: 0,
            total_calls: 0,
            per_node: HashMap::new(),
            cost_limit_microdollars,
            token_budget,
            cost_mode,
            // Convert from millidollars/Mtok to microdollars/Mtok
            cost_per_input_mtok: cost_per_input_mtok * 1000,
            cost_per_output_mtok: cost_per_output_mtok * 1000,
        }
    }

    /// Record usage from a completed LLM call.
    pub fn record_usage(
        &mut self,
        node_id: u32,
        input_tokens: u32,
        output_tokens: u32,
    ) {
        let call_cost = self.calculate_cost(input_tokens, output_tokens);

        // Update global totals
        self.total_input += input_tokens as u64;
        self.total_output += output_tokens as u64;
        self.total_cost_microdollars += call_cost;
        self.total_calls += 1;

        // Update per-node tracking
        let node = self.per_node.entry(node_id).or_default();
        node.input_tokens = node.input_tokens.saturating_add(input_tokens);
        node.output_tokens = node.output_tokens.saturating_add(output_tokens);
        node.cost_microdollars += call_cost;
        node.call_count = node.call_count.saturating_add(1);
    }

    /// Calculate the cost for a given token usage.
    fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> u64 {
        let input_cost =
            (input_tokens as u64 * self.cost_per_input_mtok) / 1_000_000;
        let output_cost =
            (output_tokens as u64 * self.cost_per_output_mtok) / 1_000_000;
        input_cost + output_cost
    }

    /// Total cost so far in microdollars.
    pub fn cost_so_far(&self) -> u64 {
        self.total_cost_microdollars
    }

    /// Total cost so far formatted as dollars.
    pub fn cost_dollars(&self) -> f64 {
        self.total_cost_microdollars as f64 / 1_000_000.0
    }

    /// Remaining cost budget in microdollars. Returns `None` if no cost budget
    /// is set (unlimited).
    pub fn cost_budget_remaining(&self) -> Option<u64> {
        if self.cost_limit_microdollars == 0 {
            return None; // unlimited
        }
        Some(self.cost_limit_microdollars.saturating_sub(self.total_cost_microdollars))
    }

    /// Remaining token budget. Returns `None` if no token budget is set
    /// (unlimited).
    pub fn token_budget_remaining(&self) -> Option<u64> {
        if self.token_budget == 0 {
            return None; // unlimited
        }
        Some(self.token_budget.saturating_sub(self.total_input + self.total_output))
    }

    /// Whether either budget (cost or token) has been exhausted.
    ///
    /// Returns `false` if no budget limit is set.
    pub fn is_budget_exhausted(&self) -> bool {
        let cost_exhausted = self.cost_limit_microdollars > 0
            && self.total_cost_microdollars >= self.cost_limit_microdollars;
        let token_exhausted = self.token_budget > 0
            && (self.total_input + self.total_output) >= self.token_budget;
        cost_exhausted || token_exhausted
    }

    /// Whether either budget will likely be exhausted by an additional call
    /// with the estimated token counts.
    pub fn would_exceed_budget(&self, estimated_input: u32, estimated_output: u32) -> bool {
        // Check cost budget
        if self.cost_limit_microdollars > 0 {
            let estimated_cost = self.calculate_cost(estimated_input, estimated_output);
            if self.total_cost_microdollars + estimated_cost > self.cost_limit_microdollars {
                return true;
            }
        }
        // Check token budget
        if self.token_budget > 0 {
            let estimated_tokens = estimated_input as u64 + estimated_output as u64;
            if self.total_input + self.total_output + estimated_tokens > self.token_budget {
                return true;
            }
        }
        false
    }

    /// Total tokens consumed (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_input.saturating_add(self.total_output)
    }

    /// Get usage for a specific node.
    pub fn node_usage(&self, node_id: u32) -> Option<&NodeUsage> {
        self.per_node.get(&node_id)
    }

    /// Get the most expensive node by cost.
    pub fn most_expensive_node(&self) -> Option<(u32, &NodeUsage)> {
        self.per_node
            .iter()
            .max_by_key(|(_, usage)| usage.cost_microdollars)
            .map(|(&id, usage)| (id, usage))
    }

    /// Update the cost rates (e.g., after provider failover to a different model).
    ///
    /// Rates are in **millidollars per million tokens** (same unit as the
    /// constructor). They are converted to microdollars internally (* 1000).
    pub fn update_rates(&mut self, cost_per_input_mtok: u64, cost_per_output_mtok: u64) {
        self.cost_per_input_mtok = cost_per_input_mtok * 1000;
        self.cost_per_output_mtok = cost_per_output_mtok * 1000;
    }

    /// Generate a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "Tokens: {} in / {} out ({} total) | Cost: ${:.4} | Calls: {} | Nodes: {}",
            self.total_input,
            self.total_output,
            self.total_tokens(),
            self.cost_dollars(),
            self.total_calls,
            self.per_node.len(),
        )
    }
}

impl Default for TokenTracker {
    fn default() -> Self {
        Self::new(0, 0, 0, 0, CostMode::Balanced)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::Priority;

    // Anthropic Claude rates: $3/Mtok input, $15/Mtok output
    // In millidollars per Mtok (the config unit): 3_000 input, 15_000 output
    // TokenTracker::new() converts these to microdollars internally (* 1000)
    const INPUT_RATE: u64 = 3_000;
    const OUTPUT_RATE: u64 = 15_000;

    // Internal microdollars after conversion
    const INPUT_RATE_MICRO: u64 = 3_000_000;
    const OUTPUT_RATE_MICRO: u64 = 15_000_000;

    #[test]
    fn test_record_usage() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);
        tracker.record_usage(1, 1000, 500);

        assert_eq!(tracker.total_input, 1000);
        assert_eq!(tracker.total_output, 500);
        assert_eq!(tracker.total_calls, 1);
        assert!(tracker.total_cost_microdollars > 0);
    }

    #[test]
    fn test_cost_calculation() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);

        // 1M input tokens = $3 = 3_000_000 microdollars
        // 1M output tokens = $15 = 15_000_000 microdollars
        tracker.record_usage(1, 1_000_000, 1_000_000);

        assert_eq!(tracker.total_cost_microdollars, INPUT_RATE_MICRO + OUTPUT_RATE_MICRO);
    }

    #[test]
    fn test_per_node_tracking() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);

        tracker.record_usage(1, 1000, 500);
        tracker.record_usage(1, 800, 300);
        tracker.record_usage(2, 500, 200);

        let node1 = tracker.node_usage(1).unwrap();
        assert_eq!(node1.input_tokens, 1800);
        assert_eq!(node1.output_tokens, 800);
        assert_eq!(node1.call_count, 2);

        let node2 = tracker.node_usage(2).unwrap();
        assert_eq!(node2.input_tokens, 500);
        assert_eq!(node2.call_count, 1);
    }

    #[test]
    fn test_budget_unlimited() {
        let tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);
        assert!(!tracker.is_budget_exhausted());
        assert!(tracker.cost_budget_remaining().is_none());
        assert!(tracker.token_budget_remaining().is_none());
    }

    #[test]
    fn test_cost_budget_exhausted() {
        let cost_budget = 100_000; // 100k microdollars = $0.10
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, cost_budget, 0, CostMode::Balanced);

        // Make a large call that exceeds cost budget
        tracker.record_usage(1, 1_000_000, 1_000_000);
        assert!(tracker.is_budget_exhausted());
    }

    #[test]
    fn test_token_budget_exhausted() {
        let token_budget = 500_000; // 500k tokens
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, token_budget, CostMode::Balanced);

        // Make a call that exceeds token budget
        tracker.record_usage(1, 300_000, 300_000); // 600k tokens > 500k
        assert!(tracker.is_budget_exhausted());
    }

    #[test]
    fn test_cost_budget_remaining() {
        let cost_budget = 10_000_000; // $10
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, cost_budget, 0, CostMode::Balanced);

        tracker.record_usage(1, 100_000, 50_000);
        let remaining = tracker.cost_budget_remaining().expect("budget is set");
        assert!(remaining > 0);
        assert!(remaining < cost_budget);
    }

    #[test]
    fn test_token_budget_remaining() {
        let token_budget = 1_000_000;
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, token_budget, CostMode::Balanced);

        tracker.record_usage(1, 100_000, 50_000);
        assert_eq!(tracker.token_budget_remaining(), Some(850_000));
    }

    #[test]
    fn test_would_exceed_budget() {
        // Test cost budget
        let cost_budget = 1_000; // Very small cost budget
        let tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, cost_budget, 0, CostMode::Balanced);

        // Large call should exceed
        assert!(tracker.would_exceed_budget(1_000_000, 1_000_000));
        // Tiny call should be fine
        assert!(!tracker.would_exceed_budget(1, 1));
    }

    #[test]
    fn test_would_exceed_token_budget() {
        // Test token budget
        let token_budget = 10_000;
        let tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, token_budget, CostMode::Balanced);

        // Large call should exceed
        assert!(tracker.would_exceed_budget(6_000, 6_000));
        // Small call should be fine
        assert!(!tracker.would_exceed_budget(100, 100));
    }

    #[test]
    fn test_cost_mode_should_audit() {
        // Thorough: audit everything
        assert!(CostMode::Thorough.should_audit(Priority::Critical));
        assert!(CostMode::Thorough.should_audit(Priority::Standard));
        assert!(CostMode::Thorough.should_audit(Priority::Mechanical));

        // Balanced: audit Critical and Standard
        assert!(CostMode::Balanced.should_audit(Priority::Critical));
        assert!(CostMode::Balanced.should_audit(Priority::Standard));
        assert!(!CostMode::Balanced.should_audit(Priority::Mechanical));

        // Efficient: audit only Critical
        assert!(CostMode::Efficient.should_audit(Priority::Critical));
        assert!(!CostMode::Efficient.should_audit(Priority::Standard));
        assert!(!CostMode::Efficient.should_audit(Priority::Mechanical));
    }

    #[test]
    fn test_cost_mode_parallel_tracks() {
        assert_eq!(CostMode::Thorough.max_parallel_tracks(), 5);
        assert_eq!(CostMode::Balanced.max_parallel_tracks(), 3);
        assert_eq!(CostMode::Efficient.max_parallel_tracks(), 2);
    }

    #[test]
    fn test_most_expensive_node() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);

        tracker.record_usage(1, 100, 50);
        tracker.record_usage(2, 10_000, 5_000);
        tracker.record_usage(3, 500, 200);

        let (id, _) = tracker.most_expensive_node().unwrap();
        assert_eq!(id, 2);
    }

    #[test]
    fn test_summary_format() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);
        tracker.record_usage(1, 1000, 500);

        let summary = tracker.summary();
        assert!(summary.contains("1000 in"));
        assert!(summary.contains("500 out"));
        assert!(summary.contains("Calls: 1"));
    }

    #[test]
    fn test_update_rates() {
        let mut tracker = TokenTracker::new(INPUT_RATE, OUTPUT_RATE, 0, 0, CostMode::Balanced);
        tracker.record_usage(1, 1_000_000, 1_000_000);
        let cost_before = tracker.total_cost_microdollars;

        // Change to cheaper rates (in millidollars/Mtok — same unit as constructor)
        tracker.update_rates(1_000, 5_000);
        tracker.record_usage(2, 1_000_000, 1_000_000);

        // Second call should be cheaper
        let cost_after = tracker.total_cost_microdollars - cost_before;
        assert!(cost_after < cost_before);
    }
}
