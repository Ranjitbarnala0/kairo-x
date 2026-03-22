//! Input packet assembly for the controller (§7.4).
//!
//! Assembles a 32-slot input packet from the current execution state.
//! Each slot is a fixed-size feature vector of d_model (288) dimensions.
//!
//! The packet captures: graph state, active node, recent LLM responses,
//! project state, context candidates, verification results, itch state,
//! session state, cost state, and user state.


/// Number of input slots in the packet.
pub const N_INPUT_SLOTS: usize = 32;

/// Feature dimension per slot (d_model).
pub const D_MODEL: usize = 288;

// ---------------------------------------------------------------------------
// Input packet
// ---------------------------------------------------------------------------

/// A 32-slot input packet ready for the controller.
///
/// Each slot is a f32 vector of D_MODEL dimensions, laid out contiguously
/// for efficient matrix multiplication.
#[derive(Debug, Clone)]
pub struct InputPacket {
    /// Flat data: [N_INPUT_SLOTS × D_MODEL] in row-major order.
    pub data: Vec<f32>,
}

impl InputPacket {
    /// Create a zero-initialized input packet.
    pub fn new() -> Self {
        Self {
            data: vec![0.0; N_INPUT_SLOTS * D_MODEL],
        }
    }

    /// Get a mutable slice for a specific slot.
    ///
    /// # Panics
    /// Panics with a descriptive message if `slot_idx >= N_INPUT_SLOTS`.
    pub fn slot_mut(&mut self, slot_idx: usize) -> &mut [f32] {
        assert!(
            slot_idx < N_INPUT_SLOTS,
            "slot index {} out of range [0, {})",
            slot_idx,
            N_INPUT_SLOTS,
        );
        let start = slot_idx * D_MODEL;
        &mut self.data[start..start + D_MODEL]
    }

    /// Get an immutable slice for a specific slot.
    ///
    /// # Panics
    /// Panics with a descriptive message if `slot_idx >= N_INPUT_SLOTS`.
    pub fn slot(&self, slot_idx: usize) -> &[f32] {
        assert!(
            slot_idx < N_INPUT_SLOTS,
            "slot index {} out of range [0, {})",
            slot_idx,
            N_INPUT_SLOTS,
        );
        let start = slot_idx * D_MODEL;
        &self.data[start..start + D_MODEL]
    }

    /// Total number of f32 values.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the packet is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for InputPacket {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Slot assignment (§7.4)
// ---------------------------------------------------------------------------

/// Slot ranges for each packet type.
pub mod slots {
    /// Execution graph summary: active nodes, top pending, parallel states.
    pub const EXECUTION_GRAPH: std::ops::Range<usize> = 0..6;
    /// Active node context: spec fingerprint, status, deps.
    pub const ACTIVE_NODE: std::ops::Range<usize> = 6..10;
    /// Recent LLM responses: last 4 classifications.
    pub const RECENT_RESPONSES: std::ops::Range<usize> = 10..14;
    /// Project state: modified files, build status.
    pub const PROJECT_STATE: std::ops::Range<usize> = 14..17;
    /// Available context candidates: top items with relevance features.
    pub const CONTEXT_CANDIDATES: std::ops::Range<usize> = 17..22;
    /// Verification state: last L1 + L2 results.
    pub const VERIFICATION: std::ops::Range<usize> = 22..25;
    /// Itch state: pending count, failed count, oldest pending.
    pub const ITCH_STATE: std::ops::Range<usize> = 25..27;
    /// Session state: turn count, token spend, context window usage.
    pub const SESSION_STATE: std::ops::Range<usize> = 27..29;
    /// Cost state: budget remaining, cost mode.
    pub const COST_STATE: std::ops::Range<usize> = 29..30;
    /// User state: last user message recency.
    pub const USER_STATE: std::ops::Range<usize> = 30..31;
    /// Padding.
    pub const PADDING: std::ops::Range<usize> = 31..32;
}

// ---------------------------------------------------------------------------
// Feature encoding helpers
// ---------------------------------------------------------------------------

/// Encode a u32 value into a feature vector using a simple position encoding.
///
/// Spreads the value across the first few dimensions of the slot,
/// with log-scaled and normalized representations.
pub fn encode_count(slot: &mut [f32], value: u32, max_expected: u32) {
    let normalized = (value as f32) / (max_expected.max(1) as f32);
    let log_scaled = (1.0 + value as f32).ln() / (1.0 + max_expected as f32).ln();

    if slot.len() >= 4 {
        slot[0] = normalized.min(1.0);
        slot[1] = log_scaled.min(1.0);
        slot[2] = if value == 0 { 1.0 } else { 0.0 }; // is_zero flag
        slot[3] = if value >= max_expected { 1.0 } else { 0.0 }; // is_at_max flag
    }
}

/// Encode a f32 value into a feature vector.
pub fn encode_float(slot: &mut [f32], value: f32, idx: usize) {
    if idx < slot.len() {
        slot[idx] = value;
    }
}

/// Encode a boolean into a feature vector.
pub fn encode_bool(slot: &mut [f32], value: bool, idx: usize) {
    if idx < slot.len() {
        slot[idx] = if value { 1.0 } else { 0.0 };
    }
}

/// One-hot encode a categorical value.
pub fn encode_onehot(slot: &mut [f32], value: usize, num_classes: usize, start_idx: usize) {
    for i in 0..num_classes {
        if start_idx + i < slot.len() {
            slot[start_idx + i] = if i == value { 1.0 } else { 0.0 };
        }
    }
}

/// Encode a priority value (Critical=2, Standard=1, Mechanical=0).
pub fn encode_priority(slot: &mut [f32], priority: crate::arena::node::Priority, start_idx: usize) {
    let val = match priority {
        crate::arena::node::Priority::Critical => 2,
        crate::arena::node::Priority::Standard => 1,
        crate::arena::node::Priority::Mechanical => 0,
    };
    encode_onehot(slot, val, 3, start_idx);
}

/// Encode a node status as a one-hot vector (12 possible states).
pub fn encode_node_status(
    slot: &mut [f32],
    status: crate::arena::node::NodeStatus,
    start_idx: usize,
) {
    let val = match status {
        crate::arena::node::NodeStatus::Pending => 0,
        crate::arena::node::NodeStatus::Ready => 1,
        crate::arena::node::NodeStatus::Implementing => 2,
        crate::arena::node::NodeStatus::AwaitingVerification => 3,
        crate::arena::node::NodeStatus::VerifyingDeterministic => 4,
        crate::arena::node::NodeStatus::VerifyingAudit => 5,
        crate::arena::node::NodeStatus::FixNeeded => 6,
        crate::arena::node::NodeStatus::Fixing => 7,
        crate::arena::node::NodeStatus::Verified => 8,
        crate::arena::node::NodeStatus::Failed => 9,
        crate::arena::node::NodeStatus::Decomposing => 10,
        crate::arena::node::NodeStatus::Deallocated => 11,
    };
    encode_onehot(slot, val, 12, start_idx);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_packet_dimensions() {
        let packet = InputPacket::new();
        assert_eq!(packet.len(), N_INPUT_SLOTS * D_MODEL);
        assert_eq!(packet.len(), 32 * 288);
        assert_eq!(packet.len(), 9216);
    }

    #[test]
    fn test_slot_access() {
        let mut packet = InputPacket::new();
        let slot = packet.slot_mut(0);
        slot[0] = 1.0;
        slot[287] = 2.0;

        assert_eq!(packet.slot(0)[0], 1.0);
        assert_eq!(packet.slot(0)[287], 2.0);
        assert_eq!(packet.slot(1)[0], 0.0); // different slot, still zero
    }

    #[test]
    fn test_encode_count() {
        let mut slot = vec![0.0; D_MODEL];
        encode_count(&mut slot, 50, 100);
        assert_eq!(slot[0], 0.5); // normalized
        assert!(slot[1] > 0.0 && slot[1] < 1.0); // log-scaled
        assert_eq!(slot[2], 0.0); // not zero
        assert_eq!(slot[3], 0.0); // not at max
    }

    #[test]
    fn test_encode_count_zero() {
        let mut slot = vec![0.0; D_MODEL];
        encode_count(&mut slot, 0, 100);
        assert_eq!(slot[2], 1.0); // is_zero flag
    }

    #[test]
    fn test_encode_onehot() {
        let mut slot = vec![0.0; D_MODEL];
        encode_onehot(&mut slot, 2, 5, 10);
        assert_eq!(slot[10], 0.0);
        assert_eq!(slot[11], 0.0);
        assert_eq!(slot[12], 1.0); // selected
        assert_eq!(slot[13], 0.0);
        assert_eq!(slot[14], 0.0);
    }
}
