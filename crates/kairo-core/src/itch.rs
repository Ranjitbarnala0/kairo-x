//! Dynamic itch register (§Flaw 9).
//!
//! The itch register is an architectural gate on the stop head: the agent
//! CANNOT terminate while any itch bit is set. Each graph node has one bit.
//! The register grows dynamically — no fixed size limit.
//!
//! Active count is maintained incrementally for O(1) `any_active()` checks.

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

/// Dynamic bit-vector itch register with O(1) active check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItchRegister {
    /// Dynamic-length bit vector (one bit per potential node).
    bits: BitVec<u64, Lsb0>,
    /// Number of set (active) bits — maintained incrementally.
    count: u32,
}

impl ItchRegister {
    /// Create a new empty itch register.
    pub fn new() -> Self {
        Self {
            bits: BitVec::new(),
            count: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: BitVec::with_capacity(capacity),
            count: 0,
        }
    }

    /// Set an itch bit (mark a node as needing resolution).
    ///
    /// Auto-resizes the bit vector if `idx` is beyond current length.
    pub fn set(&mut self, idx: usize) {
        if idx >= self.bits.len() {
            self.bits.resize(idx + 1, false);
        }
        if !self.bits[idx] {
            self.count = self.count.saturating_add(1);
            self.bits.set(idx, true);
        }
    }

    /// Clear an itch bit (mark a node as resolved).
    pub fn clear(&mut self, idx: usize) {
        if idx < self.bits.len() && self.bits[idx] {
            self.count = self.count.saturating_sub(1);
            self.bits.set(idx, false);
        }
    }

    /// Check whether any itch bit is active — O(1) via maintained count.
    ///
    /// This is THE gate check on the stop head. If this returns true,
    /// the agent cannot terminate.
    #[inline]
    pub fn any_active(&self) -> bool {
        self.count > 0
    }

    /// Number of active (set) itch bits — O(1).
    #[inline]
    pub fn active_count(&self) -> u32 {
        self.count
    }

    /// Check whether a specific bit is set.
    pub fn is_active(&self, idx: usize) -> bool {
        idx < self.bits.len() && self.bits[idx]
    }

    /// Total capacity of the register.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the register has no bits allocated.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Iterate over all active (set) bit indices.
    pub fn active_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter_ones()
    }

    /// Clear all bits and reset count.
    pub fn reset(&mut self) {
        self.bits.fill(false);
        self.count = 0;
    }

    /// Raw bytes for serialization.
    pub fn as_raw_slice(&self) -> &[u64] {
        self.bits.as_raw_slice()
    }

    /// Reconstruct from raw bytes and recompute count.
    pub fn from_raw(raw: Vec<u64>, bit_len: usize) -> Self {
        let bits = BitVec::from_vec(raw);
        let mut register = Self {
            bits,
            count: 0,
        };
        // Ensure correct length
        register.bits.resize(bit_len, false);
        // Recompute count from bits — safe cast with debug assertion
        let ones = register.bits.count_ones();
        debug_assert!(
            ones <= u32::MAX as usize,
            "itch register has more than u32::MAX active bits: {}",
            ones
        );
        register.count = ones.min(u32::MAX as usize) as u32;
        register
    }
}

impl Default for ItchRegister {
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
    fn test_basic_set_clear() {
        let mut itch = ItchRegister::new();

        assert!(!itch.any_active());
        assert_eq!(itch.active_count(), 0);

        itch.set(0);
        assert!(itch.any_active());
        assert_eq!(itch.active_count(), 1);
        assert!(itch.is_active(0));

        itch.set(5);
        assert_eq!(itch.active_count(), 2);

        itch.clear(0);
        assert_eq!(itch.active_count(), 1);
        assert!(!itch.is_active(0));
        assert!(itch.is_active(5));

        itch.clear(5);
        assert!(!itch.any_active());
    }

    #[test]
    fn test_idempotent_set() {
        let mut itch = ItchRegister::new();
        itch.set(3);
        itch.set(3);
        itch.set(3);
        assert_eq!(itch.active_count(), 1); // Only counted once
    }

    #[test]
    fn test_idempotent_clear() {
        let mut itch = ItchRegister::new();
        itch.set(3);
        itch.clear(3);
        itch.clear(3); // Should not underflow
        assert_eq!(itch.active_count(), 0);
    }

    #[test]
    fn test_clear_unset_bit() {
        let mut itch = ItchRegister::new();
        itch.clear(100); // Clear a bit that was never set — should be safe
        assert_eq!(itch.active_count(), 0);
    }

    #[test]
    fn test_auto_resize() {
        let mut itch = ItchRegister::new();
        itch.set(1000);
        assert!(itch.is_active(1000));
        assert!(!itch.is_active(999));
        assert_eq!(itch.active_count(), 1);
        assert!(itch.len() >= 1001);
    }

    #[test]
    fn test_active_indices() {
        let mut itch = ItchRegister::new();
        itch.set(2);
        itch.set(7);
        itch.set(15);

        let indices: Vec<usize> = itch.active_indices().collect();
        assert_eq!(indices, vec![2, 7, 15]);
    }

    #[test]
    fn test_reset() {
        let mut itch = ItchRegister::new();
        itch.set(1);
        itch.set(5);
        itch.set(10);
        assert_eq!(itch.active_count(), 3);

        itch.reset();
        assert!(!itch.any_active());
        assert_eq!(itch.active_count(), 0);
    }

    #[test]
    fn test_round_trip_serialization() {
        let mut itch = ItchRegister::new();
        itch.set(3);
        itch.set(42);
        itch.set(100);

        let raw = itch.as_raw_slice().to_vec();
        let bit_len = itch.len();

        let restored = ItchRegister::from_raw(raw, bit_len);
        assert_eq!(restored.active_count(), 3);
        assert!(restored.is_active(3));
        assert!(restored.is_active(42));
        assert!(restored.is_active(100));
        assert!(!restored.is_active(50));
    }

    #[test]
    fn test_large_register() {
        let mut itch = ItchRegister::new();
        for i in 0..10_000 {
            itch.set(i);
        }
        assert_eq!(itch.active_count(), 10_000);

        for i in 0..5_000 {
            itch.clear(i);
        }
        assert_eq!(itch.active_count(), 5_000);
        assert!(itch.any_active());
    }
}
