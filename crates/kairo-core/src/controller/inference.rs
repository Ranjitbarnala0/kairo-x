//! Controller inference path (§7.6).
//!
//! Runs the full forward pass: input projection → 12 liquid blocks → output heads.
//! This is the only public entry point for the controller at inference time.

use super::heads::{self, HeadOutputs};
use super::input_assembly::{InputPacket, D_MODEL, N_INPUT_SLOTS};
use super::liquid_block::{self, LiquidBlockConfig, LiquidBlockState};
use super::weights::WeightStore;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the neural controller.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Number of input slots.
    pub n_input_slots: usize,
    /// Model dimension.
    pub d_model: usize,
    /// State dimension per band.
    pub d_state: usize,
    /// Total state dimension (d_state * n_bands).
    pub d_state_total: usize,
    /// Number of state bands per layer.
    pub n_bands: usize,
    /// Feed-forward inner dimension.
    pub d_ffn: usize,
    /// Number of liquid layers.
    pub n_layers: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            n_input_slots: N_INPUT_SLOTS,
            d_model: D_MODEL,
            d_state: 192,
            d_state_total: 768, // 4 * 192
            n_bands: 4,
            d_ffn: 1152,
            n_layers: 12,
        }
    }
}

// ---------------------------------------------------------------------------
// Controller output
// ---------------------------------------------------------------------------

/// The actionable output of one controller step.
#[derive(Debug, Clone)]
pub struct ControllerOutput {
    /// Combined head outputs (all 6 heads).
    pub heads: HeadOutputs,
    /// The d_model representation from the final layer (for diagnostics).
    pub final_representation: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/// The neural controller holding recurrent state and weights.
pub struct Controller {
    config: ControllerConfig,
    weights: WeightStore,
    /// Per-layer recurrent state (persists across steps).
    layer_states: Vec<LiquidBlockState>,
}

impl Controller {
    /// Create a new controller with the given config and weight store.
    pub fn new(config: ControllerConfig, weights: WeightStore) -> Self {
        let layer_states = (0..config.n_layers)
            .map(|_| LiquidBlockState::new(config.n_bands, config.d_state))
            .collect();
        Self {
            config,
            weights,
            layer_states,
        }
    }

    /// Create a controller with zero-initialized weights (for testing or before
    /// the neural network is trained; rule-based systems provide sensible defaults).
    pub fn zeros() -> Self {
        let config = ControllerConfig::default();
        let weights = WeightStore::zeros(&config);
        Self::new(config, weights)
    }

    /// Run one forward step of the controller.
    ///
    /// `packet` is the assembled 32-slot input.
    /// `n_context_candidates` is the number of context candidates presented.
    /// `itch_active` indicates whether the itch register has active bits.
    #[allow(clippy::needless_range_loop)]
    pub fn step(
        &mut self,
        packet: &InputPacket,
        n_context_candidates: usize,
        itch_active: bool,
    ) -> ControllerOutput {
        // Input projection: flatten N_INPUT_SLOTS * D_MODEL → d_model
        let input_proj_w = self.weights.get_or_zeros(
            "input_proj.weight",
            &[self.config.d_model, self.config.n_input_slots * self.config.d_model],
        );
        let input_proj_b = self.weights.get_or_zeros(
            "input_proj.bias",
            &[self.config.d_model],
        );

        let mut projected = vec![0.0f32; self.config.d_model];
        let in_dim = self.config.n_input_slots * self.config.d_model;
        for i in 0..self.config.d_model {
            let mut sum = input_proj_b.data.get(i).copied().unwrap_or(0.0);
            let row_start = i * in_dim;
            for j in 0..in_dim.min(packet.data.len()) {
                if row_start + j < input_proj_w.data.len() {
                    sum += input_proj_w.data[row_start + j] * packet.data[j];
                }
            }
            projected[i] = sum;
        }

        // Run through liquid layers
        let block_config = LiquidBlockConfig {
            d_model: self.config.d_model,
            d_state: self.config.d_state,
            n_bands: self.config.n_bands,
            d_ffn: self.config.d_ffn,
        };

        let mut current = projected;
        for layer_idx in 0..self.config.n_layers {
            current = liquid_block::liquid_block_forward(
                &current,
                &mut self.layer_states[layer_idx],
                &self.weights,
                layer_idx,
                &block_config,
            );
        }

        // Compute output heads from final representation
        let head_outputs = heads::compute_heads(
            &current,
            &self.weights,
            n_context_candidates,
            itch_active,
        );

        ControllerOutput {
            heads: head_outputs,
            final_representation: current,
        }
    }

    /// Reset all recurrent state (e.g., at task boundary).
    pub fn reset_state(&mut self) {
        for state in &mut self.layer_states {
            state.reset();
        }
    }

    /// Serialize the recurrent state to a flat f32 vector for checkpointing.
    ///
    /// Layout: for each layer, for each band, the d_state floats are appended.
    /// Total length = n_layers * n_bands * d_state.
    pub fn serialize_state(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.config.n_layers * self.config.n_bands * self.config.d_state,
        );
        for state in &self.layer_states {
            for band in &state.band_states {
                out.extend_from_slice(band);
            }
        }
        out
    }

    /// Restore recurrent state from a flat f32 vector (produced by
    /// [`serialize_state`]).
    ///
    /// If the vector length does not match the expected size, the state is
    /// reset to zeros and the method returns `false`.
    pub fn deserialize_state(&mut self, data: &[f32]) -> bool {
        let expected = self.config.n_layers * self.config.n_bands * self.config.d_state;
        if data.len() != expected {
            tracing::warn!(
                expected,
                actual = data.len(),
                "Controller state vector size mismatch, resetting to zeros"
            );
            self.reset_state();
            return false;
        }

        let mut offset = 0;
        for state in &mut self.layer_states {
            for band in &mut state.band_states {
                let end = offset + self.config.d_state;
                band.copy_from_slice(&data[offset..end]);
                offset = end;
            }
        }
        true
    }

    /// Get the controller configuration.
    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    /// Total number of parameters in the weight store.
    pub fn total_parameters(&self) -> usize {
        self.weights.total_parameters()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_zeros() {
        let mut controller = Controller::zeros();
        let packet = InputPacket::new();

        let output = controller.step(&packet, 5, true);

        // Should produce valid head outputs
        assert_eq!(output.heads.action_probs.len(), 34);
        assert_eq!(output.final_representation.len(), D_MODEL);
    }

    #[test]
    fn test_controller_reset_state() {
        let mut controller = Controller::zeros();
        let packet = InputPacket::new();

        // Run a step to populate state
        controller.step(&packet, 0, false);

        // Reset and verify no panic
        controller.reset_state();

        // Run again after reset
        let output = controller.step(&packet, 0, false);
        assert_eq!(output.final_representation.len(), D_MODEL);
    }

    #[test]
    fn test_controller_config_default() {
        let config = ControllerConfig::default();
        assert_eq!(config.n_input_slots, 32);
        assert_eq!(config.d_model, 288);
        assert_eq!(config.n_layers, 12);
        assert_eq!(config.n_bands, 4);
        assert_eq!(config.d_state, 192);
        assert_eq!(config.d_state_total, 768);
    }
}
