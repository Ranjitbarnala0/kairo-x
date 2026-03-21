//! Liquid-hybrid recurrent block — the core computational unit of the controller.
//!
//! Each of the 12 controller layers is a liquid block that:
//! 1. Projects input to d_model
//! 2. Splits into 4 band projections
//! 3. Each band: GRU-style gating with persistent state update
//! 4. Cross-band lightweight attention
//! 5. FFN with SiLU activation
//! 6. Residual connection + LayerNorm
//!
//! The "liquid" aspect comes from the continuous-time state dynamics:
//! state evolves through GRU gates that blend new information with
//! persistent memory, creating a liquid state machine.

use super::weights::WeightStore;

// ---------------------------------------------------------------------------
// Math primitives (no external linear algebra dependency)
// ---------------------------------------------------------------------------

/// Matrix-vector multiplication: y = W @ x + b
/// W is [out_dim x in_dim], x is [in_dim], y is [out_dim]
#[allow(clippy::needless_range_loop)]
fn matvec_add(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let mut sum = b.get(i).copied().unwrap_or(0.0);
        let row_start = i * in_dim;
        for j in 0..in_dim {
            if row_start + j < w.len() {
                sum += w[row_start + j] * x[j];
            }
        }
        out[i] = sum;
    }
}

/// Matrix-vector multiplication without bias: y = W @ x
#[allow(clippy::needless_range_loop)]
fn matvec(w: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        let row_start = i * in_dim;
        for j in 0..in_dim {
            if row_start + j < w.len() {
                sum += w[row_start + j] * x[j];
            }
        }
        out[i] = sum;
    }
}

/// Element-wise sigmoid: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Element-wise tanh
fn tanh_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.tanh();
    }
}

/// SiLU activation: x * σ(x)
fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Layer normalization: x = (x - mean) / sqrt(var + eps) * gamma + beta
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean: f32 = x.iter().sum::<f32>() / n as f32;

    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = 1.0 / (var + eps).sqrt();

    for (i, val) in x.iter_mut().enumerate() {
        let g = gamma.get(i).copied().unwrap_or(1.0);
        let b = beta.get(i).copied().unwrap_or(0.0);
        *val = (*val - mean) * std_inv * g + b;
    }
}

/// Vector addition: out[i] += x[i]
fn vec_add(out: &mut [f32], x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o += v;
    }
}

// ---------------------------------------------------------------------------
// Liquid block state
// ---------------------------------------------------------------------------

/// Persistent state for one liquid block (one per layer).
///
/// Contains the 4 band states that persist across controller steps.
pub struct LiquidBlockState {
    /// Band states: [n_bands][d_state]
    pub band_states: Vec<Vec<f32>>,
}

impl LiquidBlockState {
    pub fn new(n_bands: usize, d_state: usize) -> Self {
        Self {
            band_states: (0..n_bands).map(|_| vec![0.0; d_state]).collect(),
        }
    }

    /// Reset all band states to zero.
    pub fn reset(&mut self) {
        for state in &mut self.band_states {
            state.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Liquid block forward pass
// ---------------------------------------------------------------------------

/// Execute one liquid block layer.
///
/// Takes the current input (d_model), updates the 4 band states via GRU gating,
/// applies cross-band attention, FFN, residual + layer norm.
///
/// Returns the output (d_model).
pub fn liquid_block_forward(
    input: &[f32],           // [d_model = 288]
    state: &mut LiquidBlockState,
    weights: &WeightStore,
    layer_idx: usize,
    config: &LiquidBlockConfig,
) -> Vec<f32> {
    let prefix = format!("layers.{layer_idx}");
    let d_model = config.d_model;
    let d_state = config.d_state;
    let n_bands = config.n_bands;
    let d_ffn = config.d_ffn;

    // -----------------------------------------------------------------------
    // Step 1: GRU-style band updates
    // -----------------------------------------------------------------------
    // For each band, compute GRU gates from input and previous state,
    // then update the band state.

    let mut band_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_bands);

    for band in 0..n_bands {
        let bp = format!("{prefix}.band_{band}");

        // GRU: gates = σ(W_ih @ input + W_hh @ state)
        // Standard GRU has 3 gates: reset (r), update (z), new (n)
        let gate_dim = d_state * 3;

        let gate_ih_w = weights.get_or_zeros(&format!("{bp}.gate_ih.weight"), &[gate_dim, d_model]);
        let gate_ih_b = weights.get_or_zeros(&format!("{bp}.gate_ih.bias"), &[gate_dim]);
        let gate_hh_w = weights.get_or_zeros(&format!("{bp}.gate_hh.weight"), &[gate_dim, d_state]);
        let gate_hh_b = weights.get_or_zeros(&format!("{bp}.gate_hh.bias"), &[gate_dim]);

        let mut ih = vec![0.0f32; gate_dim];
        let mut hh = vec![0.0f32; gate_dim];

        matvec_add(&gate_ih_w.data, &gate_ih_b.data, input, &mut ih, gate_dim, d_model);
        matvec_add(&gate_hh_w.data, &gate_hh_b.data, &state.band_states[band], &mut hh, gate_dim, d_state);

        // Split into r, z, n gates
        let (r_gate, rest) = ih.split_at_mut(d_state);
        let (z_gate, n_ih) = rest.split_at_mut(d_state);

        let (r_hh, rest_hh) = hh.split_at(d_state);
        let (z_hh, n_hh) = rest_hh.split_at(d_state);

        // r = σ(r_ih + r_hh)
        for i in 0..d_state {
            r_gate[i] += r_hh[i];
        }
        sigmoid(r_gate);

        // z = σ(z_ih + z_hh)
        for i in 0..d_state {
            z_gate[i] += z_hh[i];
        }
        sigmoid(z_gate);

        // n = tanh(n_ih + r * n_hh)
        let mut n_gate = vec![0.0f32; d_state];
        for i in 0..d_state {
            n_gate[i] = n_ih[i] + r_gate[i] * n_hh[i];
        }
        tanh_inplace(&mut n_gate);

        // h_new = (1 - z) * n + z * h_old
        let h_old = &state.band_states[band];
        let mut h_new = vec![0.0f32; d_state];
        for i in 0..d_state {
            h_new[i] = (1.0 - z_gate[i]) * n_gate[i] + z_gate[i] * h_old[i];
        }

        // Update state
        state.band_states[band] = h_new.clone();
        band_outputs.push(h_new);
    }

    // -----------------------------------------------------------------------
    // Step 2: Cross-band attention (multi-head, matching Python training)
    // -----------------------------------------------------------------------
    // The Python CrossBandAttention operates on (batch, n_bands, d_state):
    //   1. QKV projection: Linear(d_state, 3*d_state) applied per-band
    //   2. Reshape to (batch, n_heads, n_bands, d_head) where n_heads=4, d_head=d_state/4
    //   3. Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_head)) @ V
    //   4. Output projection: Linear(d_state, d_state) applied per-band
    //
    // This lets bands exchange information through attention, not just
    // a gated scalar projection.

    let n_heads = 4usize;
    let d_head = d_state / n_heads; // 192/4 = 48

    // QKV weight: [3*d_state, d_state] — projects each band's d_state to Q,K,V
    let qkv_w = weights.get_or_zeros(
        &format!("{prefix}.cross_band.qkv.weight"),
        &[3 * d_state, d_state],
    );
    let out_w = weights.get_or_zeros(
        &format!("{prefix}.cross_band.out_proj.weight"),
        &[d_state, d_state],
    );

    // Project each band to Q, K, V (each d_state)
    let qkv_dim = 3 * d_state;
    let mut q_bands = vec![vec![0.0f32; d_state]; n_bands]; // [n_bands][d_state]
    let mut k_bands = vec![vec![0.0f32; d_state]; n_bands];
    let mut v_bands = vec![vec![0.0f32; d_state]; n_bands];

    for (band_idx, band_out) in band_outputs.iter().enumerate() {
        let mut qkv = vec![0.0f32; qkv_dim];
        matvec(&qkv_w.data, band_out, &mut qkv, qkv_dim, d_state);
        q_bands[band_idx] = qkv[..d_state].to_vec();
        k_bands[band_idx] = qkv[d_state..2 * d_state].to_vec();
        v_bands[band_idx] = qkv[2 * d_state..].to_vec();
    }

    // Multi-head attention: for each head, compute attention over n_bands
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_state]; n_bands];

    for h in 0..n_heads {
        let head_start = h * d_head;
        let head_end = head_start + d_head;

        // Compute NxN attention scores for this head
        let mut scores = vec![vec![0.0f32; n_bands]; n_bands]; // [query_band][key_band]
        for qi in 0..n_bands {
            for ki in 0..n_bands {
                let mut dot = 0.0f32;
                for d in head_start..head_end {
                    dot += q_bands[qi][d] * k_bands[ki][d];
                }
                scores[qi][ki] = dot * scale;
            }
            // Softmax over key dimension for this query
            let max_score = scores[qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in scores[qi].iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            if exp_sum > 0.0 {
                for s in scores[qi].iter_mut() {
                    *s /= exp_sum;
                }
            }
        }

        // Apply attention weights to V for this head
        for qi in 0..n_bands {
            for d in head_start..head_end {
                let mut weighted = 0.0f32;
                for ki in 0..n_bands {
                    weighted += scores[qi][ki] * v_bands[ki][d];
                }
                attn_output[qi][d] = weighted;
            }
        }
    }

    // Output projection per-band, then average to d_model
    // (Python model projects back per-band, then the residual is added to input)
    // We sum the projected band outputs to produce the d_model residual.
    let mut cross_band_out = vec![0.0f32; d_model];
    let _d_state_total = d_state * n_bands;

    // Project each band's attention output and accumulate
    for band_out in &attn_output {
        let mut proj = vec![0.0f32; d_state];
        matvec(&out_w.data, band_out, &mut proj, d_state, d_state);

        // Map from d_state (192) back into d_model (288) space
        // by placing each band's projection into its section of d_model
        // Use a simple averaging: accumulate and divide by n_bands
        for d in 0..d_state.min(d_model) {
            cross_band_out[d] += proj[d] / n_bands as f32;
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Residual connection + LayerNorm 1
    // -----------------------------------------------------------------------
    let mut residual = input.to_vec();
    vec_add(&mut residual, &cross_band_out);

    let norm1_w = weights.get_or_zeros(&format!("{prefix}.norm1.weight"), &[d_model]);
    let norm1_b = weights.get_or_zeros(&format!("{prefix}.norm1.bias"), &[d_model]);
    layer_norm(&mut residual, &norm1_w.data, &norm1_b.data, 1e-5);

    // -----------------------------------------------------------------------
    // Step 4: FFN with SiLU activation
    // -----------------------------------------------------------------------
    let ffn_up_w = weights.get_or_zeros(&format!("{prefix}.ffn.up.weight"), &[d_ffn, d_model]);
    let ffn_up_b = weights.get_or_zeros(&format!("{prefix}.ffn.up.bias"), &[d_ffn]);
    let ffn_down_w = weights.get_or_zeros(&format!("{prefix}.ffn.down.weight"), &[d_model, d_ffn]);
    let ffn_down_b = weights.get_or_zeros(&format!("{prefix}.ffn.down.bias"), &[d_model]);

    let mut ffn_hidden = vec![0.0f32; d_ffn];
    matvec_add(&ffn_up_w.data, &ffn_up_b.data, &residual, &mut ffn_hidden, d_ffn, d_model);
    silu(&mut ffn_hidden);

    let mut ffn_out = vec![0.0f32; d_model];
    matvec_add(&ffn_down_w.data, &ffn_down_b.data, &ffn_hidden, &mut ffn_out, d_model, d_ffn);

    // -----------------------------------------------------------------------
    // Step 5: Residual connection + LayerNorm 2
    // -----------------------------------------------------------------------
    vec_add(&mut ffn_out, &residual);

    let norm2_w = weights.get_or_zeros(&format!("{prefix}.norm2.weight"), &[d_model]);
    let norm2_b = weights.get_or_zeros(&format!("{prefix}.norm2.bias"), &[d_model]);
    layer_norm(&mut ffn_out, &norm2_w.data, &norm2_b.data, 1e-5);

    ffn_out
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the liquid block.
#[derive(Debug, Clone)]
pub struct LiquidBlockConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub n_bands: usize,
    pub d_ffn: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut x = vec![0.0, 1.0, -1.0, 100.0, -100.0];
        sigmoid(&mut x);
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!(x[1] > 0.7 && x[1] < 0.8);
        assert!(x[2] > 0.2 && x[2] < 0.3);
        assert!((x[3] - 1.0).abs() < 1e-6);
        assert!(x[4].abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 1.0, -1.0];
        silu(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6); // 0 * σ(0) = 0
        assert!(x[1] > 0.7); // 1 * σ(1) ≈ 0.731
        assert!(x[2] < 0.0); // -1 * σ(-1) ≈ -0.269
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        layer_norm(&mut x, &gamma, &beta, 1e-5);

        // After layer norm with unit gamma and zero beta,
        // mean should be ~0 and std should be ~1
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_matvec_add() {
        // Simple 2x3 matrix times 3-vector plus bias
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity-ish
        let b = vec![10.0, 20.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        matvec_add(&w, &b, &x, &mut y, 2, 3);
        assert_eq!(y[0], 11.0); // 1*1 + 0*2 + 0*3 + 10
        assert_eq!(y[1], 22.0); // 0*1 + 1*2 + 0*3 + 20
    }

    #[test]
    fn test_liquid_block_forward_zero_weights() {
        let config = LiquidBlockConfig {
            d_model: 288,
            d_state: 192,
            n_bands: 4,
            d_ffn: 1152,
        };

        let weights = WeightStore::zeros(&super::super::ControllerConfig::default());
        let mut state = LiquidBlockState::new(4, 192);
        let input = vec![0.1f32; 288];

        // Should not panic with zero weights
        let output = liquid_block_forward(&input, &mut state, &weights, 0, &config);
        assert_eq!(output.len(), 288);

        // With zero weights, output should be the layer-normed input
        // (since all projections produce zeros, residual is just the input)
    }
}
