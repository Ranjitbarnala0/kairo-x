//! Liquid-hybrid recurrent block — the core computational unit of the controller.
//!
//! Matches the Python `LiquidBlock` class exactly. Each of the 12 controller
//! layers is a liquid block that:
//!
//! 1. Pre-norm: LayerNorm(input)
//! 2. Input projection: d_model -> n_bands * d_state, split into per-band d_state
//! 3. Per-band GRU update (d_state input, d_state state)
//! 4. Cross-band attention with pre-norm (cross_band_norm before attn)
//! 5. Per-band FFN: LayerNorm -> Linear -> SiLU -> Linear (with residual)
//! 6. Band-to-model projection: n_bands * d_state -> d_model + residual
//! 7. SwiGLU FFN with pre-norm: ffn_norm -> gate/up -> gated multiply -> down + residual
//!
//! The "liquid" aspect comes from the continuous-time state dynamics:
//! state evolves through GRU gates that blend new information with
//! persistent memory, creating a liquid state machine.

use super::math::{matvec, matvec_add};
use super::weights::WeightStore;

// ---------------------------------------------------------------------------
// Math primitives (block-local activations; shared linear algebra is in math.rs)
// ---------------------------------------------------------------------------

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

/// Public entry point for layer normalization, used by inference.rs for output_norm.
pub fn apply_layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    layer_norm(x, gamma, beta, 1e-5);
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

/// Execute one liquid block layer, matching the Python `LiquidBlock.forward()`.
///
/// Data flow (matching Python exactly):
///   1. Pre-norm: input_norm(x)
///   2. Input projection: d_model -> n_bands * d_state, split into bands
///   3. Per-band GRU update (input is d_state, not d_model)
///   4. Cross-band attention with pre-norm (cross_band_norm before attn)
///   5. Per-band FFN with residual (LayerNorm -> Linear -> SiLU -> Linear)
///   6. Band-to-model projection: n_bands * d_state -> d_model, residual add
///   7. SwiGLU FFN with pre-norm: ffn_norm -> gate/up -> gated -> down, residual add
///
/// Returns the output (d_model).
#[allow(clippy::needless_range_loop)]
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

    // Save residual for the first residual connection (after band_to_model)
    let residual = input.to_vec();

    // -----------------------------------------------------------------------
    // Step 1: Pre-norm (input_norm) — LayerNorm BEFORE sublayer
    // Matches Python: x_normed = self.input_norm(x)
    // -----------------------------------------------------------------------
    let input_norm_w = weights.get_or_zeros(
        &format!("{prefix}.input_norm.weight"),
        &[d_model],
    );
    let input_norm_b = weights.get_or_zeros(
        &format!("{prefix}.input_norm.bias"),
        &[d_model],
    );
    let mut x_normed = input.to_vec();
    layer_norm(&mut x_normed, &input_norm_w.data, &input_norm_b.data, 1e-5);

    // -----------------------------------------------------------------------
    // Step 2: Input projection — d_model -> n_bands * d_state
    // Matches Python: band_inputs = self.input_proj(x_normed)
    // Then split into n_bands chunks of d_state each.
    // -----------------------------------------------------------------------
    let proj_out_dim = n_bands * d_state;
    let input_proj_w = weights.get_or_zeros(
        &format!("{prefix}.input_proj.weight"),
        &[proj_out_dim, d_model],
    );
    let input_proj_b = weights.get_or_zeros(
        &format!("{prefix}.input_proj.bias"),
        &[proj_out_dim],
    );
    let mut projected = vec![0.0f32; proj_out_dim];
    matvec_add(
        &input_proj_w.data,
        &input_proj_b.data,
        &x_normed,
        &mut projected,
        proj_out_dim,
        d_model,
    );

    // -----------------------------------------------------------------------
    // Step 3: Per-band GRU updates
    // Each band gets a d_state-sized slice from the projection (not raw d_model).
    // Matches Python: BandGRU(d_input=d_state, d_state=d_state)
    //   W_input: [3*d_state, d_state]
    //   W_state: [3*d_state, d_state]
    // -----------------------------------------------------------------------
    let mut band_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_bands);

    for band in 0..n_bands {
        let bp = format!("{prefix}.band_grus.{band}");
        let band_input = &projected[band * d_state..(band + 1) * d_state];

        let gate_dim = d_state * 3;

        // Python: W_input is Linear(d_input=d_state, 3*d_state)
        // gate_ih shape: [3*d_state, d_state] (input is d_state from projection)
        let gate_ih_w = weights.get_or_zeros(
            &format!("{bp}.W_input.weight"),
            &[gate_dim, d_state],
        );
        let gate_ih_b = weights.get_or_zeros(
            &format!("{bp}.W_input.bias"),
            &[gate_dim],
        );
        // Python: W_state is Linear(d_state, 3*d_state)
        let gate_hh_w = weights.get_or_zeros(
            &format!("{bp}.W_state.weight"),
            &[gate_dim, d_state],
        );
        let gate_hh_b = weights.get_or_zeros(
            &format!("{bp}.W_state.bias"),
            &[gate_dim],
        );

        let mut ih = vec![0.0f32; gate_dim];
        let mut hh = vec![0.0f32; gate_dim];

        // ih = W_input @ band_input + bias  (d_state input, not d_model)
        matvec_add(
            &gate_ih_w.data,
            &gate_ih_b.data,
            band_input,
            &mut ih,
            gate_dim,
            d_state,
        );
        // hh = W_state @ h_prev + bias
        matvec_add(
            &gate_hh_w.data,
            &gate_hh_b.data,
            &state.band_states[band],
            &mut hh,
            gate_dim,
            d_state,
        );

        // Python chunks into (x_z, x_r, x_n) and (h_z, h_r, h_n)
        // z = sigmoid(x_z + h_z)  -- update gate
        // r = sigmoid(x_r + h_r)  -- reset gate
        // n = tanh(x_n + r * h_n) -- candidate
        let (z_ih, rest) = ih.split_at_mut(d_state);
        let (r_ih, n_ih) = rest.split_at_mut(d_state);

        let (z_hh, rest_hh) = hh.split_at(d_state);
        let (r_hh, n_hh) = rest_hh.split_at(d_state);

        // z = σ(x_z + h_z)
        for i in 0..d_state {
            z_ih[i] += z_hh[i];
        }
        sigmoid(z_ih);

        // r = σ(x_r + h_r)
        for i in 0..d_state {
            r_ih[i] += r_hh[i];
        }
        sigmoid(r_ih);

        // n = tanh(x_n + r * h_n)
        let mut n_gate = vec![0.0f32; d_state];
        for i in 0..d_state {
            n_gate[i] = n_ih[i] + r_ih[i] * n_hh[i];
        }
        tanh_inplace(&mut n_gate);

        // h_new = (1 - z) * n + z * h_old
        let h_old = &state.band_states[band];
        let mut h_new = vec![0.0f32; d_state];
        for i in 0..d_state {
            h_new[i] = (1.0 - z_ih[i]) * n_gate[i] + z_ih[i] * h_old[i];
        }

        state.band_states[band] = h_new.clone();
        band_outputs.push(h_new);
    }

    // -----------------------------------------------------------------------
    // Step 4: Cross-band attention with pre-norm
    // Matches Python:
    //   stacked = torch.stack(band_outputs, dim=1)         # (B, n_bands, d_state)
    //   stacked_normed = self.cross_band_norm(stacked)      # pre-norm
    //   cross_attn_out = self.cross_band_attn(stacked_normed)
    //   stacked = stacked + dropout(cross_attn_out)         # residual
    //   # Also update states with cross-band info
    //   new_states[i] += dropout(cross_attn_out[:, i, :])
    // -----------------------------------------------------------------------

    let n_heads = 4usize;
    let d_head = d_state / n_heads; // 192/4 = 48

    // Pre-norm: apply cross_band_norm to each band independently
    let cross_norm_w = weights.get_or_zeros(
        &format!("{prefix}.cross_band_norm.weight"),
        &[d_state],
    );
    let cross_norm_b = weights.get_or_zeros(
        &format!("{prefix}.cross_band_norm.bias"),
        &[d_state],
    );
    let mut normed_bands: Vec<Vec<f32>> = Vec::with_capacity(n_bands);
    for band_out in &band_outputs {
        let mut normed = band_out.clone();
        layer_norm(&mut normed, &cross_norm_w.data, &cross_norm_b.data, 1e-5);
        normed_bands.push(normed);
    }

    // QKV projection: [3*d_state, d_state], no bias (Python has bias=False)
    let qkv_w = weights.get_or_zeros(
        &format!("{prefix}.cross_band_attn.qkv.weight"),
        &[3 * d_state, d_state],
    );
    let out_w = weights.get_or_zeros(
        &format!("{prefix}.cross_band_attn.out_proj.weight"),
        &[d_state, d_state],
    );

    let qkv_dim = 3 * d_state;
    let mut q_bands = vec![vec![0.0f32; d_state]; n_bands];
    let mut k_bands = vec![vec![0.0f32; d_state]; n_bands];
    let mut v_bands = vec![vec![0.0f32; d_state]; n_bands];

    for (band_idx, normed_band) in normed_bands.iter().enumerate() {
        let mut qkv = vec![0.0f32; qkv_dim];
        matvec(&qkv_w.data, normed_band, &mut qkv, qkv_dim, d_state);
        q_bands[band_idx] = qkv[..d_state].to_vec();
        k_bands[band_idx] = qkv[d_state..2 * d_state].to_vec();
        v_bands[band_idx] = qkv[2 * d_state..].to_vec();
    }

    // Multi-head attention over n_bands
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_state]; n_bands];

    for h in 0..n_heads {
        let head_start = h * d_head;
        let head_end = head_start + d_head;

        let mut scores = vec![vec![0.0f32; n_bands]; n_bands];
        for qi in 0..n_bands {
            for ki in 0..n_bands {
                let mut dot = 0.0f32;
                for d in head_start..head_end {
                    dot += q_bands[qi][d] * k_bands[ki][d];
                }
                scores[qi][ki] = dot * scale;
            }
            // Softmax over key dimension
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

    // Output projection per-band: out_proj(attn_output)
    let mut cross_attn_out: Vec<Vec<f32>> = Vec::with_capacity(n_bands);
    for band_attn in &attn_output {
        let mut proj = vec![0.0f32; d_state];
        matvec(&out_w.data, band_attn, &mut proj, d_state, d_state);
        cross_attn_out.push(proj);
    }

    // Residual: stacked = band_outputs + cross_attn_out (per band)
    // Also update carried states with cross-band info
    let mut stacked: Vec<Vec<f32>> = Vec::with_capacity(n_bands);
    for i in 0..n_bands {
        let mut band_with_attn = band_outputs[i].clone();
        vec_add(&mut band_with_attn, &cross_attn_out[i]);
        stacked.push(band_with_attn);

        // Update state: new_states[i] += cross_attn_out[:, i, :]
        vec_add(&mut state.band_states[i], &cross_attn_out[i]);
    }

    // -----------------------------------------------------------------------
    // Step 5: Per-band FFN with residual
    // Matches Python:
    //   band_ffns[i] = Sequential(LayerNorm, Linear(d_state, d_state*2), SiLU,
    //                             Dropout, Linear(d_state*2, d_state), Dropout)
    //   band_ffn_out = band_i + band_ffns[i](band_i)
    //   new_states[i] += band_ffns[i](new_states[i]) * band_ffn_state_scale
    // -----------------------------------------------------------------------
    let d_band_ffn = d_state * 2; // 192 * 2 = 384

    // Load the learned state scale parameter (scalar)
    let band_ffn_state_scale_t = weights.get_or_zeros(
        &format!("{prefix}.band_ffn_state_scale"),
        &[1],
    );
    let band_ffn_state_scale = band_ffn_state_scale_t.data[0];

    for i in 0..n_bands {
        let bp = format!("{prefix}.band_ffns.{i}");

        // Per-band FFN weights: LayerNorm -> Linear(d_state, d_state*2) -> SiLU -> Linear(d_state*2, d_state)
        let bffn_norm_w = weights.get_or_zeros(&format!("{bp}.0.weight"), &[d_state]);
        let bffn_norm_b = weights.get_or_zeros(&format!("{bp}.0.bias"), &[d_state]);
        let bffn_up_w = weights.get_or_zeros(&format!("{bp}.1.weight"), &[d_band_ffn, d_state]);
        let bffn_up_b = weights.get_or_zeros(&format!("{bp}.1.bias"), &[d_band_ffn]);
        let bffn_down_w = weights.get_or_zeros(&format!("{bp}.4.weight"), &[d_state, d_band_ffn]);
        let bffn_down_b = weights.get_or_zeros(&format!("{bp}.4.bias"), &[d_state]);

        // Apply per-band FFN to stacked band
        let band_ffn_out = apply_band_ffn(
            &stacked[i],
            &bffn_norm_w.data,
            &bffn_norm_b.data,
            &bffn_up_w.data,
            &bffn_up_b.data,
            &bffn_down_w.data,
            &bffn_down_b.data,
            d_state,
            d_band_ffn,
        );

        // Residual: stacked[i] = stacked[i] + band_ffn_out
        vec_add(&mut stacked[i], &band_ffn_out);

        // Also update carried state: new_states[i] += band_ffn(state[i]) * scale
        let state_ffn_out = apply_band_ffn(
            &state.band_states[i],
            &bffn_norm_w.data,
            &bffn_norm_b.data,
            &bffn_up_w.data,
            &bffn_up_b.data,
            &bffn_down_w.data,
            &bffn_down_b.data,
            d_state,
            d_band_ffn,
        );
        for j in 0..d_state {
            state.band_states[i][j] += state_ffn_out[j] * band_ffn_state_scale;
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Project back to d_model — band_to_model
    // Matches Python: x_model = self.band_to_model(flat)
    //   where flat = stacked.reshape(B, n_bands * d_state)
    //   band_to_model: Linear(n_bands * d_state, d_model)
    // Then first residual: x_model = residual + dropout(x_model)
    // -----------------------------------------------------------------------
    let flat_dim = n_bands * d_state;
    let mut flat = vec![0.0f32; flat_dim];
    for i in 0..n_bands {
        flat[i * d_state..(i + 1) * d_state].copy_from_slice(&stacked[i]);
    }

    let b2m_w = weights.get_or_zeros(
        &format!("{prefix}.band_to_model.weight"),
        &[d_model, flat_dim],
    );
    let b2m_b = weights.get_or_zeros(
        &format!("{prefix}.band_to_model.bias"),
        &[d_model],
    );
    let mut x_model = vec![0.0f32; d_model];
    matvec_add(&b2m_w.data, &b2m_b.data, &flat, &mut x_model, d_model, flat_dim);

    // First residual connection: x_model = residual + x_model
    for i in 0..d_model {
        x_model[i] += residual[i];
    }

    // -----------------------------------------------------------------------
    // Step 7: SwiGLU FFN with pre-norm
    // Matches Python:
    //   ffn_residual = x_model
    //   x_normed = self.ffn_norm(x_model)
    //   gate = F.silu(self.ffn_gate(x_normed))
    //   up = self.ffn_up(x_normed)
    //   x_model = ffn_residual + self.ffn_dropout(self.ffn_down(gate * up))
    //
    // ffn_gate: Linear(d_model, d_ffn, bias=False)
    // ffn_up:   Linear(d_model, d_ffn, bias=False)
    // ffn_down: Linear(d_ffn, d_model, bias=False)
    // -----------------------------------------------------------------------
    let ffn_residual = x_model.clone();

    // Pre-norm
    let ffn_norm_w = weights.get_or_zeros(
        &format!("{prefix}.ffn_norm.weight"),
        &[d_model],
    );
    let ffn_norm_b = weights.get_or_zeros(
        &format!("{prefix}.ffn_norm.bias"),
        &[d_model],
    );
    layer_norm(&mut x_model, &ffn_norm_w.data, &ffn_norm_b.data, 1e-5);

    // SwiGLU: gate = silu(W_gate @ x), up = W_up @ x, out = W_down @ (gate * up)
    // All three are bias=False in Python
    let ffn_gate_w = weights.get_or_zeros(
        &format!("{prefix}.ffn_gate.weight"),
        &[d_ffn, d_model],
    );
    let ffn_up_w = weights.get_or_zeros(
        &format!("{prefix}.ffn_up.weight"),
        &[d_ffn, d_model],
    );
    let ffn_down_w = weights.get_or_zeros(
        &format!("{prefix}.ffn_down.weight"),
        &[d_model, d_ffn],
    );

    let mut gate = vec![0.0f32; d_ffn];
    matvec(&ffn_gate_w.data, &x_model, &mut gate, d_ffn, d_model);
    silu(&mut gate);

    let mut up = vec![0.0f32; d_ffn];
    matvec(&ffn_up_w.data, &x_model, &mut up, d_ffn, d_model);

    // Gated: gate * up
    let mut gated = vec![0.0f32; d_ffn];
    for i in 0..d_ffn {
        gated[i] = gate[i] * up[i];
    }

    // Down projection
    let mut ffn_out = vec![0.0f32; d_model];
    matvec(&ffn_down_w.data, &gated, &mut ffn_out, d_model, d_ffn);

    // Second residual connection: output = ffn_residual + ffn_out
    for i in 0..d_model {
        ffn_out[i] += ffn_residual[i];
    }

    ffn_out
}

/// Apply a per-band FFN: LayerNorm -> Linear(d_state, d_band_ffn) -> SiLU -> Linear(d_band_ffn, d_state)
///
/// Matches Python Sequential(LayerNorm(d_state), Linear(d_state, d_state*2),
///     SiLU(), Dropout(), Linear(d_state*2, d_state), Dropout())
/// Dropout is a no-op at inference time.
#[allow(clippy::too_many_arguments)]
fn apply_band_ffn(
    input: &[f32],
    norm_w: &[f32],
    norm_b: &[f32],
    up_w: &[f32],
    up_b: &[f32],
    down_w: &[f32],
    down_b: &[f32],
    d_state: usize,
    d_band_ffn: usize,
) -> Vec<f32> {
    // LayerNorm
    let mut normed = input.to_vec();
    layer_norm(&mut normed, norm_w, norm_b, 1e-5);

    // Linear up: d_state -> d_band_ffn
    let mut hidden = vec![0.0f32; d_band_ffn];
    matvec_add(up_w, up_b, &normed, &mut hidden, d_band_ffn, d_state);

    // SiLU
    silu(&mut hidden);

    // Linear down: d_band_ffn -> d_state
    let mut out = vec![0.0f32; d_state];
    matvec_add(down_w, down_b, &hidden, &mut out, d_state, d_band_ffn);

    out
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
