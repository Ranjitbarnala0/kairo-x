//! Output heads for the controller (§7.5).
//!
//! 6 output heads that decode the controller's d_model representation
//! into actionable decisions:
//!
//! 1. **Action head**: 34 discrete actions (softmax)
//! 2. **Context selection head**: per-candidate relevance scores (sigmoid)
//! 3. **Context budget head**: token count [512, 16384] (sigmoid scaled)
//! 4. **Enforcement intensity head**: float [-0.3, +0.3] (tanh scaled)
//! 5. **Session edge-case head**: continue/reset binary (sigmoid)
//! 6. **Stop head** (itch-gated): continue/escalate/retry/terminate (softmax with mask)

use super::weights::WeightStore;

// ---------------------------------------------------------------------------
// Head outputs
// ---------------------------------------------------------------------------

/// Combined output from all 6 controller heads.
#[derive(Debug, Clone)]
pub struct HeadOutputs {
    /// Action probabilities: 34 values summing to 1.0.
    pub action_probs: Vec<f32>,
    /// Selected action index (argmax of action_probs).
    pub action: u8,
    /// Context selection scores: per-candidate [0, 1] relevance.
    /// Length matches the number of candidates presented in input.
    pub context_scores: Vec<f32>,
    /// Context token budget: value in [512, 16384].
    pub context_budget: u32,
    /// Enforcement intensity adjustment: [-0.3, +0.3].
    pub enforcement_intensity_adj: f32,
    /// Session edge-case: probability of continuing the session.
    pub session_continue_prob: f32,
    /// Stop decision probabilities: [continue, escalate, retry, terminate].
    pub stop_probs: Vec<f32>,
    /// Selected stop action.
    pub stop_action: StopAction,
}

/// Stop head actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopAction {
    /// Keep working on the current task.
    Continue,
    /// Escalate to user (ask for guidance).
    Escalate,
    /// Retry the current node.
    Retry,
    /// Terminate the task (only if itch register allows).
    Terminate,
}

// ---------------------------------------------------------------------------
// Head computation
// ---------------------------------------------------------------------------

/// Compute all 6 output heads from the controller's final representation.
#[allow(clippy::needless_range_loop)]
pub fn compute_heads(
    representation: &[f32],  // [d_model = 288]
    weights: &WeightStore,
    n_context_candidates: usize,
    itch_active: bool,       // If true, mask the terminate action
) -> HeadOutputs {
    let d_model = representation.len();

    // -----------------------------------------------------------------------
    // Head 1: Action (34-way softmax)
    // -----------------------------------------------------------------------
    let action_w = weights.get_or_zeros("heads.action.weight", &[34, d_model]);
    let action_b = weights.get_or_zeros("heads.action.bias", &[34]);

    let mut action_logits = vec![0.0f32; 34];
    matvec_add(
        &action_w.data,
        &action_b.data,
        representation,
        &mut action_logits,
        34,
        d_model,
    );
    let action_probs = softmax(&action_logits);
    let action = argmax(&action_probs) as u8;

    // -----------------------------------------------------------------------
    // Head 2: Context selection (per-candidate sigmoid)
    // -----------------------------------------------------------------------
    // The context selection head uses a learned projection from d_model to 1
    // applied to each candidate's features (embedded in the input).
    // For simplicity, we project the representation to n_context_candidates scores.
    let context_scores = if n_context_candidates > 0 {
        // Use a simple dot-product with the representation's first N dims
        // as candidate scores. (Real implementation uses per-candidate features
        // from the input packet, scored by the learned head.)
        let ctx_w = weights.get_or_zeros(
            "heads.context_selection.weight",
            &[n_context_candidates.max(1), d_model],
        );
        let mut scores = vec![0.0f32; n_context_candidates];
        for i in 0..n_context_candidates {
            let start = i * d_model;
            if start + d_model <= ctx_w.data.len() {
                let mut sum = 0.0f32;
                for j in 0..d_model {
                    sum += ctx_w.data[start + j] * representation[j];
                }
                scores[i] = sigmoid_scalar(sum);
            }
        }
        scores
    } else {
        Vec::new()
    };

    // -----------------------------------------------------------------------
    // Head 3: Context budget ([512, 16384])
    // -----------------------------------------------------------------------
    let budget_w = weights.get_or_zeros("heads.context_budget.weight", &[1, d_model]);
    let budget_b = weights.get_or_zeros("heads.context_budget.bias", &[1]);

    let mut budget_raw = 0.0f32;
    for j in 0..d_model {
        budget_raw += budget_w.data[j] * representation[j];
    }
    budget_raw += budget_b.data[0];

    // Sigmoid scaled to [512, 16384], clamped to handle NaN/inf edge cases
    let budget_sigmoid = sigmoid_scalar(budget_raw);
    let budget_f32 = 512.0 + budget_sigmoid * (16384.0 - 512.0);
    let context_budget = if budget_f32.is_finite() {
        (budget_f32 as u32).clamp(512, 16384)
    } else {
        8192 // Safe default if NaN propagated through the network
    };

    // -----------------------------------------------------------------------
    // Head 4: Enforcement intensity adjustment ([-0.3, +0.3])
    // -----------------------------------------------------------------------
    let ei_w = weights.get_or_zeros("heads.enforcement_intensity.weight", &[1, d_model]);
    let ei_b = weights.get_or_zeros("heads.enforcement_intensity.bias", &[1]);

    let mut ei_raw = 0.0f32;
    for j in 0..d_model {
        ei_raw += ei_w.data[j] * representation[j];
    }
    ei_raw += ei_b.data[0];

    // Tanh scaled to [-0.3, +0.3]
    let enforcement_intensity_adj = ei_raw.tanh() * 0.3;

    // -----------------------------------------------------------------------
    // Head 5: Session edge-case (binary sigmoid)
    // -----------------------------------------------------------------------
    let se_w = weights.get_or_zeros("heads.session_edge.weight", &[1, d_model]);
    let se_b = weights.get_or_zeros("heads.session_edge.bias", &[1]);

    let mut se_raw = 0.0f32;
    for j in 0..d_model {
        se_raw += se_w.data[j] * representation[j];
    }
    se_raw += se_b.data[0];

    let session_continue_prob = sigmoid_scalar(se_raw);

    // -----------------------------------------------------------------------
    // Head 6: Stop (4-way softmax with itch masking)
    // -----------------------------------------------------------------------
    let stop_w = weights.get_or_zeros("heads.stop.weight", &[4, d_model]);
    let stop_b = weights.get_or_zeros("heads.stop.bias", &[4]);

    let mut stop_logits = vec![0.0f32; 4];
    matvec_add(
        &stop_w.data,
        &stop_b.data,
        representation,
        &mut stop_logits,
        4,
        d_model,
    );

    // ITCH GATE: if itch register has active bits, mask the terminate action
    // by setting its logit to -infinity (effectively zero probability after softmax)
    if itch_active {
        stop_logits[3] = f32::NEG_INFINITY;
    }

    let stop_probs = softmax(&stop_logits);
    let stop_idx = argmax(&stop_probs);
    let stop_action = match stop_idx {
        0 => StopAction::Continue,
        1 => StopAction::Escalate,
        2 => StopAction::Retry,
        3 => StopAction::Terminate,
        _ => StopAction::Continue,
    };

    HeadOutputs {
        action_probs,
        action,
        context_scores,
        context_budget,
        enforcement_intensity_adj,
        session_continue_prob,
        stop_probs,
        stop_action,
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::ControllerConfig;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let logits = vec![1.0, 2.0, 3.0, f32::NEG_INFINITY];
        let probs = softmax(&logits);
        assert!(probs[3].abs() < 1e-6); // Masked out
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_head_outputs_zero_weights() {
        let config = ControllerConfig::default();
        let weights = WeightStore::zeros(&config);
        let representation = vec![0.1f32; 288];

        let outputs = compute_heads(&representation, &weights, 5, true);

        // Action should be 34 probabilities summing to 1
        assert_eq!(outputs.action_probs.len(), 34);
        let sum: f32 = outputs.action_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Context scores should have 5 entries
        assert_eq!(outputs.context_scores.len(), 5);

        // Budget should be in [512, 16384]
        assert!(outputs.context_budget >= 512 && outputs.context_budget <= 16384);

        // Enforcement intensity should be in [-0.3, 0.3]
        assert!(outputs.enforcement_intensity_adj >= -0.3 && outputs.enforcement_intensity_adj <= 0.3);

        // Session continue prob should be in [0, 1]
        assert!(outputs.session_continue_prob >= 0.0 && outputs.session_continue_prob <= 1.0);

        // With itch active, terminate should be masked
        assert!(outputs.stop_probs[3].abs() < 1e-6);
        assert_ne!(outputs.stop_action, StopAction::Terminate);
    }

    #[test]
    fn test_itch_gate() {
        let config = ControllerConfig::default();
        let weights = WeightStore::zeros(&config);
        let representation = vec![0.0f32; 288];

        // Without itch: terminate is available
        let out_no_itch = compute_heads(&representation, &weights, 0, false);
        // Uniform probs with zero weights → terminate should have non-zero prob
        assert!(out_no_itch.stop_probs[3] > 0.0);

        // With itch: terminate is masked
        let out_with_itch = compute_heads(&representation, &weights, 0, true);
        assert!(out_with_itch.stop_probs[3].abs() < 1e-6);
    }
}
