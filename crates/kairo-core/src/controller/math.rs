//! Shared math primitives for the controller neural network.
//!
//! These low-level linear algebra routines are used by both the liquid block
//! layers and the output heads. No external BLAS dependency — everything is
//! pure Rust for deterministic, portable inference.

/// Matrix-vector multiplication with bias: y = W @ x + b
///
/// `w` is row-major `[out_dim x in_dim]`, `x` is `[in_dim]`,
/// `b` is `[out_dim]`, `out` is `[out_dim]`.
///
/// Safely handles cases where `w` or `b` may be shorter than expected
/// (e.g. from `get_or_zeros` returning a mismatched fallback).
#[allow(clippy::needless_range_loop)]
pub(crate) fn matvec_add(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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
///
/// `w` is row-major `[out_dim x in_dim]`, `x` is `[in_dim]`,
/// `out` is `[out_dim]`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn matvec(
    w: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_add() {
        // 2x3 matrix times 3-vector plus bias
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = vec![10.0, 20.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        matvec_add(&w, &b, &x, &mut y, 2, 3);
        assert_eq!(y[0], 11.0); // 1*1 + 0*2 + 0*3 + 10
        assert_eq!(y[1], 22.0); // 0*1 + 1*2 + 0*3 + 20
    }

    #[test]
    fn test_matvec() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];

        matvec(&w, &x, &mut y, 2, 2);
        assert_eq!(y[0], 3.0); // 1*1 + 2*1
        assert_eq!(y[1], 7.0); // 3*1 + 4*1
    }

    #[test]
    fn test_matvec_add_short_weights() {
        // Gracefully handle w shorter than expected
        let w = vec![1.0, 2.0]; // only 2 elements, but we ask for 2x2
        let b = vec![0.0, 0.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];

        matvec_add(&w, &b, &x, &mut y, 2, 2);
        assert_eq!(y[0], 3.0); // 1*1 + 2*1
        assert_eq!(y[1], 0.0); // out of bounds, stays 0
    }
}
