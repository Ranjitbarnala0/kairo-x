//! Weight loading and storage for the controller neural network.
//!
//! Loads pre-trained weights from binary files exported by the Python
//! training pipeline. Format: raw f32 arrays with a JSON manifest
//! describing tensor shapes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number for weight file validation ("KAIR" in ASCII).
/// Must appear as the first 4 bytes of weights.bin to confirm correct
/// endianness and file integrity.
const WEIGHT_MAGIC: [u8; 4] = [0x4B, 0x41, 0x49, 0x52];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Manifest parse error: {0}")]
    ManifestParse(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Weight file corrupted: {0}")]
    Corrupted(String),

    #[error(
        "Weight file magic mismatch: expected {expected:02X?}, got {actual:02X?}. \
         This indicates endianness mismatch, file corruption, or an \
         incompatible weight format."
    )]
    MagicMismatch {
        expected: [u8; 4],
        actual: [u8; 4],
    },
}

// ---------------------------------------------------------------------------
// Manifest — describes the layout of a weight file
// ---------------------------------------------------------------------------

/// Describes all tensors in a weight file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightManifest {
    /// Model architecture version (for compatibility checking).
    pub version: u32,
    /// Total number of parameters.
    pub total_parameters: usize,
    /// Individual tensor descriptors.
    pub tensors: Vec<TensorDescriptor>,
}

/// Describes a single tensor in the weight file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Tensor name (e.g., "layers.0.gate_weight").
    pub name: String,
    /// Shape dimensions (e.g., [288, 192]).
    pub shape: Vec<usize>,
    /// Byte offset from the start of the data section.
    pub offset: usize,
    /// Number of f32 elements.
    pub num_elements: usize,
}

// ---------------------------------------------------------------------------
// Tensor — a multi-dimensional f32 array
// ---------------------------------------------------------------------------

/// A dense f32 tensor stored in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        Self {
            data: vec![0.0; num_elements],
            shape: shape.to_vec(),
        }
    }

    /// Create a tensor filled with ones (used for LayerNorm gamma defaults).
    pub fn ones(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        Self {
            data: vec![1.0; num_elements],
            shape: shape.to_vec(),
        }
    }

    /// Create from raw data with shape validation.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, WeightError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(WeightError::Corrupted(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected,
            )));
        }
        Ok(Self { data, shape })
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Get a slice for 2D matrix row access: tensor[row, :]
    ///
    /// Returns `None` if the tensor is not 2D or the row index is out of range.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        if self.shape.len() != 2 || i >= self.shape[0] {
            return None;
        }
        let cols = self.shape[1];
        let start = i * cols;
        self.data.get(start..start + cols)
    }

    /// Mutable row access.
    ///
    /// Returns `None` if the tensor is not 2D or the row index is out of range.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [f32]> {
        if self.shape.len() != 2 || i >= self.shape[0] {
            return None;
        }
        let cols = self.shape[1];
        let start = i * cols;
        self.data.get_mut(start..start + cols)
    }
}

// ---------------------------------------------------------------------------
// Weight store — holds all loaded weights
// ---------------------------------------------------------------------------

/// Holds all loaded model weights indexed by name.
#[derive(Debug)]
pub struct WeightStore {
    tensors: HashMap<String, Tensor>,
    manifest: WeightManifest,
}

impl WeightStore {
    /// Load weights from a directory containing manifest.json and weights.bin.
    pub fn load(dir: &Path) -> Result<Self, WeightError> {
        let manifest_path = dir.join("manifest.json");
        let weights_path = dir.join("weights.bin");

        // Load manifest
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: WeightManifest = serde_json::from_str(&manifest_json).map_err(|e| {
            WeightError::ManifestParse(format!("Failed to parse manifest: {e}"))
        })?;

        // Load weight data
        let mut weight_file = std::fs::File::open(&weights_path)?;
        let mut weight_data = Vec::new();
        weight_file.read_to_end(&mut weight_data)?;

        // Validate magic number (first 4 bytes must be "KAIR")
        if weight_data.len() < WEIGHT_MAGIC.len() {
            return Err(WeightError::Corrupted(format!(
                "Weight file too small ({} bytes), expected at least {} bytes for magic header",
                weight_data.len(),
                WEIGHT_MAGIC.len(),
            )));
        }
        let actual_magic: [u8; 4] = [
            weight_data[0],
            weight_data[1],
            weight_data[2],
            weight_data[3],
        ];
        if actual_magic != WEIGHT_MAGIC {
            return Err(WeightError::MagicMismatch {
                expected: WEIGHT_MAGIC,
                actual: actual_magic,
            });
        }

        // Advance past magic header — tensor offsets in the manifest are
        // relative to the start of the data section (byte 4 onward).
        let weight_data = &weight_data[WEIGHT_MAGIC.len()..];

        // Parse tensors
        let mut tensors = HashMap::new();

        for desc in &manifest.tensors {
            let byte_offset = desc.offset;
            let byte_len = desc.num_elements * 4; // f32 = 4 bytes

            if byte_offset + byte_len > weight_data.len() {
                return Err(WeightError::Corrupted(format!(
                    "Tensor '{}' extends beyond file (offset={}, len={}, file_size={})",
                    desc.name,
                    byte_offset,
                    byte_len,
                    weight_data.len(),
                )));
            }

            let float_data: Vec<f32> = weight_data[byte_offset..byte_offset + byte_len]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_data(float_data, desc.shape.clone())?;
            tensors.insert(desc.name.clone(), tensor);
        }

        tracing::info!(
            parameters = manifest.total_parameters,
            tensors = tensors.len(),
            "Controller weights loaded"
        );

        Ok(Self { tensors, manifest })
    }

    /// Create an uninitialized (zero) weight store for testing/default behavior.
    ///
    /// The controller will produce random-ish outputs, but the rule-based systems
    /// provide sensible defaults regardless.
    pub fn zeros(config: &super::ControllerConfig) -> Self {
        let mut tensors = HashMap::new();

        // Create zero-initialized tensors for all expected weights
        // Input projection: [n_input_slots * d_model, d_model]
        let input_dim = config.n_input_slots * config.d_model;
        tensors.insert(
            "input_proj.weight".to_string(),
            Tensor::zeros(&[config.d_model, input_dim]),
        );
        tensors.insert(
            "input_proj.bias".to_string(),
            Tensor::zeros(&[config.d_model]),
        );

        // Layer weights — matching Python LiquidBlock structure exactly
        for layer_idx in 0..config.n_layers {
            let prefix = format!("layers.{layer_idx}");

            // Input normalization: LayerNorm(d_model) — pre-norm before input_proj
            tensors.insert(
                format!("{prefix}.input_norm.weight"),
                Tensor::ones(&[config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.input_norm.bias"),
                Tensor::zeros(&[config.d_model]),
            );

            // Input projection: Linear(d_model, n_bands * d_state)
            let proj_out = config.n_bands * config.d_state;
            tensors.insert(
                format!("{prefix}.input_proj.weight"),
                Tensor::zeros(&[proj_out, config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.input_proj.bias"),
                Tensor::zeros(&[proj_out]),
            );

            // Per-band GRU weights (4 bands)
            // Python: BandGRU(d_input=d_state, d_state=d_state)
            //   W_input: Linear(d_state, 3*d_state)
            //   W_state: Linear(d_state, 3*d_state)
            for band in 0..config.n_bands {
                let bp = format!("{prefix}.band_grus.{band}");

                tensors.insert(
                    format!("{bp}.W_input.weight"),
                    Tensor::zeros(&[config.d_state * 3, config.d_state]),
                );
                tensors.insert(
                    format!("{bp}.W_input.bias"),
                    Tensor::zeros(&[config.d_state * 3]),
                );
                tensors.insert(
                    format!("{bp}.W_state.weight"),
                    Tensor::zeros(&[config.d_state * 3, config.d_state]),
                );
                tensors.insert(
                    format!("{bp}.W_state.bias"),
                    Tensor::zeros(&[config.d_state * 3]),
                );

                // Learnable initial state h0: [d_state]
                tensors.insert(
                    format!("{bp}.h0"),
                    Tensor::zeros(&[config.d_state]),
                );
            }

            // Cross-band normalization: LayerNorm(d_state) — pre-norm before attention
            tensors.insert(
                format!("{prefix}.cross_band_norm.weight"),
                Tensor::ones(&[config.d_state]),
            );
            tensors.insert(
                format!("{prefix}.cross_band_norm.bias"),
                Tensor::zeros(&[config.d_state]),
            );

            // Cross-band attention (matches Python CrossBandAttention):
            // qkv: Linear(d_state, 3*d_state, bias=False)
            // out_proj: Linear(d_state, d_state, bias=False)
            tensors.insert(
                format!("{prefix}.cross_band_attn.qkv.weight"),
                Tensor::zeros(&[config.d_state * 3, config.d_state]),
            );
            tensors.insert(
                format!("{prefix}.cross_band_attn.out_proj.weight"),
                Tensor::zeros(&[config.d_state, config.d_state]),
            );

            // Per-band FFN weights (4 bands)
            // Python: Sequential(LayerNorm(d_state), Linear(d_state, d_state*2),
            //         SiLU, Dropout, Linear(d_state*2, d_state), Dropout)
            // Module indices: 0=LayerNorm, 1=Linear_up, 2=SiLU, 3=Dropout, 4=Linear_down, 5=Dropout
            let d_band_ffn = config.d_state * 2;
            for band in 0..config.n_bands {
                let bp = format!("{prefix}.band_ffns.{band}");

                // LayerNorm (index 0)
                tensors.insert(
                    format!("{bp}.0.weight"),
                    Tensor::ones(&[config.d_state]),
                );
                tensors.insert(
                    format!("{bp}.0.bias"),
                    Tensor::zeros(&[config.d_state]),
                );
                // Linear up (index 1): d_state -> d_state*2
                tensors.insert(
                    format!("{bp}.1.weight"),
                    Tensor::zeros(&[d_band_ffn, config.d_state]),
                );
                tensors.insert(
                    format!("{bp}.1.bias"),
                    Tensor::zeros(&[d_band_ffn]),
                );
                // Linear down (index 4): d_state*2 -> d_state
                tensors.insert(
                    format!("{bp}.4.weight"),
                    Tensor::zeros(&[config.d_state, d_band_ffn]),
                );
                tensors.insert(
                    format!("{bp}.4.bias"),
                    Tensor::zeros(&[config.d_state]),
                );
            }

            // Learned band FFN state scale (scalar, initialized to 0.1 in Python)
            {
                let mut scale_tensor = Tensor::zeros(&[1]);
                scale_tensor.data[0] = 0.1;
                tensors.insert(
                    format!("{prefix}.band_ffn_state_scale"),
                    scale_tensor,
                );
            }

            // Band-to-model projection: Linear(n_bands * d_state, d_model)
            let flat_dim = config.n_bands * config.d_state;
            tensors.insert(
                format!("{prefix}.band_to_model.weight"),
                Tensor::zeros(&[config.d_model, flat_dim]),
            );
            tensors.insert(
                format!("{prefix}.band_to_model.bias"),
                Tensor::zeros(&[config.d_model]),
            );

            // SwiGLU FFN normalization: LayerNorm(d_model) — pre-norm before FFN
            tensors.insert(
                format!("{prefix}.ffn_norm.weight"),
                Tensor::ones(&[config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.ffn_norm.bias"),
                Tensor::zeros(&[config.d_model]),
            );

            // SwiGLU FFN weights (all bias=False in Python):
            // ffn_gate: Linear(d_model, d_ffn, bias=False)
            // ffn_up:   Linear(d_model, d_ffn, bias=False)
            // ffn_down: Linear(d_ffn, d_model, bias=False)
            tensors.insert(
                format!("{prefix}.ffn_gate.weight"),
                Tensor::zeros(&[config.d_ffn, config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.ffn_up.weight"),
                Tensor::zeros(&[config.d_ffn, config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.ffn_down.weight"),
                Tensor::zeros(&[config.d_model, config.d_ffn]),
            );
        }

        // Output normalization: LayerNorm(d_model) — applied after all layers, before heads
        tensors.insert(
            "output_norm.weight".to_string(),
            Tensor::ones(&[config.d_model]),
        );
        tensors.insert(
            "output_norm.bias".to_string(),
            Tensor::zeros(&[config.d_model]),
        );

        // Output head weights
        tensors.insert(
            "heads.action.weight".to_string(),
            Tensor::zeros(&[34, config.d_model]),
        );
        tensors.insert(
            "heads.action.bias".to_string(),
            Tensor::zeros(&[34]),
        );
        tensors.insert(
            "heads.context_budget.weight".to_string(),
            Tensor::zeros(&[1, config.d_model]),
        );
        tensors.insert(
            "heads.context_budget.bias".to_string(),
            Tensor::zeros(&[1]),
        );
        tensors.insert(
            "heads.enforcement_intensity.weight".to_string(),
            Tensor::zeros(&[1, config.d_model]),
        );
        tensors.insert(
            "heads.enforcement_intensity.bias".to_string(),
            Tensor::zeros(&[1]),
        );
        tensors.insert(
            "heads.session_edge.weight".to_string(),
            Tensor::zeros(&[1, config.d_model]),
        );
        tensors.insert(
            "heads.session_edge.bias".to_string(),
            Tensor::zeros(&[1]),
        );
        tensors.insert(
            "heads.stop.weight".to_string(),
            Tensor::zeros(&[4, config.d_model]),
        );
        tensors.insert(
            "heads.stop.bias".to_string(),
            Tensor::zeros(&[4]),
        );

        let manifest = WeightManifest {
            version: 1,
            total_parameters: tensors.values().map(|t| t.num_elements()).sum(),
            tensors: Vec::new(), // Zero store doesn't need a real manifest
        };

        Self { tensors, manifest }
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Result<&Tensor, WeightError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightError::MissingTensor(name.to_string()))
    }

    /// Get a tensor by name, or return a zero tensor if missing.
    /// Used during development when weights haven't been trained yet.
    pub fn get_or_zeros(&self, name: &str, shape: &[usize]) -> Tensor {
        self.tensors
            .get(name)
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(shape))
    }

    /// Total parameter count.
    pub fn total_parameters(&self) -> usize {
        self.manifest.total_parameters
    }
}
