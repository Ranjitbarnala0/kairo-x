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
    pub fn row(&self, row: usize) -> &[f32] {
        assert!(self.shape.len() == 2, "row() requires 2D tensor");
        let cols = self.shape[1];
        let start = row * cols;
        &self.data[start..start + cols]
    }

    /// Mutable row access.
    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        assert!(self.shape.len() == 2, "row_mut() requires 2D tensor");
        let cols = self.shape[1];
        let start = row * cols;
        &mut self.data[start..start + cols]
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

        // Layer weights
        for layer_idx in 0..config.n_layers {
            let prefix = format!("layers.{layer_idx}");

            // Band projections (4 bands)
            for band in 0..config.n_bands {
                let bp = format!("{prefix}.band_{band}");

                // GRU gate weights
                tensors.insert(
                    format!("{bp}.gate_ih.weight"),
                    Tensor::zeros(&[config.d_state * 3, config.d_model]),
                );
                tensors.insert(
                    format!("{bp}.gate_ih.bias"),
                    Tensor::zeros(&[config.d_state * 3]),
                );
                tensors.insert(
                    format!("{bp}.gate_hh.weight"),
                    Tensor::zeros(&[config.d_state * 3, config.d_state]),
                );
                tensors.insert(
                    format!("{bp}.gate_hh.bias"),
                    Tensor::zeros(&[config.d_state * 3]),
                );
            }

            // Cross-band attention (matches Python CrossBandAttention):
            // qkv: combined Q/K/V projection [3*d_state, d_state]
            // out_proj: output projection [d_state, d_state]
            tensors.insert(
                format!("{prefix}.cross_band.qkv.weight"),
                Tensor::zeros(&[config.d_state * 3, config.d_state]),
            );
            tensors.insert(
                format!("{prefix}.cross_band.out_proj.weight"),
                Tensor::zeros(&[config.d_state, config.d_state]),
            );

            // FFN
            tensors.insert(
                format!("{prefix}.ffn.up.weight"),
                Tensor::zeros(&[config.d_ffn, config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.ffn.up.bias"),
                Tensor::zeros(&[config.d_ffn]),
            );
            tensors.insert(
                format!("{prefix}.ffn.down.weight"),
                Tensor::zeros(&[config.d_model, config.d_ffn]),
            );
            tensors.insert(
                format!("{prefix}.ffn.down.bias"),
                Tensor::zeros(&[config.d_model]),
            );

            // LayerNorms — gamma initialized to 1.0 (standard default),
            // NOT zero. Zero gamma kills the signal entirely, making the
            // controller produce constant output regardless of input.
            tensors.insert(
                format!("{prefix}.norm1.weight"),
                Tensor::ones(&[config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.norm1.bias"),
                Tensor::zeros(&[config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.norm2.weight"),
                Tensor::ones(&[config.d_model]),
            );
            tensors.insert(
                format!("{prefix}.norm2.bias"),
                Tensor::zeros(&[config.d_model]),
            );
        }

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
