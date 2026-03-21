//! Neural controller for KAIRO-X (§7).
//!
//! 42M-parameter liquid-hybrid recurrent enforcement controller.
//! This module implements the inference path in pure Rust — no Python,
//! no ONNX, no external runtime. Weights are loaded from binary files
//! exported by the Python training pipeline.
//!
//! **Architecture:**
//! - 12 liquid-hybrid recurrent layers
//! - 4 state bands × 192 dims = 768 total state dims
//! - d_model = 288, d_ffn = 1152
//! - 32 input slots, 6 output heads
//! - ~41.5M parameters
//!
//! **Primary learned skills:**
//! 1. Context selection — what information to include in each LLM call
//! 2. Context budget — how many tokens to allocate

pub mod weights;
pub mod input_assembly;
pub mod liquid_block;
pub mod heads;
pub mod inference;

pub use inference::{Controller, ControllerConfig, ControllerOutput};
pub use input_assembly::InputPacket;
pub use weights::WeightStore;
