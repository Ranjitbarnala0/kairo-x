//! Enforcement subsystem — controls LLM behavior through system prompt templates,
//! compliance tracking, and placeholder detection.
//!
//! The enforcement module works in conjunction with the classification module:
//! classification detects problems, enforcement prevents them. This includes:
//!
//! - **Templates** (`templates.rs`): 7 intensity-scaled enforcement prompt templates
//! - **Selector** (`selector.rs`): Rule-based template selection by call type and priority
//! - **Placeholder detection** (`placeholder.rs`): Language-aware code stub finder
//! - **Compliance** (`compliance.rs`): Rolling window tracking of LLM output quality

pub mod compliance;
pub mod placeholder;
pub mod selector;
pub mod templates;

pub use compliance::ComplianceTracker;
pub use placeholder::{detect_placeholders, PlaceholderMatch};
pub use selector::{effective_intensity, select_template};
pub use templates::Template;
