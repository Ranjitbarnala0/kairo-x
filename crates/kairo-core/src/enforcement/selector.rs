//! Rule-based enforcement template selection.
//!
//! Pure lookup table: given an [`LLMCallType`] and a [`Priority`], returns the
//! appropriate enforcement [`Template`] and a computed intensity value.

use kairo_llm::call::LLMCallType;

use crate::arena::node::Priority;
use crate::session::token_tracker::CostMode;
use super::templates::Template;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Selection result containing the template and computed intensity.
#[derive(Debug, Clone)]
pub struct TemplateSelection {
    /// The enforcement template to use.
    pub template: Template,
    /// The intensity at which to render the template (0.0 to 1.0).
    pub intensity: f32,
}

/// Select the appropriate enforcement template and intensity for a given
/// LLM call type and node priority.
///
/// This is a pure lookup table with no neural network or heuristic logic.
/// The intensity is derived from the priority's base intensity, with
/// adjustments based on the call type.
///
/// # Selection table
///
/// | Call Type  | Critical       | Standard       | Mechanical     |
/// |------------|----------------|----------------|----------------|
/// | Plan       | Plan @ 0.9     | Plan @ 0.6     | Plan @ 0.3     |
/// | Implement  | CriticalImpl   | StandardImpl   | StandardImpl@.15|
/// | Verify     | Verification   | Verification   | Verification   |
/// | Audit      | DeepAudit      | DeepAudit      | Verification   |
/// | Fix        | Fix @ high     | Fix @ med      | Fix @ low      |
/// | Explain    | None           | None           | None           |
/// | Decompose  | Plan @ 0.7     | Plan @ 0.5     | Plan @ 0.3     |
/// | Debug      | Fix @ high     | Fix @ med      | Fix @ low      |
pub fn select_template(action: LLMCallType, priority: Priority) -> TemplateSelection {
    let base_intensity = priority.base_intensity();

    match (action, priority) {
        // ----- Plan -----
        (LLMCallType::Plan, _) => TemplateSelection {
            template: Template::Plan,
            intensity: base_intensity,
        },

        // ----- Implement -----
        (LLMCallType::Implement, Priority::Critical) => TemplateSelection {
            template: Template::CriticalImpl,
            intensity: base_intensity,
        },
        (LLMCallType::Implement, Priority::Standard) => TemplateSelection {
            template: Template::StandardImpl,
            intensity: base_intensity,
        },
        (LLMCallType::Implement, Priority::Mechanical) => TemplateSelection {
            template: Template::StandardImpl,
            intensity: 0.15,
        },

        // ----- Verify -----
        (LLMCallType::Verify, _) => TemplateSelection {
            template: Template::Verification,
            intensity: base_intensity,
        },

        // ----- Audit -----
        (LLMCallType::Audit, Priority::Critical | Priority::Standard) => TemplateSelection {
            template: Template::DeepAudit,
            intensity: base_intensity,
        },
        (LLMCallType::Audit, Priority::Mechanical) => TemplateSelection {
            template: Template::Verification,
            intensity: base_intensity,
        },

        // ----- Fix -----
        (LLMCallType::Fix, _) => TemplateSelection {
            template: Template::Fix,
            intensity: base_intensity,
        },

        // ----- Explain -----
        (LLMCallType::Explain, _) => TemplateSelection {
            template: Template::None,
            intensity: 0.0,
        },

        // ----- Decompose -----
        (LLMCallType::Decompose, Priority::Critical) => TemplateSelection {
            template: Template::Plan,
            intensity: 0.7,
        },
        (LLMCallType::Decompose, Priority::Standard) => TemplateSelection {
            template: Template::Plan,
            intensity: 0.5,
        },
        (LLMCallType::Decompose, Priority::Mechanical) => TemplateSelection {
            template: Template::Plan,
            intensity: 0.3,
        },

        // ----- Debug -----
        (LLMCallType::Debug, _) => TemplateSelection {
            template: Template::Fix,
            intensity: base_intensity,
        },
    }
}

/// Compute the effective enforcement intensity given a node's priority,
/// a neural adjustment factor (from the controller), and the active cost mode.
///
/// The base intensity comes from the priority level, the cost factor from the
/// cost mode, and the neural adjustment is added before the cost scaling.
/// The result is clamped to [0.0, 1.0].
pub fn effective_intensity(priority: Priority, neural_adjustment: f32, cost_mode: CostMode) -> f32 {
    let base = match priority {
        Priority::Critical => 0.9,
        Priority::Standard => 0.6,
        Priority::Mechanical => 0.3,
    };
    let cost_factor = match cost_mode {
        CostMode::Thorough => 1.2,
        CostMode::Balanced => 1.0,
        CostMode::Efficient => 0.6,
    };
    ((base + neural_adjustment) * cost_factor).clamp(0.0, 1.0)
}

/// Select template with intensity adjusted by compliance history.
///
/// When the compliance tracker indicates recent bad responses, intensity
/// is increased to apply stronger enforcement pressure.
pub fn select_template_with_compliance(
    action: LLMCallType,
    priority: Priority,
    recent_success_rate: f32,
    consecutive_failures: u32,
) -> TemplateSelection {
    let mut selection = select_template(action, priority);

    // Increase intensity based on recent failures
    if consecutive_failures >= 3 {
        // Three or more consecutive failures: maximum intensity
        selection.intensity = 1.0;
    } else if consecutive_failures >= 2 {
        // Two failures: boost by 0.2
        selection.intensity = (selection.intensity + 0.2).min(1.0);
    } else if recent_success_rate < 0.5 {
        // Low success rate: boost by 0.15
        selection.intensity = (selection.intensity + 0.15).min(1.0);
    }

    selection
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_selection() {
        let sel = select_template(LLMCallType::Plan, Priority::Critical);
        assert_eq!(sel.template, Template::Plan);
        assert!((sel.intensity - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_critical_impl_selection() {
        let sel = select_template(LLMCallType::Implement, Priority::Critical);
        assert_eq!(sel.template, Template::CriticalImpl);
    }

    #[test]
    fn test_standard_impl_selection() {
        let sel = select_template(LLMCallType::Implement, Priority::Standard);
        assert_eq!(sel.template, Template::StandardImpl);
    }

    #[test]
    fn test_mechanical_impl_baseline_enforcement() {
        let sel = select_template(LLMCallType::Implement, Priority::Mechanical);
        assert_eq!(sel.template, Template::StandardImpl);
        assert!((sel.intensity - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audit_critical_deep() {
        let sel = select_template(LLMCallType::Audit, Priority::Critical);
        assert_eq!(sel.template, Template::DeepAudit);
    }

    #[test]
    fn test_audit_mechanical_regular() {
        let sel = select_template(LLMCallType::Audit, Priority::Mechanical);
        assert_eq!(sel.template, Template::Verification);
    }

    #[test]
    fn test_explain_no_enforcement() {
        let sel = select_template(LLMCallType::Explain, Priority::Critical);
        assert_eq!(sel.template, Template::None);
    }

    #[test]
    fn test_fix_inherits_priority_intensity() {
        let crit = select_template(LLMCallType::Fix, Priority::Critical);
        let std = select_template(LLMCallType::Fix, Priority::Standard);
        let mech = select_template(LLMCallType::Fix, Priority::Mechanical);

        assert!(crit.intensity > std.intensity);
        assert!(std.intensity > mech.intensity);
        assert_eq!(crit.template, Template::Fix);
    }

    #[test]
    fn test_compliance_boost_consecutive_failures() {
        let sel = select_template_with_compliance(
            LLMCallType::Implement,
            Priority::Standard,
            0.5,
            3, // 3 consecutive failures
        );
        assert!((sel.intensity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compliance_boost_two_failures() {
        let sel = select_template_with_compliance(
            LLMCallType::Implement,
            Priority::Standard,
            0.6,
            2,
        );
        // Standard base = 0.6, + 0.2 boost = 0.8
        assert!((sel.intensity - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compliance_boost_low_success_rate() {
        let sel = select_template_with_compliance(
            LLMCallType::Implement,
            Priority::Standard,
            0.4, // below 0.5
            0,
        );
        // Standard base = 0.6, + 0.15 boost = 0.75
        assert!((sel.intensity - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compliance_no_boost_when_healthy() {
        let sel = select_template_with_compliance(
            LLMCallType::Implement,
            Priority::Standard,
            0.9,
            0,
        );
        assert!((sel.intensity - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_decompose_selection() {
        let sel = select_template(LLMCallType::Decompose, Priority::Critical);
        assert_eq!(sel.template, Template::Plan);
        assert!((sel.intensity - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_debug_uses_fix_template() {
        let sel = select_template(LLMCallType::Debug, Priority::Standard);
        assert_eq!(sel.template, Template::Fix);
    }
}
