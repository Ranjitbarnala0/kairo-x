//! Enforcement prompt templates for KAIRO-X LLM calls.
//!
//! Each template contains sentences ordered by intensity. The [`Template::render`]
//! method takes an intensity value (0.0 to 1.0) and includes sentences up to
//! the corresponding threshold, producing enforcement text that scales from
//! gentle reminders to strict demands.
//!
//! Templates are statically defined — no allocation at render time beyond
//! the output `String`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Template type enum
// ---------------------------------------------------------------------------

/// The 7 enforcement template types, corresponding to different LLM call contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Template {
    /// Planning and decomposition calls.
    Plan,
    /// Critical priority implementation calls.
    CriticalImpl,
    /// Standard priority implementation calls.
    StandardImpl,
    /// Verification (LLM layer 2) calls.
    Verification,
    /// Deep adversarial audit calls.
    DeepAudit,
    /// Fix/repair calls after verification failure.
    Fix,
    /// No enforcement — used for mechanical priority or explain calls.
    None,
}

impl Template {
    /// Render this template at the given intensity level.
    ///
    /// `intensity` ranges from 0.0 (minimal enforcement) to 1.0 (maximum).
    /// Sentences are included if their threshold is <= the given intensity.
    ///
    /// Returns an empty string for [`Template::None`].
    pub fn render(&self, intensity: f32) -> String {
        let intensity = intensity.clamp(0.0, 1.0);
        let sentences = self.sentences();

        let included: Vec<&str> = sentences
            .iter()
            .filter(|s| s.threshold <= intensity)
            .map(|s| s.text)
            .collect();

        if included.is_empty() {
            return String::new();
        }

        included.join(" ")
    }

    /// Get the static sentence list for this template type.
    fn sentences(&self) -> &'static [EnforcementSentence] {
        match self {
            Self::Plan => PLAN_SENTENCES,
            Self::CriticalImpl => CRITICAL_IMPL_SENTENCES,
            Self::StandardImpl => STANDARD_IMPL_SENTENCES,
            Self::Verification => VERIFICATION_SENTENCES,
            Self::DeepAudit => DEEP_AUDIT_SENTENCES,
            Self::Fix => FIX_SENTENCES,
            Self::None => &[],
        }
    }
}

impl std::fmt::Display for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plan => write!(f, "Plan"),
            Self::CriticalImpl => write!(f, "CriticalImpl"),
            Self::StandardImpl => write!(f, "StandardImpl"),
            Self::Verification => write!(f, "Verification"),
            Self::DeepAudit => write!(f, "DeepAudit"),
            Self::Fix => write!(f, "Fix"),
            Self::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sentence structure
// ---------------------------------------------------------------------------

/// A single enforcement sentence with an intensity threshold.
struct EnforcementSentence {
    /// Minimum intensity required to include this sentence.
    threshold: f32,
    /// The sentence text.
    text: &'static str,
}

// ---------------------------------------------------------------------------
// Template sentence definitions
// ---------------------------------------------------------------------------

static PLAN_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "Break this task into clearly defined sections and implementable components.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Each component must be specific enough that a single LLM call can implement it completely.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Identify all dependencies between components and order them topologically.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "Assign a priority (critical, standard, or mechanical) to each component based on its architectural importance.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "Every component must have a concrete specification — not vague descriptions like \"handle errors\" but precise behavior: which errors, how to handle them, what to return.",
    },
    EnforcementSentence {
        threshold: 0.9,
        text: "Do not bundle unrelated functionality into a single component. If a component does more than one thing, split it.",
    },
];

static CRITICAL_IMPL_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "Implement this component completely with production-grade code.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Include proper error handling for every fallible operation.",
    },
    EnforcementSentence {
        threshold: 0.3,
        text: "Handle all edge cases specified or implied by the specification.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Do NOT use TODO, FIXME, placeholder, stub, or any form of incomplete implementation.",
    },
    EnforcementSentence {
        threshold: 0.5,
        text: "Every function must have a complete body — no empty functions, no pass-only functions, no unimplemented!() macros.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "If a function is too complex to implement in this call, decompose it into helper functions and implement all of them.",
    },
    EnforcementSentence {
        threshold: 0.7,
        text: "This is a critical component. It will be audited by a separate AI. Anything less than complete implementation will be rejected and you will have to redo it.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "Write defensive code: validate inputs at boundaries, use appropriate types to prevent misuse, and document invariants.",
    },
    EnforcementSentence {
        threshold: 0.9,
        text: "ABSOLUTE REQUIREMENT: Every line of code must be a real implementation. If you output a single placeholder, TODO, or stub, the entire response will be rejected. Previous responses from you contained placeholders and were rejected.",
    },
];

static STANDARD_IMPL_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "Implement this component with complete, working code.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Include error handling for operations that can fail.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Do not leave any function body empty or use placeholder comments.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "Every code path must be implemented — no TODO markers, no stubs, no \"implement later\" comments.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "Your implementation will be verified. Incomplete or placeholder code will be rejected and you will need to redo it.",
    },
    EnforcementSentence {
        threshold: 0.9,
        text: "WARNING: Previous attempts contained placeholder code and were rejected. Output only complete, tested-in-your-head implementations.",
    },
];

static VERIFICATION_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "Verify this implementation against the specification.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Respond with PASS if the implementation is correct and complete, or FAIL followed by numbered issues.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Check for: spec compliance, edge case handling, error handling, and logical correctness.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "Look for placeholder code: TODO comments, empty function bodies, unimplemented!() calls, stub implementations.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "Be thorough. Previous verifications missed issues that caused problems later. If you are uncertain about any aspect, that is a FAIL.",
    },
];

static DEEP_AUDIT_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "You are auditing code written by another AI. Your job is to find problems.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Assume there are bugs until proven otherwise. Do not rubber-stamp.",
    },
    EnforcementSentence {
        threshold: 0.3,
        text: "Check every function for: correct logic, proper error handling, edge cases, and spec compliance.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Look for subtle issues: off-by-one errors, race conditions, resource leaks, integer overflow, missing null checks.",
    },
    EnforcementSentence {
        threshold: 0.5,
        text: "Verify that error handling is not just present but correct — wrong error messages, swallowed errors, and missing error propagation are all issues.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "Check for placeholder code that previous verification may have missed: functions that return default values without real logic, empty catch blocks, TODO-like comments.",
    },
    EnforcementSentence {
        threshold: 0.7,
        text: "Respond PASS only if you are genuinely confident the code is production-ready. FAIL with numbered issues otherwise.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "Think adversarially. What inputs could break this? What happens under load? What happens when dependencies fail?",
    },
    EnforcementSentence {
        threshold: 0.9,
        text: "This is the last line of defense before this code ships. Previous audits were too lenient and bugs escaped to production. Be strict.",
    },
];

static FIX_SENTENCES: &[EnforcementSentence] = &[
    EnforcementSentence {
        threshold: 0.0,
        text: "Fix the issues identified in the verification above.",
    },
    EnforcementSentence {
        threshold: 0.2,
        text: "Address every numbered issue — do not skip any.",
    },
    EnforcementSentence {
        threshold: 0.4,
        text: "Use SEARCH/REPLACE blocks with minimal context for each fix.",
    },
    EnforcementSentence {
        threshold: 0.6,
        text: "Do not introduce new placeholders or TODOs while fixing. Every fix must be a complete implementation.",
    },
    EnforcementSentence {
        threshold: 0.8,
        text: "After fixing, mentally re-verify the affected code paths to ensure the fix is correct and does not introduce regressions.",
    },
    EnforcementSentence {
        threshold: 0.9,
        text: "WARNING: Your previous fix attempt was rejected because it introduced new issues or did not fully address the original problems. Be precise and thorough.",
    },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_template_is_empty() {
        let result = Template::None.render(1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_plan_minimal_intensity() {
        let result = Template::Plan.render(0.0);
        assert!(result.contains("Break this task"));
        assert!(!result.contains("priority"));
    }

    #[test]
    fn test_plan_full_intensity() {
        let result = Template::Plan.render(1.0);
        assert!(result.contains("Break this task"));
        assert!(result.contains("priority"));
        assert!(result.contains("dependencies"));
    }

    #[test]
    fn test_critical_impl_escalation() {
        let low = Template::CriticalImpl.render(0.1);
        let mid = Template::CriticalImpl.render(0.5);
        let high = Template::CriticalImpl.render(1.0);

        // Higher intensity includes more text
        assert!(mid.len() > low.len());
        assert!(high.len() > mid.len());

        // High intensity includes the "ABSOLUTE REQUIREMENT" warning
        assert!(high.contains("ABSOLUTE REQUIREMENT"));
        assert!(!low.contains("ABSOLUTE REQUIREMENT"));
    }

    #[test]
    fn test_standard_impl_at_medium() {
        let result = Template::StandardImpl.render(0.5);
        assert!(result.contains("working code"));
        assert!(result.contains("error handling"));
    }

    #[test]
    fn test_verification_includes_pass_fail_instruction() {
        let result = Template::Verification.render(0.3);
        assert!(result.contains("PASS"));
        assert!(result.contains("FAIL"));
    }

    #[test]
    fn test_deep_audit_adversarial() {
        let result = Template::DeepAudit.render(0.9);
        assert!(result.contains("adversarially"));
        assert!(result.contains("bugs"));
    }

    #[test]
    fn test_fix_addresses_all_issues() {
        let result = Template::Fix.render(0.5);
        assert!(result.contains("every numbered issue"));
    }

    #[test]
    fn test_intensity_clamping() {
        // Values outside 0-1 should be clamped
        let neg = Template::Plan.render(-0.5);
        let zero = Template::Plan.render(0.0);
        assert_eq!(neg, zero);

        let high = Template::Plan.render(1.5);
        let max = Template::Plan.render(1.0);
        assert_eq!(high, max);
    }

    #[test]
    fn test_template_display() {
        assert_eq!(format!("{}", Template::Plan), "Plan");
        assert_eq!(format!("{}", Template::CriticalImpl), "CriticalImpl");
        assert_eq!(format!("{}", Template::DeepAudit), "DeepAudit");
        assert_eq!(format!("{}", Template::None), "None");
    }

    #[test]
    fn test_sentences_monotonically_increase_threshold() {
        // Verify that sentences within each template are ordered by threshold
        let templates = [
            Template::Plan,
            Template::CriticalImpl,
            Template::StandardImpl,
            Template::Verification,
            Template::DeepAudit,
            Template::Fix,
        ];

        for template in templates {
            let sentences = template.sentences();
            for window in sentences.windows(2) {
                assert!(
                    window[0].threshold <= window[1].threshold,
                    "Template {:?} has non-monotonic thresholds: {} > {}",
                    template,
                    window[0].threshold,
                    window[1].threshold,
                );
            }
        }
    }
}
