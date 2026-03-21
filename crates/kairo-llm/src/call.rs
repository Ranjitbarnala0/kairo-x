//! LLM call assembly — constructs the message payload for each LLM call type.
//!
//! The call assembler is responsible for combining enforcement templates,
//! context packages, and action-specific instructions into a message sequence
//! ready for the provider to send.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Message types (provider-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }

    /// Estimated token count (rough: 1 token ≈ 4 chars for English).
    pub fn estimated_tokens(&self) -> u32 {
        (self.content.len() as u32 / 4).max(1)
    }
}

// ---------------------------------------------------------------------------
// LLM call type — the 8 action types from §8.1
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LLMCallType {
    /// Generate a natural language plan for the task
    Plan,
    /// Implement code for a graph node
    Implement,
    /// Verify implementation correctness (LLM layer 2)
    Verify,
    /// Deep adversarial audit of implementation
    Audit,
    /// Fix issues found during verification
    Fix,
    /// Explain code or behavior
    Explain,
    /// Decompose a node into smaller sub-nodes
    Decompose,
    /// Debug an issue
    Debug,
}

impl LLMCallType {
    /// Default temperature for this call type.
    pub fn default_temperature(&self) -> f32 {
        match self {
            // Audit uses higher temperature for diverse evaluation
            Self::Audit => 0.3,
            // Everything else uses deterministic sampling
            _ => 0.0,
        }
    }

    /// Whether this call type typically continues an existing session.
    pub fn prefers_continue(&self) -> bool {
        matches!(
            self,
            Self::Verify | Self::Audit | Self::Fix | Self::Explain | Self::Debug
        )
    }
}

// ---------------------------------------------------------------------------
// LLM request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub temperature: f32,
    pub max_output_tokens: u32,
    /// What type of call this is (for response handling)
    pub call_type: LLMCallType,
    /// Stop sequences (if any)
    pub stop_sequences: Vec<String>,
}

impl LLMRequest {
    /// Total estimated input tokens across all messages.
    pub fn estimated_input_tokens(&self) -> u32 {
        self.messages.iter().map(|m| m.estimated_tokens()).sum()
    }
}

// ---------------------------------------------------------------------------
// Call assembler
// ---------------------------------------------------------------------------

/// System prompt that is included at the start of every new LLM session.
pub const SYSTEM_PROMPT: &str = r#"You are an expert software engineer working as part of the KAIRO-X enforcement agent system. You write production-grade code with proper error handling, edge cases, and no placeholders.

Rules:
1. Never output TODO, FIXME, placeholder, stub, or incomplete implementations.
2. Every function must be fully implemented with proper error handling.
3. Follow the project's existing patterns and conventions.
4. When editing existing files, use SEARCH/REPLACE blocks (described in instructions).
5. When you need more context, respond with NEED_CONTEXT markers.
6. For verification, respond with PASS or FAIL followed by details.

If you cannot implement something due to missing information, say so explicitly.
Do not guess or leave gaps."#;

/// Assembles the message sequence for an LLM call.
///
/// For new sessions: system prompt + full context + enforcement template + instruction.
/// For continuing sessions: new context (if any) + enforcement template + instruction.
#[allow(clippy::too_many_arguments)]
pub fn assemble_call(
    action: LLMCallType,
    node_title: &str,
    node_spec: &str,
    context_rendered: &str,
    new_context_only: &str,
    template_rendered: &str,
    is_new_session: bool,
    model: &str,
    max_output_tokens: u32,
) -> LLMRequest {
    let mut messages = Vec::new();
    let instruction = instruction_for(action, node_title, node_spec);

    if is_new_session {
        messages.push(Message::system(SYSTEM_PROMPT.to_string()));
        messages.push(Message::user(format!(
            "{template_rendered}\n\n{context_rendered}\n\n{instruction}"
        )));
    } else {
        // Continuing session — only include new context
        if new_context_only.is_empty() {
            messages.push(Message::user(format!(
                "{template_rendered}\n\n{instruction}"
            )));
        } else {
            messages.push(Message::user(format!(
                "{template_rendered}\n\nAdditional context:\n{new_context_only}\n\n{instruction}"
            )));
        }
    }

    LLMRequest {
        messages,
        model: model.to_string(),
        temperature: action.default_temperature(),
        max_output_tokens,
        call_type: action,
        stop_sequences: Vec::new(),
    }
}

/// Generate the action-specific instruction text for the LLM.
fn instruction_for(action: LLMCallType, node_title: &str, node_spec: &str) -> String {
    match action {
        LLMCallType::Plan => format!(
            "Break this task into sections, subsections, and implementable components.\n\
             For each component, describe what it does and what it depends on.\n\n\
             Task: {node_title}\n\n\
             Specification:\n{node_spec}"
        ),
        LLMCallType::Implement => format!(
            "Implement the following component. Write complete, production-grade code.\n\
             No placeholders, no TODOs, no stubs.\n\n\
             Component: {node_title}\n\n\
             Specification:\n{node_spec}\n\n\
             For new files: output the complete file content.\n\
             For edits to existing files: use SEARCH/REPLACE blocks:\n\
             <<<SEARCH\n\
             exact existing code to find\n\
             >>>\n\
             <<<REPLACE\n\
             new code to replace it with\n\
             >>>\n\n\
             Make each SEARCH block as small as possible — just the lines that need to change,\n\
             plus 1-2 lines of surrounding context for unique identification."
        ),
        LLMCallType::Verify => format!(
            "Verify the implementation of: {node_title}\n\n\
             Specification:\n{node_spec}\n\n\
             Respond with PASS if the implementation is correct and complete.\n\
             Respond with FAIL followed by numbered issues if anything is wrong.\n\
             Check: spec compliance, edge cases, error handling, logic correctness."
        ),
        LLMCallType::Audit => format!(
            "You are auditing code written by another AI. Your job is to find problems.\n\
             Assume there are bugs until proven otherwise.\n\n\
             Component: {node_title}\n\n\
             Specification:\n{node_spec}\n\n\
             Automated checks passed. Find what they missed:\n\
             - Spec compliance: does the code do everything the spec requires?\n\
             - Edge cases: what inputs or conditions aren't handled?\n\
             - Error handling: what can go wrong that isn't caught?\n\
             - Logic: is the algorithm correct for all cases?\n\n\
             Respond PASS if genuinely correct. Respond FAIL with numbered issues if not.\n\
             Do not rubber-stamp. If you are uncertain about any aspect, that's a FAIL."
        ),
        LLMCallType::Fix => format!(
            "Fix the issues found in: {node_title}\n\n\
             The issues are described in the conversation above.\n\
             Apply fixes using SEARCH/REPLACE blocks.\n\
             Make sure every reported issue is addressed."
        ),
        LLMCallType::Explain => format!(
            "Explain the implementation of: {node_title}\n\n\
             Specification:\n{node_spec}"
        ),
        LLMCallType::Decompose => format!(
            "The component \"{node_title}\" is too large to implement in a single step.\n\n\
             Specification:\n{node_spec}\n\n\
             Break it into smaller, independently implementable sub-components.\n\
             Each sub-component should be self-contained and testable.\n\
             List dependencies between sub-components."
        ),
        LLMCallType::Debug => format!(
            "Debug the issue with: {node_title}\n\n\
             The error details are in the conversation above.\n\
             Identify the root cause and suggest a fix."
        ),
    }
}

// ---------------------------------------------------------------------------
// JSON plan structuring prompt (two-pass, §Flaw 4)
// ---------------------------------------------------------------------------

/// The prompt sent in the second pass to convert natural language plan to JSON.
pub const PLAN_TO_JSON_PROMPT: &str = r#"Now convert your plan above into this exact JSON format.
Output ONLY valid JSON, no other text:
[
  {
    "id": 1,
    "title": "short title",
    "spec": "what to implement",
    "priority": "critical|standard|mechanical",
    "depends_on": [2, 3]
  }
]"#;

/// Prompt for fixing malformed JSON (retry after parse failure).
///
/// Substitutes the actual parse error into the prompt text.
pub fn fix_json_prompt(error: &str) -> String {
    format!(
        "The JSON you provided is malformed. Here is the parse error:\n\n{error}\n\nPlease output corrected valid JSON only, no other text."
    )
}
