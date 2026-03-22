//! Anthropic Claude API provider implementation.
//!
//! Translates KAIRO-X message format to Anthropic's Messages API format
//! and back.

use crate::call::{LLMRequest, Message, MessageRole};
use crate::response::{LLMRawResponse, StopReason};
use super::{ProviderError, ProviderKind, ProviderSpec};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Anthropic API types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

/// `Clone` is cheap: `reqwest::Client` is internally `Arc`-wrapped, so
/// cloning only bumps a reference count — the underlying connection pool
/// is shared across all clones.
#[derive(Clone)]
pub struct AnthropicProvider {
    client: Client,
    spec: ProviderSpec,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from a spec.
    ///
    /// Reads the API key from the environment variable specified in `spec.api_key_env`.
    pub fn new(spec: ProviderSpec) -> Result<Self, ProviderError> {
        let api_key = std::env::var(&spec.api_key_env).map_err(|_| {
            ProviderError::AuthFailed(format!(
                "Environment variable {} not set",
                spec.api_key_env
            ))
        })?;

        let base_url = spec
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout for long generations
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(ProviderError::Http)?;

        Ok(Self {
            client,
            spec,
            api_key,
            base_url,
        })
    }

    /// Send a request to the Anthropic Messages API.
    pub async fn send(&self, request: &LLMRequest) -> Result<LLMRawResponse, ProviderError> {
        // Separate system message from conversation messages
        let (system_text, conversation_messages) = Self::split_system_message(&request.messages);

        // Convert to Anthropic message format
        let anthropic_messages: Vec<AnthropicMessage> = conversation_messages
            .iter()
            .map(|msg| AnthropicMessage {
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "user".to_string(), // shouldn't happen after split
                },
                content: msg.content.clone(),
            })
            .collect();

        let api_request = AnthropicRequest {
            model: request.model.clone(),
            max_tokens: request.max_output_tokens,
            system: system_text,
            messages: anthropic_messages,
            temperature: Some(request.temperature),
            stop_sequences: request.stop_sequences.clone(),
        };

        let url = format!("{}/v1/messages", self.base_url);

        let http_response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2024-10-22")
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout { timeout_ms: 300_000 }
                } else {
                    ProviderError::Http(e)
                }
            })?;

        let status = http_response.status().as_u16();

        if status == 429 {
            // Rate limited — extract retry-after if available
            let retry_after = http_response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(|secs| secs * 1000)
                .unwrap_or(5000);

            return Err(ProviderError::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        let response_body = http_response
            .text()
            .await
            .map_err(ProviderError::Http)?;

        if status != 200 {
            // Try to parse structured error
            if let Ok(api_error) = serde_json::from_str::<AnthropicError>(&response_body) {
                if api_error.error.error_type == "authentication_error" {
                    return Err(ProviderError::AuthFailed(api_error.error.message));
                }
                return Err(ProviderError::ApiError {
                    status,
                    body: api_error.error.message,
                });
            }
            return Err(ProviderError::ApiError {
                status,
                body: if response_body.len() > 500 {
                    let truncated: String = response_body.chars().take(500).collect();
                    format!("{}...(truncated)", truncated)
                } else {
                    response_body
                },
            });
        }

        let api_response: AnthropicResponse =
            serde_json::from_str(&response_body).map_err(|e| {
                ProviderError::Deserialize(format!(
                    "Failed to parse Anthropic response: {e}"
                ))
            })?;

        // Extract text content from content blocks
        let content = api_response
            .content
            .iter()
            .filter_map(|block| {
                if block.block_type == "text" {
                    block.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        let stop_reason = match api_response.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("stop_sequence") => StopReason::StopSequence,
            Some("max_tokens") => StopReason::MaxTokens,
            _ => StopReason::Unknown,
        };

        Ok(LLMRawResponse {
            content,
            stop_reason,
            input_tokens: api_response.usage.input_tokens,
            output_tokens: api_response.usage.output_tokens,
            model: api_response.model,
            response_id: Some(api_response.id),
        })
    }

    /// Health check: send a minimal request to verify API connectivity.
    pub async fn health_check(&self) -> Result<(), ProviderError> {
        let url = format!("{}/v1/messages", self.base_url);

        let minimal_request = AnthropicRequest {
            model: self.spec.model.clone(),
            max_tokens: 1,
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "ping".to_string(),
            }],
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2024-10-22")
            .header("content-type", "application/json")
            .json(&minimal_request)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status().as_u16();
        if status == 200 || status == 400 {
            // 400 is fine for health check — means API is reachable
            Ok(())
        } else if status == 401 {
            Err(ProviderError::AuthFailed(
                "Invalid API key".to_string(),
            ))
        } else {
            Err(ProviderError::Unavailable(format!(
                "Anthropic API returned status {status}"
            )))
        }
    }

    pub fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    pub fn model(&self) -> &str {
        &self.spec.model
    }

    pub fn max_context_tokens(&self) -> u32 {
        self.spec.max_context_tokens
    }

    pub fn max_output_tokens(&self) -> u32 {
        self.spec.max_output_tokens
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Split system messages out of the message list.
    /// Anthropic API takes system as a separate top-level field.
    fn split_system_message(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
        let mut system_parts = Vec::new();
        let mut conversation = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system_parts.push(msg.content.as_str());
                }
                _ => {
                    conversation.push(msg);
                }
            }
        }

        let system_text = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        (system_text, conversation)
    }
}
