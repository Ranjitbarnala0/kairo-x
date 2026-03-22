//! Local LLM provider — connects to locally-hosted models via OpenAI-compatible API.
//!
//! Supports vLLM, Ollama (with OpenAI compat mode), llama.cpp server,
//! or any other local inference server that exposes an OpenAI-compatible endpoint.

use crate::call::{LLMRequest, MessageRole};
use crate::response::{LLMRawResponse, StopReason};
use super::{ProviderError, ProviderKind, ProviderSpec};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Uses same OpenAI-compatible format — local servers typically implement this
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LocalRequest {
    model: String,
    messages: Vec<LocalMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Serialize)]
struct LocalMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct LocalResponse {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    choices: Vec<LocalChoice>,
    #[serde(default)]
    usage: LocalUsage,
}

#[derive(Deserialize)]
struct LocalChoice {
    message: LocalResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct LocalResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct LocalUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LocalProvider {
    client: Client,
    spec: ProviderSpec,
    base_url: String,
}

impl LocalProvider {
    pub fn new(spec: ProviderSpec) -> Result<Self, ProviderError> {
        let base_url = spec
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8000".to_string());

        if !base_url.starts_with("https://")
            && !base_url.contains("localhost")
            && !base_url.contains("127.0.0.1")
        {
            tracing::warn!(
                url = %base_url,
                "Local provider using insecure HTTP to non-localhost URL"
            );
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(600)) // Local models can be slow
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(ProviderError::Http)?;

        Ok(Self {
            client,
            spec,
            base_url,
        })
    }

    pub async fn send(&self, request: &LLMRequest) -> Result<LLMRawResponse, ProviderError> {
        let messages: Vec<LocalMessage> = request
            .messages
            .iter()
            .map(|msg| LocalMessage {
                role: match msg.role {
                    MessageRole::System => "system".to_string(),
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect();

        let api_request = LocalRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_output_tokens,
            temperature: Some(request.temperature),
            stop: request.stop_sequences.clone(),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let http_response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout { timeout_ms: 600_000 }
                } else if e.is_connect() {
                    ProviderError::Unavailable(format!(
                        "Cannot connect to local server at {}. Is it running?",
                        self.base_url
                    ))
                } else {
                    ProviderError::Http(e)
                }
            })?;

        let status = http_response.status().as_u16();
        let response_body = http_response
            .text()
            .await
            .map_err(ProviderError::Http)?;

        if status != 200 {
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

        let api_response: LocalResponse =
            serde_json::from_str(&response_body).map_err(|e| {
                ProviderError::Deserialize(format!(
                    "Failed to parse local server response: {e}. Body: {}",
                    &response_body[..response_body.len().min(500)]
                ))
            })?;

        let choice = api_response
            .choices
            .first()
            .ok_or_else(|| ProviderError::Deserialize("No choices in response".to_string()))?;

        let content = choice.message.content.clone().unwrap_or_default();

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::Unknown,
        };

        // Local servers may not report accurate token counts.
        // If not provided, estimate from content length.
        let input_tokens = if api_response.usage.prompt_tokens > 0 {
            api_response.usage.prompt_tokens
        } else {
            request.estimated_input_tokens()
        };

        let output_tokens = if api_response.usage.completion_tokens > 0 {
            api_response.usage.completion_tokens
        } else {
            (content.len() as u32 / 4).max(1)
        };

        Ok(LLMRawResponse {
            content,
            stop_reason,
            input_tokens,
            output_tokens,
            model: if api_response.model.is_empty() {
                self.spec.model.clone()
            } else {
                api_response.model
            },
            response_id: if api_response.id.is_empty() {
                None
            } else {
                Some(api_response.id)
            },
        })
    }

    pub async fn health_check(&self) -> Result<(), ProviderError> {
        // Try the models endpoint first; fall back to a simple GET on base URL
        let url = format!("{}/v1/models", self.base_url);

        match self.client.get(&url).send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                if status == 200 {
                    Ok(())
                } else {
                    Err(ProviderError::Unavailable(format!(
                        "Local server returned status {status}"
                    )))
                }
            }
            Err(e) => {
                if e.is_connect() {
                    Err(ProviderError::Unavailable(format!(
                        "Cannot connect to local server at {}",
                        self.base_url
                    )))
                } else {
                    Err(ProviderError::Http(e))
                }
            }
        }
    }

    pub fn kind(&self) -> ProviderKind {
        ProviderKind::Local
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
}
