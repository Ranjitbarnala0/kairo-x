//! OpenAI-compatible API provider implementation.
//!
//! Supports OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint
//! (vLLM, Ollama with OpenAI compat, etc.)

use crate::call::{LLMRequest, MessageRole};
use crate::response::{LLMRawResponse, StopReason};
use super::{ProviderError, ProviderKind, ProviderSpec};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// OpenAI API types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAIError {
    error: OpenAIErrorDetail,
}

#[derive(Deserialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OpenAIProvider {
    client: Client,
    spec: ProviderSpec,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
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
            .unwrap_or_else(|| "https://api.openai.com".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
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

    pub async fn send(&self, request: &LLMRequest) -> Result<LLMRawResponse, ProviderError> {
        let openai_messages: Vec<OpenAIMessage> = request
            .messages
            .iter()
            .map(|msg| OpenAIMessage {
                role: match msg.role {
                    MessageRole::System => "system".to_string(),
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect();

        let api_request = OpenAIRequest {
            model: request.model.clone(),
            messages: openai_messages,
            max_tokens: request.max_output_tokens,
            temperature: if request.temperature == 0.0 {
                Some(0.0)
            } else {
                Some(request.temperature)
            },
            stop: request.stop_sequences.clone(),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let http_response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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
            if let Ok(api_error) = serde_json::from_str::<OpenAIError>(&response_body) {
                if status == 401 {
                    return Err(ProviderError::AuthFailed(api_error.error.message));
                }
                return Err(ProviderError::ApiError {
                    status,
                    body: api_error.error.message,
                });
            }
            return Err(ProviderError::ApiError {
                status,
                body: response_body,
            });
        }

        let api_response: OpenAIResponse =
            serde_json::from_str(&response_body).map_err(|e| {
                ProviderError::Deserialize(format!(
                    "Failed to parse OpenAI response: {e}"
                ))
            })?;

        let choice = api_response
            .choices
            .first()
            .ok_or_else(|| ProviderError::Deserialize("No choices in response".to_string()))?;

        let content = choice
            .message
            .content
            .clone()
            .unwrap_or_default();

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::Unknown,
        };

        Ok(LLMRawResponse {
            content,
            stop_reason,
            input_tokens: api_response.usage.prompt_tokens,
            output_tokens: api_response.usage.completion_tokens,
            model: api_response.model,
            response_id: Some(api_response.id),
        })
    }

    pub async fn health_check(&self) -> Result<(), ProviderError> {
        let url = format!("{}/v1/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status().as_u16();
        if status == 200 {
            Ok(())
        } else if status == 401 {
            Err(ProviderError::AuthFailed("Invalid API key".to_string()))
        } else {
            Err(ProviderError::Unavailable(format!(
                "OpenAI API returned status {status}"
            )))
        }
    }

    pub fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAI
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
