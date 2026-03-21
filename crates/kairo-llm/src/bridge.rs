//! LLM bridge — high-level orchestration of LLM calls with retry logic,
//! failover, rate limit handling, and exponential backoff.
//!
//! This is the single entry point for all LLM communication in KAIRO-X.

use crate::call::LLMRequest;
use crate::context_request;
use crate::providers::{ProviderError, ProviderManager, ProviderSpec, ProviderKind};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::openai::OpenAIProvider;
use crate::providers::local::LocalProvider;
use crate::response::{ContextRequest, LLMRawResponse};
use crate::truncation;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

// ---------------------------------------------------------------------------
// Bridge errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("All providers failed after retries")]
    AllProvidersFailed,

    #[error("Response truncated after {max_continuations} continuation attempts")]
    TruncationExhausted { max_continuations: u32 },

    #[error("Context request limit exceeded ({rounds} rounds)")]
    ContextRequestLimitExceeded { rounds: u32 },
}

// ---------------------------------------------------------------------------
// Bridge configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Maximum retries per provider before failover
    pub max_retries_per_provider: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub backoff_base_ms: u64,
    /// Maximum backoff delay (milliseconds)
    pub backoff_max_ms: u64,
    /// Consecutive failures before switching to fallback provider
    pub failover_threshold: u32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            max_retries_per_provider: 3,
            backoff_base_ms: 1000,
            backoff_max_ms: 30_000,
            failover_threshold: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// LLM Bridge
// ---------------------------------------------------------------------------

/// The LLM Bridge is the single entry point for all LLM communication.
///
/// It handles:
/// - Provider selection (primary/fallback/audit)
/// - Retry with exponential backoff
/// - Rate limit handling
/// - Automatic failover
/// - Truncation detection and continuation
/// - NEED_CONTEXT detection (signals back to caller)
///
/// Provider instances are lazily created and cached so that the underlying
/// `reqwest::Client` (and its HTTP connection pool) is reused across calls.
/// Creating a new `Client` per request would bypass connection pooling,
/// adding ~50-200ms of TLS handshake overhead to every LLM call.
pub struct LLMBridge {
    manager: ProviderManager,
    config: BridgeConfig,
    /// Cached provider instances.  Protected by `Mutex<Option<T>>` and
    /// cloned out before the async `.send()` call so the lock is not
    /// held across an await point.
    cached_anthropic: Mutex<Option<AnthropicProvider>>,
    cached_openai: Mutex<Option<OpenAIProvider>>,
    cached_local: Mutex<Option<LocalProvider>>,
}

/// Result of a bridge call, which may include context requests
/// that the caller needs to fulfill.
pub enum BridgeCallResult {
    /// Successful response with content.
    Success(LLMRawResponse),
    /// LLM is requesting additional context before it can proceed.
    NeedsContext {
        /// The context requests parsed from the response
        requests: Vec<ContextRequest>,
        /// The raw response (caller may need it for context)
        raw_response: LLMRawResponse,
    },
    /// Response was truncated beyond recovery — caller should decompose the node.
    NeedsDecomposition(LLMRawResponse),
}

impl LLMBridge {
    pub fn new(manager: ProviderManager, config: BridgeConfig) -> Self {
        Self {
            manager,
            config,
            cached_anthropic: Mutex::new(None),
            cached_openai: Mutex::new(None),
            cached_local: Mutex::new(None),
        }
    }

    /// Get the model name of the currently active provider.
    pub fn active_model(&self) -> &str {
        &self.manager.active_spec().model
    }

    /// Get the max output tokens of the currently active provider.
    pub fn active_max_output_tokens(&self) -> u32 {
        self.manager.active_spec().max_output_tokens
    }

    /// Send an LLM request with full retry, failover, and truncation handling.
    ///
    /// Returns the final (possibly concatenated) response, or signals that
    /// context is needed or the node should be decomposed.
    pub async fn call(&self, mut request: LLMRequest) -> Result<BridgeCallResult, BridgeError> {
        let mut full_response = self.call_with_retries(&request).await?;

        // Handle truncation: continue in same session up to MAX_CONTINUATIONS
        let mut continuation_count = 0;
        while truncation::is_truncated(
            &full_response.content,
            full_response.stop_reason,
            request.call_type,
        ) && continuation_count < truncation::MAX_CONTINUATIONS
        {
            continuation_count += 1;
            tracing::info!(
                continuation = continuation_count,
                "Response truncated, sending continuation request"
            );

            // Build continuation request: add assistant response + continuation prompt
            request
                .messages
                .push(crate::call::Message::assistant(full_response.content.clone()));
            request
                .messages
                .push(crate::call::Message::user(
                    truncation::continuation_prompt().to_string(),
                ));

            let continuation = self.call_with_retries(&request).await?;

            // Concatenate responses
            full_response.content = truncation::concatenate_responses(
                &full_response.content,
                &continuation.content,
            );
            full_response.output_tokens += continuation.output_tokens;
            full_response.input_tokens += continuation.input_tokens; // accumulate input tokens
            full_response.stop_reason = continuation.stop_reason;
        }

        // If still truncated after max continuations, signal decomposition needed
        if truncation::is_truncated(
            &full_response.content,
            full_response.stop_reason,
            request.call_type,
        ) {
            return Ok(BridgeCallResult::NeedsDecomposition(full_response));
        }

        // Check for NEED_CONTEXT markers
        if let Some(requests) = context_request::parse_context_request(&full_response.content) {
            return Ok(BridgeCallResult::NeedsContext {
                requests,
                raw_response: full_response,
            });
        }

        Ok(BridgeCallResult::Success(full_response))
    }

    /// Call for audit — uses the audit provider if configured.
    ///
    /// If an audit override provider is configured (and differs from the active
    /// provider), dispatch directly to that provider. The request is expected to
    /// already carry temperature 0.3 from `LLMCallType::Audit`.
    pub async fn call_audit(
        &self,
        request: LLMRequest,
    ) -> Result<BridgeCallResult, BridgeError> {
        let audit_spec = self.manager.audit_spec();
        let active_spec = self.manager.active_spec();

        // If the audit spec is a different provider/model, dispatch directly to it
        if audit_spec.kind != active_spec.kind || audit_spec.model != active_spec.model {
            let response = self.send_to_provider(audit_spec, &request).await?;
            return Ok(BridgeCallResult::Success(response));
        }

        // Otherwise, fall through to normal call (with retry/failover)
        self.call(request).await
    }

    // -----------------------------------------------------------------------
    // Internal: retry logic with exponential backoff
    // -----------------------------------------------------------------------

    async fn call_with_retries(
        &self,
        request: &LLMRequest,
    ) -> Result<LLMRawResponse, BridgeError> {
        let mut last_error: Option<ProviderError> = None;
        let mut attempt = 0;

        loop {
            if attempt >= self.config.max_retries_per_provider {
                // Check if we can failover
                if self.manager.record_failure() {
                    tracing::warn!(
                        "Primary provider failed {} times, switching to fallback",
                        attempt
                    );
                    attempt = 0; // Reset attempts for fallback
                    continue;
                }

                return Err(last_error
                    .map(BridgeError::Provider)
                    .unwrap_or(BridgeError::AllProvidersFailed));
            }

            let spec = self.manager.active_spec();

            match self.send_to_provider(spec, request).await {
                Ok(response) => {
                    self.manager.record_success();
                    return Ok(response);
                }
                Err(ProviderError::RateLimited { retry_after_ms }) => {
                    tracing::warn!(
                        retry_after_ms,
                        "Rate limited, waiting before retry"
                    );
                    sleep(Duration::from_millis(retry_after_ms)).await;
                    attempt += 1;
                    last_error = Some(ProviderError::RateLimited { retry_after_ms });
                }
                Err(e) => {
                    let backoff = self.calculate_backoff(attempt);
                    tracing::warn!(
                        attempt,
                        backoff_ms = backoff,
                        error = %e,
                        "LLM call failed, retrying"
                    );
                    sleep(Duration::from_millis(backoff)).await;
                    attempt += 1;
                    last_error = Some(e);
                }
            }
        }
    }

    /// Dispatch to the appropriate provider implementation based on the spec.
    ///
    /// Provider instances are cached so the underlying `reqwest::Client`
    /// (and its connection pool) is reused across calls.  The provider is
    /// cloned out of the Mutex before the async `.send()` call so the lock
    /// is not held across an await point.  `Clone` on providers is cheap
    /// because `reqwest::Client` is internally `Arc`-wrapped.
    async fn send_to_provider(
        &self,
        spec: &ProviderSpec,
        request: &LLMRequest,
    ) -> Result<LLMRawResponse, ProviderError> {
        match spec.kind {
            ProviderKind::Anthropic => {
                let provider = {
                    let mut guard = self.cached_anthropic.lock().unwrap_or_else(|p| p.into_inner());
                    if guard.is_none() {
                        *guard = Some(AnthropicProvider::new(spec.clone())?);
                    }
                    guard.as_ref().unwrap().clone()
                };
                provider.send(request).await
            }
            ProviderKind::OpenAI => {
                let provider = {
                    let mut guard = self.cached_openai.lock().unwrap_or_else(|p| p.into_inner());
                    if guard.is_none() {
                        *guard = Some(OpenAIProvider::new(spec.clone())?);
                    }
                    guard.as_ref().unwrap().clone()
                };
                provider.send(request).await
            }
            ProviderKind::Local => {
                let provider = {
                    let mut guard = self.cached_local.lock().unwrap_or_else(|p| p.into_inner());
                    if guard.is_none() {
                        *guard = Some(LocalProvider::new(spec.clone())?);
                    }
                    guard.as_ref().unwrap().clone()
                };
                provider.send(request).await
            }
        }
    }

    /// Exponential backoff with jitter.
    fn calculate_backoff(&self, attempt: u32) -> u64 {
        let base = self.config.backoff_base_ms;
        let exponential = base.saturating_mul(2u64.pow(attempt));
        let capped = exponential.min(self.config.backoff_max_ms);
        // Add 0-50% jitter (always positive to avoid underflow)
        let jitter_range = capped / 2;
        let jitter = if jitter_range > 0 {
            rand::random::<u64>() % jitter_range
        } else {
            0
        };
        capped.saturating_add(jitter)
    }
}
