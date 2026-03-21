pub mod anthropic;
pub mod openai;
pub mod local;

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Provider errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API returned error status {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("Response deserialization failed: {0}")]
    Deserialize(String),

    #[error("Connection timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Provider unavailable: {0}")]
    Unavailable(String),
}

// ---------------------------------------------------------------------------
// Provider kind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    Anthropic,
    OpenAI,
    Local,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anthropic => write!(f, "Anthropic"),
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Local => write!(f, "Local"),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider specification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub kind: ProviderKind,
    pub model: String,
    pub api_key_env: String,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
    /// Microdollars per million input tokens
    pub cost_per_input_mtok: u32,
    /// Microdollars per million output tokens
    pub cost_per_output_mtok: u32,
    /// Base URL override (for local or custom endpoints)
    pub base_url: Option<String>,
}

// ---------------------------------------------------------------------------
// Provider configuration with failover
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub primary: ProviderSpec,
    pub fallback: Option<ProviderSpec>,
    /// Optional different model for audit calls (cross-model defense)
    pub audit_override: Option<ProviderSpec>,
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

// Provider implementations (AnthropicProvider, OpenAIProvider, LocalProvider)
// each expose the same public API surface:
//   async fn send(&self, request: &LLMRequest) -> Result<LLMRawResponse, ProviderError>
//   async fn health_check(&self) -> Result<(), ProviderError>
//   fn kind(&self) -> ProviderKind
//   fn model(&self) -> &str
//   fn max_context_tokens(&self) -> u32
//   fn max_output_tokens(&self) -> u32
//
// We use concrete dispatch in the bridge (match on ProviderKind) rather than
// dynamic dispatch via trait objects — avoids async-trait overhead and keeps
// the code simpler for 3 known provider types.

// ---------------------------------------------------------------------------
// Failover manager
// ---------------------------------------------------------------------------

/// Manages provider failover: tracks consecutive failures on primary,
/// switches to fallback after threshold.
pub struct ProviderManager {
    pub config: ProviderConfig,
    /// Mutex protects the failover state so that the check-and-switch is atomic.
    failover_state: Mutex<FailoverState>,
}

/// Interior state protected by a mutex for atomic failover decisions.
#[derive(Debug)]
struct FailoverState {
    consecutive_failures: u32,
    using_fallback: bool,
    failover_threshold: u32,
}

impl ProviderManager {
    pub fn new(config: ProviderConfig, failover_threshold: u32) -> Self {
        Self {
            config,
            failover_state: Mutex::new(FailoverState {
                consecutive_failures: 0,
                using_fallback: false,
                failover_threshold,
            }),
        }
    }

    /// Acquire the failover state lock, recovering from poisoning.
    ///
    /// A poisoned mutex means another thread panicked while holding the lock.
    /// The failover state is simple enough that the data is still valid after
    /// a panic, so we recover by accepting the poisoned guard.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, FailoverState> {
        self.failover_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record a successful call — resets failure counter.
    pub fn record_success(&self) {
        let mut state = self.lock_state();
        state.consecutive_failures = 0;
    }

    /// Record a failed call. Returns true if we switched to fallback.
    ///
    /// The failure count increment and the fallback decision are performed
    /// atomically under a single lock acquisition.
    pub fn record_failure(&self) -> bool {
        let mut state = self.lock_state();
        state.consecutive_failures += 1;
        if state.consecutive_failures >= state.failover_threshold
            && self.config.fallback.is_some()
            && !state.using_fallback
        {
            state.using_fallback = true;
            true
        } else {
            false
        }
    }

    /// Check if we're currently using the fallback provider.
    pub fn is_using_fallback(&self) -> bool {
        let state = self.lock_state();
        state.using_fallback
    }

    /// Get the currently active provider spec.
    pub fn active_spec(&self) -> &ProviderSpec {
        if self.is_using_fallback() {
            self.config
                .fallback
                .as_ref()
                .unwrap_or(&self.config.primary)
        } else {
            &self.config.primary
        }
    }

    /// Get the provider spec for audit calls (may be different model).
    pub fn audit_spec(&self) -> &ProviderSpec {
        self.config
            .audit_override
            .as_ref()
            .unwrap_or_else(|| self.active_spec())
    }

    /// Reset to primary provider (e.g., after a cooldown period).
    pub fn reset_to_primary(&self) {
        let mut state = self.lock_state();
        state.consecutive_failures = 0;
        state.using_fallback = false;
    }
}
