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

#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub kind: ProviderKind,
    pub model: String,
    #[serde(skip_serializing)]
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

impl std::fmt::Debug for ProviderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSpec")
            .field("kind", &self.kind)
            .field("model", &self.model)
            .field("api_key_env", &"[REDACTED]")
            .field("max_context_tokens", &self.max_context_tokens)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl ProviderSpec {
    /// Validate that the spec contains sane values.
    /// Returns an error string describing what is wrong, or `Ok(())`.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_context_tokens == 0 {
            return Err("max_context_tokens must be > 0".to_string());
        }
        if self.max_output_tokens == 0 {
            return Err("max_output_tokens must be > 0".to_string());
        }
        Ok(())
    }
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

/// Cooldown period (in milliseconds) before we attempt to recover the
/// primary provider after a failover. 5 minutes.
const PRIMARY_RECOVERY_COOLDOWN_MS: i64 = 300_000;

/// Number of consecutive successful calls on the fallback provider required
/// before we consider switching back to primary.
const RECOVERY_SUCCESS_THRESHOLD: u32 = 10;

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
    /// How many successful calls have been made while on the fallback provider.
    /// Once this reaches [`RECOVERY_SUCCESS_THRESHOLD`], we attempt to switch
    /// back to primary.
    fallback_success_count: u32,
    /// Epoch-millis timestamp of the last time we attempted (or switched back
    /// to) the primary provider. Used for cooldown gating.
    last_primary_attempt_ms: i64,
}

impl ProviderManager {
    pub fn new(config: ProviderConfig, failover_threshold: u32) -> Self {
        Self {
            config,
            failover_state: Mutex::new(FailoverState {
                consecutive_failures: 0,
                using_fallback: false,
                failover_threshold,
                fallback_success_count: 0,
                last_primary_attempt_ms: 0,
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
    /// While on fallback, also increments `fallback_success_count` so the
    /// recovery logic knows when conditions are stable enough to try primary.
    pub fn record_success(&self) {
        let mut state = self.lock_state();
        state.consecutive_failures = 0;
        if state.using_fallback {
            state.fallback_success_count += 1;
        }
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
    ///
    /// # Deprecated
    ///
    /// This method is **TOCTOU-unsafe**: it releases the failover lock before
    /// the caller reads the returned reference, so the active provider can
    /// change between the check and the use. Prefer [`active_spec_cloned`]
    /// for all new call-sites — it returns a snapshot taken while the lock
    /// is held.
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

    /// Return a cloned snapshot of the active spec, determined while holding
    /// the failover lock. This avoids the TOCTOU race in [`active_spec`]
    /// where the fallback state could change between the `is_using_fallback()`
    /// check and the field access.
    pub fn active_spec_cloned(&self) -> ProviderSpec {
        let state = self.lock_state();
        if state.using_fallback {
            self.config
                .fallback
                .clone()
                .unwrap_or_else(|| self.config.primary.clone())
        } else {
            self.config.primary.clone()
        }
    }

    /// Get the provider spec for audit calls (may be different model).
    pub fn audit_spec(&self) -> &ProviderSpec {
        self.config
            .audit_override
            .as_ref()
            .unwrap_or_else(|| self.active_spec())
    }

    /// Hard-reset to primary provider (e.g., after a cooldown period or
    /// operator intervention). Clears all failover and recovery counters.
    pub fn reset_to_primary(&self) {
        let mut state = self.lock_state();
        state.consecutive_failures = 0;
        state.using_fallback = false;
        state.fallback_success_count = 0;
        state.last_primary_attempt_ms = chrono::Utc::now().timestamp_millis();
    }

    // -----------------------------------------------------------------------
    // Primary recovery
    // -----------------------------------------------------------------------

    /// Returns `true` if we are on fallback and enough time has elapsed since
    /// the last primary attempt to justify probing primary again.
    ///
    /// Callers should gate a single primary-probe request on this returning
    /// `true`. If the probe succeeds, call [`try_reset_to_primary`].
    pub fn should_try_primary_recovery(&self) -> bool {
        let state = self.lock_state();
        if !state.using_fallback {
            return false;
        }
        let now = chrono::Utc::now().timestamp_millis();
        now - state.last_primary_attempt_ms >= PRIMARY_RECOVERY_COOLDOWN_MS
    }

    /// Attempt to switch back to the primary provider.
    ///
    /// Succeeds only when we are on fallback **and** the fallback has
    /// accumulated at least [`RECOVERY_SUCCESS_THRESHOLD`] consecutive
    /// successful calls — indicating stable connectivity.
    ///
    /// Returns `true` if the switch happened.
    pub fn try_reset_to_primary(&self) -> bool {
        let mut state = self.lock_state();
        if state.using_fallback
            && state.fallback_success_count >= RECOVERY_SUCCESS_THRESHOLD
        {
            state.using_fallback = false;
            state.consecutive_failures = 0;
            state.fallback_success_count = 0;
            state.last_primary_attempt_ms = chrono::Utc::now().timestamp_millis();
            tracing::info!(
                "Recovered to primary provider after {} successful fallback calls",
                RECOVERY_SUCCESS_THRESHOLD,
            );
            true
        } else {
            false
        }
    }
}
