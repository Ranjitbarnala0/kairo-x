//! Configuration loading and merging.
//!
//! Reads the built-in `default_config.toml`, then overlays any project-local
//! `.kairo/config.toml`, and finally merges CLI arguments on top. The result
//! is a [`RuntimeConfig`] ready to pass to the runtime.

use anyhow::{Context, Result};
use kairo_core::runtime::RuntimeConfig;
use kairo_core::session::token_tracker::CostMode;
use kairo_llm::bridge::BridgeConfig;
use kairo_llm::providers::{ProviderConfig, ProviderKind, ProviderSpec};
use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Intermediate TOML representation
// ---------------------------------------------------------------------------

/// Top-level shape of the TOML config file.
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct RawConfig {
    general: GeneralSection,
    provider: ProviderSection,
    persistence: PersistenceSection,
    budget: BudgetSection,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct GeneralSection {
    cost_mode: String,
    max_parallel_tracks: u8,
}

impl Default for GeneralSection {
    fn default() -> Self {
        Self {
            cost_mode: "balanced".to_string(),
            max_parallel_tracks: 3,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct ProviderSection {
    primary: Option<RawProviderSpec>,
    fallback: Option<RawProviderSpec>,
    audit_override: Option<RawProviderSpec>,
    retry: RetrySection,
}

#[derive(Debug, Deserialize, Clone)]
struct RawProviderSpec {
    kind: String,
    model: String,
    api_key_env: String,
    max_context_tokens: u32,
    max_output_tokens: u32,
    cost_per_input_mtok: u32,
    cost_per_output_mtok: u32,
    base_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct RetrySection {
    max_retries: u32,
    backoff_base_ms: u64,
    backoff_max_ms: u64,
    failover_threshold: u32,
}

impl Default for RetrySection {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_base_ms: 1000,
            backoff_max_ms: 30_000,
            failover_threshold: 3,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct PersistenceSection {
    checkpoint_interval: u32,
    max_checkpoints: usize,
}

impl Default for PersistenceSection {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10,
            max_checkpoints: 20,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BudgetSection {
    /// Token budget limit (total input+output tokens). 0 or absent = unlimited.
    token_limit: Option<u64>,
    /// Cost budget limit in microdollars. 0 or absent = unlimited.
    cost_limit_microdollars: Option<u64>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn parse_cost_mode(s: &str) -> CostMode {
    match s.to_lowercase().as_str() {
        "thorough" => CostMode::Thorough,
        "balanced" => CostMode::Balanced,
        "efficient" => CostMode::Efficient,
        other => {
            tracing::warn!(value = %other, "Unknown cost_mode, defaulting to Balanced");
            CostMode::Balanced
        }
    }
}

fn parse_provider_kind(s: &str) -> ProviderKind {
    match s.to_lowercase().as_str() {
        "anthropic" => ProviderKind::Anthropic,
        "openai" => ProviderKind::OpenAI,
        "local" => ProviderKind::Local,
        other => {
            tracing::warn!(value = %other, "Unknown provider kind, defaulting to Anthropic");
            ProviderKind::Anthropic
        }
    }
}

fn raw_to_provider_spec(raw: &RawProviderSpec) -> ProviderSpec {
    ProviderSpec {
        kind: parse_provider_kind(&raw.kind),
        model: raw.model.clone(),
        api_key_env: raw.api_key_env.clone(),
        max_context_tokens: raw.max_context_tokens,
        max_output_tokens: raw.max_output_tokens,
        cost_per_input_mtok: raw.cost_per_input_mtok,
        cost_per_output_mtok: raw.cost_per_output_mtok,
        base_url: raw.base_url.clone(),
    }
}

/// Build the default primary provider spec when no config file is present.
fn default_primary_spec(provider_kind: ProviderKind, model: &str) -> ProviderSpec {
    let (api_key_env, max_ctx, max_out, cost_in, cost_out) = match provider_kind {
        ProviderKind::Anthropic => ("ANTHROPIC_API_KEY", 200_000, 8192, 3_000, 15_000),
        ProviderKind::OpenAI => ("OPENAI_API_KEY", 128_000, 4096, 2_500, 10_000),
        ProviderKind::Local => ("LOCAL_API_KEY", 128_000, 4096, 0, 0),
    };
    ProviderSpec {
        kind: provider_kind,
        model: model.to_string(),
        api_key_env: api_key_env.to_string(),
        max_context_tokens: max_ctx,
        max_output_tokens: max_out,
        cost_per_input_mtok: cost_in,
        cost_per_output_mtok: cost_out,
        base_url: None,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// CLI overrides that are layered on top of the config file values.
pub struct CliOverrides {
    pub cost_mode: CostMode,
    /// Whether cost_mode was explicitly set on the CLI (vs. using the default).
    pub cost_mode_explicit: bool,
    pub parallel: u8,
    /// Whether parallel was explicitly set on the CLI.
    pub parallel_explicit: bool,
    pub budget: Option<u64>,
    pub provider_kind: ProviderKind,
    pub model: String,
}

/// Load configuration from disk (default + project-local), then merge CLI
/// overrides. Returns a fully-resolved [`RuntimeConfig`].
pub fn load_config(project_root: &Path, overrides: &CliOverrides) -> Result<RuntimeConfig> {
    // 1. Try the built-in default config shipped alongside the binary.
    //    We embed it at compile time so the binary is self-contained.
    let default_toml = include_str!("../../../config/default_config.toml");
    let mut raw: RawConfig =
        toml::from_str(default_toml).context("failed to parse built-in default_config.toml")?;

    // 2. Overlay project-local config if it exists.
    let project_config_path = project_root.join(".kairo/config.toml");
    if project_config_path.exists() {
        let project_toml = std::fs::read_to_string(&project_config_path)
            .with_context(|| format!("failed to read {}", project_config_path.display()))?;
        let project_raw: RawConfig = toml::from_str(&project_toml)
            .with_context(|| format!("failed to parse {}", project_config_path.display()))?;
        merge_raw_config(&mut raw, project_raw);
    }

    // 3. Build the ProviderConfig — start from the config file, then apply
    //    CLI overrides for kind/model on the primary spec.
    let mut primary = match &raw.provider.primary {
        Some(p) => raw_to_provider_spec(p),
        None => default_primary_spec(overrides.provider_kind, &overrides.model),
    };

    // CLI flags always win for provider kind and model.
    primary.kind = overrides.provider_kind;
    primary.model = overrides.model.clone();
    // Adjust api_key_env to match the CLI-selected provider kind.
    match overrides.provider_kind {
        ProviderKind::Anthropic => primary.api_key_env = "ANTHROPIC_API_KEY".to_string(),
        ProviderKind::OpenAI => primary.api_key_env = "OPENAI_API_KEY".to_string(),
        ProviderKind::Local => primary.api_key_env = "LOCAL_API_KEY".to_string(),
    }

    let fallback = raw.provider.fallback.as_ref().map(raw_to_provider_spec);
    let audit_override = raw.provider.audit_override.as_ref().map(raw_to_provider_spec);

    let provider_config = ProviderConfig {
        primary,
        fallback,
        audit_override,
    };

    let bridge_config = BridgeConfig {
        max_retries_per_provider: raw.provider.retry.max_retries,
        backoff_base_ms: raw.provider.retry.backoff_base_ms,
        backoff_max_ms: raw.provider.retry.backoff_max_ms,
        failover_threshold: raw.provider.retry.failover_threshold,
    };

    // 4. Resolve final values: CLI overrides > project config > default config.
    // For cost_mode: CLI default is Balanced, so if user didn't explicitly set it,
    // use the config file value. If user did set it, CLI wins.
    let config_cost_mode = parse_cost_mode(&raw.general.cost_mode);
    let cost_mode = if overrides.cost_mode_explicit {
        overrides.cost_mode
    } else {
        config_cost_mode
    };
    let max_parallel = if overrides.parallel_explicit {
        overrides.parallel
    } else {
        raw.general.max_parallel_tracks
    };
    let token_budget = overrides.budget.or(raw.budget.token_limit);
    let cost_limit_microdollars = raw.budget.cost_limit_microdollars;

    Ok(RuntimeConfig {
        project_root: project_root.to_path_buf(),
        provider_config,
        cost_mode,
        max_parallel_tracks: max_parallel,
        checkpoint_interval: raw.persistence.checkpoint_interval,
        max_checkpoints: raw.persistence.max_checkpoints,
        token_budget,
        cost_limit_microdollars,
        bridge_config,
    })
}

/// Merge `overlay` on top of `base`. The overlay (project config) always
/// wins — if a user put it in their config file, they meant it. The
/// previous approach of comparing against hardcoded defaults was fragile
/// because renaming a default value would silently break the merge.
fn merge_raw_config(base: &mut RawConfig, overlay: RawConfig) {
    // General section — always take overlay values
    base.general = overlay.general;

    // Provider: overlay sections replace base if present
    if overlay.provider.primary.is_some() {
        base.provider.primary = overlay.provider.primary;
    }
    if overlay.provider.fallback.is_some() {
        base.provider.fallback = overlay.provider.fallback;
    }
    if overlay.provider.audit_override.is_some() {
        base.provider.audit_override = overlay.provider.audit_override;
    }
    // Retry config — always take overlay
    base.provider.retry = overlay.provider.retry;

    // Persistence — always take overlay
    base.persistence = overlay.persistence;

    // Budget — always take overlay
    base.budget = overlay.budget;
}
