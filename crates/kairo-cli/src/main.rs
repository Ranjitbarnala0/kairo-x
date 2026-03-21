//! KAIRO-X CLI — the command-line interface for the KAIRO-X enforcement agent.
//!
//! Provides `kairo init`, `kairo run`, `kairo status`, and `kairo resume`.

mod commands;
mod config;
mod display;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CLI argument definitions
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "kairo",
    about = "KAIRO-X — autonomous code enforcement agent",
    version,
    propagate_version = true
)]
struct Cli {
    /// Project root directory (defaults to current directory)
    #[arg(long, global = true, default_value = ".")]
    project_dir: PathBuf,

    /// Enable verbose logging (repeat for more: -v, -vv, -vvv)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fingerprint the project and display detected info (language, framework, commands).
    /// Creates the .kairo/ directory.
    Init,

    /// Run a task described by a string or spec file.
    Run {
        /// Task description or path to a spec file.
        task: String,

        /// Cost mode: thorough, balanced, or efficient. If omitted, uses config file value.
        #[arg(long, value_enum)]
        cost_mode: Option<CostModeArg>,

        /// Maximum parallel execution tracks. If omitted, uses config file value.
        #[arg(long)]
        parallel: Option<u8>,

        /// Optional token budget limit.
        #[arg(long)]
        budget: Option<u64>,

        /// Resume from the latest checkpoint instead of starting fresh.
        #[arg(long, default_value_t = false)]
        resume: bool,

        /// LLM provider to use.
        #[arg(long, value_enum, default_value_t = ProviderArg::Anthropic)]
        provider: ProviderArg,

        /// Model name to use.
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,
    },

    /// Show status of the latest in-progress or completed task.
    Status,

    /// Resume execution from the latest checkpoint.
    Resume,
}

// ---------------------------------------------------------------------------
// Clap value-enum mirrors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CostModeArg {
    Thorough,
    Balanced,
    Efficient,
}

impl From<CostModeArg> for kairo_core::session::token_tracker::CostMode {
    fn from(arg: CostModeArg) -> Self {
        match arg {
            CostModeArg::Thorough => Self::Thorough,
            CostModeArg::Balanced => Self::Balanced,
            CostModeArg::Efficient => Self::Efficient,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArg {
    Anthropic,
    Openai,
    Local,
}

impl From<ProviderArg> for kairo_llm::providers::ProviderKind {
    fn from(arg: ProviderArg) -> Self {
        match arg {
            ProviderArg::Anthropic => Self::Anthropic,
            ProviderArg::Openai => Self::OpenAI,
            ProviderArg::Local => Self::Local,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialise tracing-subscriber. Level depends on verbosity flag.
    let filter = match cli.verbose {
        0 => "kairo=info,warn",
        1 => "kairo=debug,info",
        2 => "kairo=trace,debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    // Canonicalize project root (resolve `.` etc.)
    let project_root = std::fs::canonicalize(&cli.project_dir).unwrap_or_else(|_| {
        std::env::current_dir().expect("cannot determine current directory")
    });

    match cli.command {
        Commands::Init => {
            commands::handle_init(&project_root).await?;
        }
        Commands::Run {
            task,
            cost_mode,
            parallel,
            budget,
            resume,
            provider,
            model,
        } => {
            commands::handle_run(
                &project_root,
                &task,
                cost_mode.map(|c| c.into()),
                parallel,
                budget,
                resume,
                provider.into(),
                &model,
            )
            .await?;
        }
        Commands::Status => {
            commands::handle_status(&project_root).await?;
        }
        Commands::Resume => {
            commands::handle_resume(&project_root).await?;
        }
    }

    Ok(())
}
