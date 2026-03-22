//! Display formatting for CLI output.
//!
//! Provides colorized, human-friendly output for fingerprint results,
//! task progress, status summaries, and completion summaries.
//! Uses ANSI escape codes directly to avoid a dependency on a color crate.

use kairo_core::arena::query::StatusSummary;
use kairo_core::fingerprint::ProjectFingerprint;
use kairo_core::persistence::CheckpointMeta;
use kairo_core::runtime::TaskCompletionSummary;
use kairo_core::session::token_tracker::TokenTracker;

// ---------------------------------------------------------------------------
// ANSI color helpers
// ---------------------------------------------------------------------------

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const WHITE: &str = "\x1b[37m";

use std::io::IsTerminal;

static USE_COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        if std::env::var("NO_COLOR").is_ok() {
            return false;
        }
        std::io::stdout().is_terminal()
    })
}

/// Wrap `text` in the given ANSI color if colors are enabled.
fn paint(color: &str, text: &str) -> String {
    if use_color() {
        format!("{color}{text}{RESET}")
    } else {
        text.to_string()
    }
}

fn bold(text: &str) -> String {
    paint(BOLD, text)
}

// ---------------------------------------------------------------------------
// Fingerprint display
// ---------------------------------------------------------------------------

/// Print a colorized table of the project fingerprint.
pub fn print_fingerprint(fp: &ProjectFingerprint) {
    let color = use_color();

    println!();
    println!(
        "{}",
        if color {
            format!("{BOLD}{CYAN}KAIRO-X Project Fingerprint{RESET}")
        } else {
            "KAIRO-X Project Fingerprint".to_string()
        }
    );
    println!("{}", paint(DIM, &"=".repeat(40)));
    println!();

    // Language
    print_row(
        "Language",
        fp.primary_language.display_name(),
        GREEN,
    );

    // All detected languages
    if fp.languages.len() > 1 {
        let langs: Vec<String> = fp
            .languages
            .iter()
            .map(|(lang, count)| format!("{} ({} files)", lang.display_name(), count))
            .collect();
        print_row("All languages", &langs.join(", "), DIM);
    }

    // Package manager
    if let Some(pm) = &fp.package_manager {
        print_row("Package manager", pm.display_name(), CYAN);
    }

    // Framework
    if let Some(fw) = &fp.framework {
        print_row("Framework", fw, MAGENTA);
    }

    // Monorepo
    if fp.monorepo {
        print_row("Monorepo", "yes", YELLOW);
    }

    println!();
    println!("{}", paint(DIM, &"-".repeat(40)));
    println!();

    // Commands table
    print_command_row("Build", &fp.build_command);
    print_command_row("Test", &fp.test_command);
    print_command_row("Lint", &fp.lint_command);
    print_command_row("Typecheck", &fp.typecheck_command);

    println!();
    println!("{}", paint(DIM, &"-".repeat(40)));
    println!();

    // Source and test roots
    if let Some(src) = &fp.source_root {
        print_row("Source root", &src.display().to_string(), WHITE);
    }
    if let Some(test) = &fp.test_root {
        print_row("Test root", &test.display().to_string(), WHITE);
    }

    // Verification capability
    if fp.can_verify_deterministically() {
        println!();
        println!(
            "  {} Deterministic verification is available.",
            paint(GREEN, "[ok]")
        );
    } else {
        println!();
        println!(
            "  {} No build/test commands detected. Verification will rely on LLM audit only.",
            paint(YELLOW, "[warn]")
        );
    }

    println!();
}

fn print_row(label: &str, value: &str, color: &str) {
    if use_color() {
        println!("  {BOLD}{label:<18}{RESET} {color}{value}{RESET}");
    } else {
        println!("  {label:<18} {value}");
    }
}

fn print_command_row(label: &str, cmd: &Option<String>) {
    match cmd {
        Some(c) => print_row(label, c, GREEN),
        None => print_row(label, "(not detected)", DIM),
    }
}

// ---------------------------------------------------------------------------
// Task progress display
// ---------------------------------------------------------------------------

/// Print a one-line progress update suitable for a running task.
pub fn print_progress(status: &StatusSummary, tracker: &TokenTracker) {
    let pct = (status.progress() * 100.0) as u32;
    let bar = progress_bar(status.progress(), 30);

    let cost_str = format_cost(tracker.total_cost_microdollars);

    if use_color() {
        println!(
            "  {bar} {BOLD}{pct:>3}%{RESET}  \
             {GREEN}{}/{}{RESET} done  \
             {DIM}|{RESET} {YELLOW}{} active{RESET}  \
             {DIM}|{RESET} {RED}{} failed{RESET}  \
             {DIM}|{RESET} tokens: {} in / {} out  \
             {DIM}|{RESET} cost: {cost_str}",
            status.completed,
            status.total,
            status.active,
            status.failed,
            tracker.total_input,
            tracker.total_output,
        );
    } else {
        println!(
            "  {bar} {:>3}%  {}/{} done | {} active | {} failed | tokens: {} in / {} out | cost: {cost_str}",
            pct,
            status.completed,
            status.total,
            status.active,
            status.failed,
            tracker.total_input,
            tracker.total_output,
        );
    }
}

fn progress_bar(ratio: f64, width: usize) -> String {
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);

    if use_color() {
        format!(
            "{GREEN}[{}{}]{RESET}",
            "#".repeat(filled),
            "-".repeat(empty)
        )
    } else {
        format!("[{}{}]", "#".repeat(filled), "-".repeat(empty))
    }
}

// ---------------------------------------------------------------------------
// Status summary display
// ---------------------------------------------------------------------------

/// Print a full status summary (for `kairo status`).
pub fn print_status_summary(
    status: &StatusSummary,
    tracker: &TokenTracker,
    checkpoints: &[CheckpointMeta],
) {
    println!();
    println!("{}", bold("KAIRO-X Task Status"));
    println!("{}", paint(DIM, &"=".repeat(40)));
    println!();

    // Node counts
    print_row("Total nodes", &status.total.to_string(), WHITE);
    print_row("Completed", &status.completed.to_string(), GREEN);
    print_row("Pending", &status.pending.to_string(), CYAN);
    print_row("Active", &status.active.to_string(), YELLOW);
    print_row("Needs fix", &status.fix_needed.to_string(), YELLOW);
    print_row("Failed", &status.failed.to_string(), RED);
    println!();

    // Progress bar
    let pct = (status.progress() * 100.0) as u32;
    println!("  Progress:  {} {pct}%", progress_bar(status.progress(), 30));
    println!();

    // Token spend
    println!("{}", paint(DIM, &"-".repeat(40)));
    println!();
    print_row("Total calls", &tracker.total_calls.to_string(), WHITE);
    print_row("Input tokens", &tracker.total_input.to_string(), WHITE);
    print_row("Output tokens", &tracker.total_output.to_string(), WHITE);
    print_row(
        "Estimated cost",
        &format_cost(tracker.total_cost_microdollars),
        YELLOW,
    );

    if let Some(remaining) = tracker.cost_budget_remaining() {
        let remaining_str = format_cost(remaining);
        let budget_str = format_cost(tracker.cost_limit_microdollars);
        print_row(
            "Cost budget remaining",
            &format!("{remaining_str} / {budget_str}"),
            if tracker.is_budget_exhausted() {
                RED
            } else {
                GREEN
            },
        );
    }
    if let Some(remaining) = tracker.token_budget_remaining() {
        print_row(
            "Token budget remaining",
            &format!("{remaining} / {}", tracker.token_budget),
            if tracker.is_budget_exhausted() {
                RED
            } else {
                GREEN
            },
        );
    }
    println!();

    // Checkpoints
    if !checkpoints.is_empty() {
        println!("{}", paint(DIM, &"-".repeat(40)));
        println!();
        println!("  {}", bold("Checkpoints:"));
        for cp in checkpoints.iter().rev().take(5) {
            println!(
                "    step {:>5}  |  {} nodes  |  {} completed  |  {}",
                cp.step,
                cp.node_count,
                cp.completed_count,
                cp.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            );
        }
        if checkpoints.len() > 5 {
            println!(
                "    {}",
                paint(DIM, &format!("... and {} more", checkpoints.len() - 5))
            );
        }
    } else {
        println!("  {}", paint(DIM, "No checkpoints found."));
    }

    println!();
}

// ---------------------------------------------------------------------------
// Completion summary display
// ---------------------------------------------------------------------------

/// Print the final completion summary after a task finishes.
pub fn print_completion_summary(summary: &TaskCompletionSummary) {
    let all_passed = summary.status.failed == 0;

    println!();
    if all_passed {
        println!(
            "  {} {}",
            paint(GREEN, "[COMPLETE]"),
            bold("Task finished successfully.")
        );
    } else {
        println!(
            "  {} {} ({} node(s) failed)",
            paint(RED, "[PARTIAL]"),
            bold("Task finished with failures."),
            summary.status.failed,
        );
    }
    println!();

    print_row(
        "Nodes processed",
        &summary.total_nodes_processed.to_string(),
        WHITE,
    );
    print_row(
        "Completed",
        &summary.status.completed.to_string(),
        GREEN,
    );
    print_row(
        "Failed",
        &summary.status.failed.to_string(),
        if summary.status.failed > 0 { RED } else { GREEN },
    );
    print_row(
        "LLM calls",
        &summary.total_llm_calls.to_string(),
        WHITE,
    );
    print_row(
        "Input tokens",
        &summary.total_input_tokens.to_string(),
        WHITE,
    );
    print_row(
        "Output tokens",
        &summary.total_output_tokens.to_string(),
        WHITE,
    );
    print_row(
        "Estimated cost",
        &format_cost(summary.estimated_cost_microdollars),
        YELLOW,
    );
    println!();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a cost in microcents as a human-readable dollar string.
fn format_cost(microcents: u64) -> String {
    let dollars = microcents as f64 / 1_000_000.0;
    if dollars < 0.01 {
        format!("${dollars:.4}")
    } else {
        format!("${dollars:.2}")
    }
}
