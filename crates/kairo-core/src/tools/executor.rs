//! Shell command executor with timeout support.
//!
//! Provides a structured interface for running external processes, capturing
//! stdout/stderr, exit codes, and execution duration. Used by the git module
//! and other tools that shell out to external programs.

use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tokio::process::Command;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of a completed command execution.
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// The command that was executed (for diagnostics).
    pub command: String,
    /// Standard output, decoded as UTF-8 (lossy).
    pub stdout: String,
    /// Standard error, decoded as UTF-8 (lossy).
    pub stderr: String,
    /// Process exit code. `None` if the process was terminated by a signal.
    pub exit_code: Option<i32>,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
}

impl CommandResult {
    /// Whether the command exited successfully (exit code 0).
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }

    /// Combined stdout and stderr, with stderr appended after a separator.
    pub fn combined_output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n--- stderr ---\n{}", self.stdout, self.stderr)
        }
    }
}

/// Errors from command execution.
#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("command timed out after {timeout:?}: {command}")]
    Timeout { command: String, timeout: Duration },

    #[error("failed to spawn command '{command}': {source}")]
    SpawnFailed {
        command: String,
        source: std::io::Error,
    },

    #[error("failed to read command output: {source}")]
    OutputReadFailed { source: std::io::Error },

    #[error("command failed (exit {exit_code}): {stderr}")]
    NonZeroExit {
        exit_code: i32,
        stderr: String,
        result: CommandResult,
    },

    #[error("disallowed shell command: {0}")]
    DisallowedCommand(String),
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Default command timeout (60 seconds).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum bytes captured from stdout or stderr (50 MiB).
///
/// Prevents unbounded memory growth when a child process produces enormous
/// output.  Anything beyond this limit is silently truncated.
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Allowlist of command prefixes accepted by [`execute_shell`].
///
/// Only shell strings that begin with one of these prefixes (or match the
/// bare command name with no arguments) are permitted.  Everything else is
/// rejected with [`ExecutorError::DisallowedCommand`].
const ALLOWED_SHELL_PREFIXES: &[&str] = &[
    "cargo ", "rustc ", "npm ", "node ", "python ", "pytest ",
    "go ", "dotnet ", "make ", "cmake ", "git ",
    "ls ", "cat ", "head ", "tail ", "wc ", "sort ", "grep ", "find ",
    "echo ", "mkdir ", "cp ", "mv ",
];

/// Execute a shell command with timeout support.
///
/// # Arguments
///
/// * `cmd` — the program to run (e.g., `"git"`, `"cargo"`).
/// * `args` — command-line arguments.
/// * `working_dir` — working directory for the child process. If `None`,
///   inherits the parent's cwd.
/// * `timeout` — maximum wall-clock time before killing the process.
///   If `None`, defaults to 60 seconds.
/// * `env` — optional slice of `(key, value)` pairs injected into the
///   child's environment.  Existing variables are preserved; duplicates
///   are overwritten.
///
/// # Returns
///
/// A `CommandResult` with captured stdout, stderr, exit code, and duration.
/// Does NOT return an error for non-zero exit codes — check `result.success()`.
/// Only returns errors for spawn failures and timeouts.
pub async fn execute(
    cmd: &str,
    args: &[&str],
    working_dir: Option<&Path>,
    timeout: Option<Duration>,
    env: Option<&[(&str, &str)]>,
) -> Result<CommandResult, ExecutorError> {
    let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
    let cmd_display = format!("{} {}", cmd, args.join(" "));
    debug!(cmd = %cmd_display, timeout = ?timeout, "executing command");

    let mut command = Command::new(cmd);
    command
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // Prevent the child from inheriting stdin to avoid hangs.
        .stdin(std::process::Stdio::null());

    if let Some(dir) = working_dir {
        command.current_dir(dir);
    }

    if let Some(vars) = env {
        for &(key, value) in vars {
            command.env(key, value);
        }
    }

    let start = std::time::Instant::now();

    let mut child = command.spawn().map_err(|e| ExecutorError::SpawnFailed {
        command: cmd_display.clone(),
        source: e,
    })?;

    // Take ownership of the stdout/stderr handles so we can still kill the
    // child if the timeout fires. `wait_with_output()` consumes `self`, which
    // prevents calling `kill()` afterward.
    let child_stdout = child.stdout.take();
    let child_stderr = child.stderr.take();

    let wait_result = tokio::time::timeout(timeout, child.wait()).await;

    let duration = start.elapsed();

    match wait_result {
        Ok(Ok(exit_status)) => {
            let stdout = read_pipe_bounded(child_stdout, &cmd_display, "stdout").await;
            let stderr = read_stderr_bounded(child_stderr, &cmd_display, "stderr").await;
            let exit_code = exit_status.code();

            trace!(
                cmd = %cmd_display,
                exit_code = ?exit_code,
                duration_ms = duration.as_millis() as u64,
                stdout_len = stdout.len(),
                stderr_len = stderr.len(),
                "command completed"
            );

            Ok(CommandResult {
                command: cmd_display,
                stdout,
                stderr,
                exit_code,
                duration,
            })
        }
        Ok(Err(io_err)) => Err(ExecutorError::OutputReadFailed { source: io_err }),
        Err(_elapsed) => {
            // Timeout: kill the child process (we still own `child`).
            warn!(cmd = %cmd_display, timeout = ?timeout, "command timed out, killing");
            let _ = child.kill().await;
            Err(ExecutorError::Timeout {
                command: cmd_display,
                timeout,
            })
        }
    }
}

/// Read up to [`MAX_OUTPUT_BYTES`] from an optional child pipe.
///
/// If the read encounters an I/O error the partial buffer collected so far is
/// returned and a warning is logged — the caller still gets whatever bytes were
/// captured before the failure.
async fn read_pipe_bounded(
    pipe: Option<tokio::process::ChildStdout>,
    cmd_display: &str,
    stream_name: &str,
) -> String {
    match pipe {
        Some(out) => {
            use tokio::io::AsyncReadExt;
            let mut bounded = out.take(MAX_OUTPUT_BYTES as u64);
            let mut buf = Vec::new();
            if let Err(e) = bounded.read_to_end(&mut buf).await {
                warn!(
                    cmd = %cmd_display,
                    stream = stream_name,
                    error = %e,
                    bytes_captured = buf.len(),
                    "error reading command {stream_name}, returning partial output",
                );
            }
            String::from_utf8_lossy(&buf).into_owned()
        }
        None => String::new(),
    }
}

/// Read up to [`MAX_OUTPUT_BYTES`] from an optional child stderr pipe.
///
/// Mirror of [`read_pipe_bounded`] for the stderr handle type.
async fn read_stderr_bounded(
    pipe: Option<tokio::process::ChildStderr>,
    cmd_display: &str,
    stream_name: &str,
) -> String {
    match pipe {
        Some(out) => {
            use tokio::io::AsyncReadExt;
            let mut bounded = out.take(MAX_OUTPUT_BYTES as u64);
            let mut buf = Vec::new();
            if let Err(e) = bounded.read_to_end(&mut buf).await {
                warn!(
                    cmd = %cmd_display,
                    stream = stream_name,
                    error = %e,
                    bytes_captured = buf.len(),
                    "error reading command {stream_name}, returning partial output",
                );
            }
            String::from_utf8_lossy(&buf).into_owned()
        }
        None => String::new(),
    }
}

/// Execute a command and return an error if it exits with a non-zero code.
///
/// Convenience wrapper around `execute` for cases where you expect success.
pub async fn execute_expecting_success(
    cmd: &str,
    args: &[&str],
    working_dir: Option<&Path>,
    timeout: Option<Duration>,
    env: Option<&[(&str, &str)]>,
) -> Result<CommandResult, ExecutorError> {
    let result = execute(cmd, args, working_dir, timeout, env).await?;
    if !result.success() {
        let exit_code = result.exit_code.unwrap_or(-1);
        return Err(ExecutorError::NonZeroExit {
            exit_code,
            stderr: result.stderr.clone(),
            result,
        });
    }
    Ok(result)
}

/// Execute a shell command string via `sh -c` (Unix) with timeout.
///
/// Use this for piped commands or complex shell expressions. Prefer
/// `execute()` for simple commands (avoids shell injection risks).
///
/// The command must begin with one of the prefixes in
/// [`ALLOWED_SHELL_PREFIXES`]; otherwise an
/// [`ExecutorError::DisallowedCommand`] is returned.
pub(crate) async fn execute_shell(
    shell_cmd: &str,
    working_dir: Option<&Path>,
    timeout: Option<Duration>,
) -> Result<CommandResult, ExecutorError> {
    let trimmed = shell_cmd.trim_start();
    let allowed = ALLOWED_SHELL_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix));

    if !allowed {
        warn!(shell_cmd, "rejected disallowed shell command");
        return Err(ExecutorError::DisallowedCommand(shell_cmd.to_string()));
    }

    debug!(shell_cmd, "executing shell command");
    execute("sh", &["-c", shell_cmd], working_dir, timeout, None).await
}

// ---------------------------------------------------------------------------
// Builder (optional ergonomic API)
// ---------------------------------------------------------------------------

/// Builder for constructing and executing commands with a fluent API.
pub struct CommandBuilder {
    cmd: String,
    args: Vec<String>,
    working_dir: Option<PathBuf>,
    timeout: Option<Duration>,
    env: Vec<(String, String)>,
}

impl CommandBuilder {
    /// Create a new command builder.
    pub fn new(cmd: impl Into<String>) -> Self {
        Self {
            cmd: cmd.into(),
            args: Vec::new(),
            working_dir: None,
            timeout: None,
            env: Vec::new(),
        }
    }

    /// Add a single argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add multiple arguments.
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(|a| a.into()));
        self
    }

    /// Set the working directory.
    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add an environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Execute the command.
    pub async fn run(self) -> Result<CommandResult, ExecutorError> {
        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        let cmd_display = format!(
            "{} {}",
            self.cmd,
            self.args.join(" ")
        );

        debug!(cmd = %cmd_display, timeout = ?timeout, "executing command (builder)");

        let mut command = Command::new(&self.cmd);
        let arg_refs: Vec<&str> = self.args.iter().map(|s| s.as_str()).collect();
        command
            .args(&arg_refs)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null());

        if let Some(dir) = &self.working_dir {
            command.current_dir(dir);
        }

        for (key, value) in &self.env {
            command.env(key, value);
        }

        let start = std::time::Instant::now();

        let mut child = command.spawn().map_err(|e| ExecutorError::SpawnFailed {
            command: cmd_display.clone(),
            source: e,
        })?;

        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        let wait_result = tokio::time::timeout(timeout, child.wait()).await;
        let duration = start.elapsed();

        match wait_result {
            Ok(Ok(exit_status)) => {
                let stdout = read_pipe_bounded(child_stdout, &cmd_display, "stdout").await;
                let stderr = read_stderr_bounded(child_stderr, &cmd_display, "stderr").await;
                let exit_code = exit_status.code();

                Ok(CommandResult {
                    command: cmd_display,
                    stdout,
                    stderr,
                    exit_code,
                    duration,
                })
            }
            Ok(Err(io_err)) => Err(ExecutorError::OutputReadFailed { source: io_err }),
            Err(_) => {
                let _ = child.kill().await;
                Err(ExecutorError::Timeout {
                    command: cmd_display,
                    timeout,
                })
            }
        }
    }

    /// Execute and return error on non-zero exit.
    pub async fn run_expecting_success(self) -> Result<CommandResult, ExecutorError> {
        let result = self.run().await?;
        if !result.success() {
            let exit_code = result.exit_code.unwrap_or(-1);
            return Err(ExecutorError::NonZeroExit {
                exit_code,
                stderr: result.stderr.clone(),
                result,
            });
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_echo() {
        let result = execute("echo", &["hello", "world"], None, None, None)
            .await
            .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello world");
        assert!(result.stderr.is_empty());
        assert_eq!(result.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_execute_nonexistent_command() {
        let result = execute(
            "this_command_does_not_exist_kairo_test",
            &[],
            None,
            None,
            None,
        )
        .await;
        assert!(matches!(result, Err(ExecutorError::SpawnFailed { .. })));
    }

    #[tokio::test]
    async fn test_execute_non_zero_exit() {
        let result = execute("false", &[], None, None, None).await.unwrap();
        assert!(!result.success());
        assert_ne!(result.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_execute_expecting_success_fails() {
        let result = execute_expecting_success("false", &[], None, None, None).await;
        assert!(matches!(result, Err(ExecutorError::NonZeroExit { .. })));
    }

    #[tokio::test]
    async fn test_execute_with_working_dir() {
        let result = execute("pwd", &[], Some(Path::new("/tmp")), None, None)
            .await
            .unwrap();
        assert!(result.success());
        // On some systems /tmp is a symlink, so check the real path.
        let output = result.stdout.trim();
        assert!(
            output == "/tmp" || output.ends_with("/tmp"),
            "unexpected pwd output: {output}"
        );
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let result = execute(
            "sleep",
            &["30"],
            None,
            Some(Duration::from_millis(100)),
            None,
        )
        .await;
        assert!(matches!(result, Err(ExecutorError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_execute_captures_stderr() {
        // Uses "echo " prefix which is in the allowlist.
        let result = execute_shell(
            "echo 'out' && echo 'err' >&2",
            None,
            None,
        )
        .await
        .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "out");
        assert_eq!(result.stderr.trim(), "err");
    }

    #[tokio::test]
    async fn test_execute_shell() {
        // Uses "echo " prefix which is in the allowlist.
        let result = execute_shell(
            "echo hello | tr 'h' 'H'",
            None,
            None,
        )
        .await
        .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "Hello");
    }

    #[tokio::test]
    async fn test_execute_shell_disallowed() {
        let result = execute_shell("curl http://example.com", None, None).await;
        assert!(
            matches!(result, Err(ExecutorError::DisallowedCommand(ref cmd)) if cmd.contains("curl")),
            "expected DisallowedCommand, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_execute_with_env() {
        let env_vars: &[(&str, &str)] = &[("KAIRO_TEST_VAR", "hello_from_kairo")];
        let result = execute(
            "sh",
            &["-c", "echo $KAIRO_TEST_VAR"],
            None,
            None,
            Some(env_vars),
        )
        .await
        .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello_from_kairo");
    }

    #[tokio::test]
    async fn test_command_result_combined_output() {
        let r = CommandResult {
            command: "test".to_string(),
            stdout: "out".to_string(),
            stderr: "err".to_string(),
            exit_code: Some(0),
            duration: Duration::from_millis(1),
        };
        assert!(r.combined_output().contains("out"));
        assert!(r.combined_output().contains("err"));
        assert!(r.combined_output().contains("stderr"));
    }

    #[tokio::test]
    async fn test_command_result_combined_output_empty_stderr() {
        let r = CommandResult {
            command: "test".to_string(),
            stdout: "out".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_millis(1),
        };
        assert_eq!(r.combined_output(), "out");
    }

    #[tokio::test]
    async fn test_command_builder() {
        let result = CommandBuilder::new("echo")
            .arg("builder")
            .arg("test")
            .run()
            .await
            .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "builder test");
    }

    #[tokio::test]
    async fn test_command_builder_with_working_dir() {
        let result = CommandBuilder::new("ls")
            .working_dir("/tmp")
            .run()
            .await
            .unwrap();
        assert!(result.success());
    }

    #[tokio::test]
    async fn test_duration_tracked() {
        let result = execute("sleep", &["0.1"], None, None, None).await.unwrap();
        assert!(result.duration >= Duration::from_millis(50));
    }
}
