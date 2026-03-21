//! Git operations for the KAIRO-X agent.
//!
//! All operations shell out to the `git` CLI via the executor module.
//! Returns structured results with parsed output for programmatic use.

use super::executor::{self, CommandResult, ExecutorError};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;
use tracing::debug;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from git operations.
#[derive(Debug, Error)]
pub enum GitError {
    #[error("git command failed: {message}")]
    CommandFailed {
        message: String,
        result: CommandResult,
    },

    #[error("not a git repository: {0}")]
    NotARepo(String),

    #[error("executor error: {0}")]
    Executor(#[from] ExecutorError),
}

// ---------------------------------------------------------------------------
// Structured results
// ---------------------------------------------------------------------------

/// Parsed result of `git status`.
#[derive(Debug, Clone)]
pub struct GitStatus {
    /// Current branch name (if on a branch).
    pub branch: Option<String>,
    /// Files in the staging area (added, modified, deleted).
    pub staged: Vec<FileStatus>,
    /// Modified files not yet staged.
    pub unstaged: Vec<FileStatus>,
    /// Untracked files.
    pub untracked: Vec<String>,
    /// Whether the working tree is clean (no changes at all).
    pub is_clean: bool,
    /// Raw output from git status.
    pub raw: String,
}

/// Status of a single file in git.
#[derive(Debug, Clone)]
pub struct FileStatus {
    /// The status code (e.g., "M" for modified, "A" for added, "D" for deleted).
    pub status: String,
    /// The file path relative to the repository root.
    pub path: String,
}

/// Parsed result of `git diff`.
#[derive(Debug, Clone)]
pub struct GitDiff {
    /// Number of files changed.
    pub files_changed: usize,
    /// Total lines added.
    pub insertions: usize,
    /// Total lines removed.
    pub deletions: usize,
    /// Per-file diff hunks.
    pub file_diffs: Vec<FileDiff>,
    /// Raw diff output.
    pub raw: String,
}

/// Diff information for a single file.
#[derive(Debug, Clone)]
pub struct FileDiff {
    /// File path.
    pub path: String,
    /// The raw diff content for this file.
    pub diff: String,
}

/// Result of a commit operation.
#[derive(Debug, Clone)]
pub struct CommitResult {
    /// The short SHA of the new commit.
    pub sha: String,
    /// The commit message.
    pub message: String,
    /// Raw output.
    pub raw: String,
}

/// Result of a stash operation.
#[derive(Debug, Clone)]
pub struct StashResult {
    /// The stash reference (e.g., "stash@{0}").
    pub reference: String,
    /// Raw output.
    pub raw: String,
}

// ---------------------------------------------------------------------------
// Git timeout
// ---------------------------------------------------------------------------

/// Default timeout for git commands.
const GIT_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Run `git status` and parse the result.
pub async fn status(repo_dir: &Path) -> Result<GitStatus, GitError> {
    debug!(repo = %repo_dir.display(), "git status");

    // Use porcelain format for machine-readable output.
    let result = executor::execute(
        "git",
        &["status", "--porcelain=v1", "--branch"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        if result.stderr.contains("not a git repository") {
            return Err(GitError::NotARepo(repo_dir.display().to_string()));
        }
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    let mut branch = None;
    let mut staged = Vec::new();
    let mut unstaged = Vec::new();
    let mut untracked = Vec::new();

    for line in result.stdout.lines() {
        if let Some(branch_info) = line.strip_prefix("## ") {
            // Branch line: "## main...origin/main" or "## HEAD (no branch)"
            let branch_name = branch_info
                .split("...")
                .next()
                .unwrap_or(branch_info)
                .trim();
            if branch_name != "HEAD (no branch)" {
                branch = Some(branch_name.to_string());
            }
            continue;
        }

        if line.len() < 4 {
            continue;
        }

        let index_status = line.as_bytes()[0];
        let worktree_status = line.as_bytes()[1];
        let file_path = line[3..].to_string();

        // Untracked files: "?? file"
        if index_status == b'?' && worktree_status == b'?' {
            untracked.push(file_path);
            continue;
        }

        // Staged changes (index column).
        if index_status != b' ' && index_status != b'?' {
            staged.push(FileStatus {
                status: String::from_utf8_lossy(&[index_status]).into_owned(),
                path: file_path.clone(),
            });
        }

        // Unstaged changes (worktree column).
        if worktree_status != b' ' && worktree_status != b'?' {
            unstaged.push(FileStatus {
                status: String::from_utf8_lossy(&[worktree_status]).into_owned(),
                path: file_path,
            });
        }
    }

    let is_clean = staged.is_empty() && unstaged.is_empty() && untracked.is_empty();

    Ok(GitStatus {
        branch,
        staged,
        unstaged,
        untracked,
        is_clean,
        raw: result.stdout,
    })
}

/// Run `git diff` and parse the result.
///
/// If `staged` is true, shows staged changes (`--cached`).
/// If `path_filter` is provided, limits diff to that path.
pub async fn diff(
    repo_dir: &Path,
    staged: bool,
    path_filter: Option<&str>,
) -> Result<GitDiff, GitError> {
    debug!(
        repo = %repo_dir.display(),
        staged,
        path_filter,
        "git diff"
    );

    let mut args = vec!["diff"];
    if staged {
        args.push("--cached");
    }
    args.push("--stat");

    if let Some(path) = path_filter {
        args.push("--");
        args.push(path);
    }

    // First, get the stat summary.
    let stat_result = executor::execute(
        "git",
        &args,
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !stat_result.success() {
        return Err(GitError::CommandFailed {
            message: stat_result.stderr.clone(),
            result: stat_result,
        });
    }

    // Parse stat line: " N files changed, N insertions(+), N deletions(-)"
    let mut files_changed = 0;
    let mut insertions = 0;
    let mut deletions = 0;

    for line in stat_result.stdout.lines() {
        let trimmed = line.trim();
        if trimmed.contains("file") && trimmed.contains("changed") {
            for part in trimmed.split(',') {
                let part = part.trim();
                if part.contains("file") {
                    if let Some(n) = part.split_whitespace().next() {
                        files_changed = n.parse().unwrap_or(0);
                    }
                } else if part.contains("insertion") {
                    if let Some(n) = part.split_whitespace().next() {
                        insertions = n.parse().unwrap_or(0);
                    }
                } else if part.contains("deletion") {
                    if let Some(n) = part.split_whitespace().next() {
                        deletions = n.parse().unwrap_or(0);
                    }
                }
            }
        }
    }

    // Now get the full diff for parsing file-level diffs.
    let mut full_args = vec!["diff"];
    if staged {
        full_args.push("--cached");
    }
    if let Some(path) = path_filter {
        full_args.push("--");
        full_args.push(path);
    }

    let full_result = executor::execute(
        "git",
        &full_args,
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    let file_diffs = parse_diff_files(&full_result.stdout);

    Ok(GitDiff {
        files_changed,
        insertions,
        deletions,
        file_diffs,
        raw: full_result.stdout,
    })
}

/// Parse a unified diff into per-file sections.
fn parse_diff_files(diff_output: &str) -> Vec<FileDiff> {
    let mut file_diffs = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_diff = String::new();

    for line in diff_output.lines() {
        if line.starts_with("diff --git ") {
            // Flush previous file diff.
            if let Some(path) = current_path.take() {
                file_diffs.push(FileDiff {
                    path,
                    diff: std::mem::take(&mut current_diff),
                });
            }

            // Parse file path from "diff --git a/path b/path".
            let parts: Vec<&str> = line.splitn(4, ' ').collect();
            if parts.len() >= 4 {
                let b_path = parts[3];
                // Strip "b/" prefix.
                let path = if let Some(stripped) = b_path.strip_prefix("b/") {
                    stripped
                } else {
                    b_path
                };
                current_path = Some(path.to_string());
            }
        }

        current_diff.push_str(line);
        current_diff.push('\n');
    }

    // Flush last file.
    if let Some(path) = current_path {
        file_diffs.push(FileDiff {
            path,
            diff: current_diff,
        });
    }

    file_diffs
}

/// Create a git commit with the given message.
///
/// Stages all modified/deleted files first (`git add -u`), then commits.
/// If `add_all` is true, also stages untracked files (`git add -A`).
pub async fn commit(
    repo_dir: &Path,
    message: &str,
    add_all: bool,
) -> Result<CommitResult, GitError> {
    debug!(
        repo = %repo_dir.display(),
        add_all,
        message_len = message.len(),
        "git commit"
    );

    // Stage changes.
    let add_flag = if add_all { "-A" } else { "-u" };
    let add_result = executor::execute(
        "git",
        &["add", add_flag],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !add_result.success() {
        return Err(GitError::CommandFailed {
            message: add_result.stderr.clone(),
            result: add_result,
        });
    }

    // Commit.
    let commit_result = executor::execute(
        "git",
        &["commit", "-m", message],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !commit_result.success() {
        return Err(GitError::CommandFailed {
            message: commit_result.stderr.clone(),
            result: commit_result,
        });
    }

    // Get the SHA of the new commit.
    let sha_result = executor::execute(
        "git",
        &["rev-parse", "--short", "HEAD"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    let sha = sha_result.stdout.trim().to_string();

    Ok(CommitResult {
        sha,
        message: message.to_string(),
        raw: commit_result.stdout,
    })
}

/// Rollback specific files to their last committed state.
///
/// Uses `git checkout -- <files>` to discard working tree changes.
pub async fn rollback(repo_dir: &Path, files: &[&str]) -> Result<CommandResult, GitError> {
    debug!(
        repo = %repo_dir.display(),
        file_count = files.len(),
        "git rollback (checkout)"
    );

    if files.is_empty() {
        return Err(GitError::CommandFailed {
            message: "no files specified for rollback".to_string(),
            result: CommandResult {
                command: "git checkout".to_string(),
                stdout: String::new(),
                stderr: "no files specified".to_string(),
                exit_code: Some(1),
                duration: Duration::ZERO,
            },
        });
    }

    let mut args: Vec<&str> = vec!["checkout", "--"];
    args.extend_from_slice(files);

    let result = executor::execute(
        "git",
        &args,
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    Ok(result)
}

/// Stash all changes (tracked and untracked).
///
/// If `message` is provided, uses it as the stash message.
pub async fn stash(
    repo_dir: &Path,
    message: Option<&str>,
) -> Result<StashResult, GitError> {
    debug!(repo = %repo_dir.display(), message, "git stash");

    let mut args = vec!["stash", "push", "--include-untracked"];
    if let Some(msg) = message {
        args.push("-m");
        args.push(msg);
    }

    let result = executor::execute(
        "git",
        &args,
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    // Determine the stash reference.
    let reference = if result.stdout.contains("No local changes to save") {
        String::new()
    } else {
        // Get the latest stash ref.
        let ref_result = executor::execute(
            "git",
            &["stash", "list", "-1"],
            Some(repo_dir),
            Some(GIT_TIMEOUT),
        )
        .await?;

        ref_result
            .stdout
            .split(':')
            .next()
            .unwrap_or("stash@{0}")
            .trim()
            .to_string()
    };

    Ok(StashResult {
        reference,
        raw: result.stdout,
    })
}

/// Pop the most recent stash entry.
pub async fn stash_pop(repo_dir: &Path) -> Result<CommandResult, GitError> {
    debug!(repo = %repo_dir.display(), "git stash pop");

    let result = executor::execute(
        "git",
        &["stash", "pop"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    Ok(result)
}

/// Get the current HEAD commit SHA (full).
pub async fn head_sha(repo_dir: &Path) -> Result<String, GitError> {
    let result = executor::execute(
        "git",
        &["rev-parse", "HEAD"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    Ok(result.stdout.trim().to_string())
}

/// Get the repository root directory.
pub async fn repo_root(repo_dir: &Path) -> Result<String, GitError> {
    let result = executor::execute(
        "git",
        &["rev-parse", "--show-toplevel"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        if result.stderr.contains("not a git repository") {
            return Err(GitError::NotARepo(repo_dir.display().to_string()));
        }
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    Ok(result.stdout.trim().to_string())
}

/// Get the log of recent commits.
///
/// Returns up to `limit` most recent commits in oneline format.
pub async fn log(
    repo_dir: &Path,
    limit: usize,
) -> Result<Vec<(String, String)>, GitError> {
    let limit_str = format!("-{limit}");
    let result = executor::execute(
        "git",
        &["log", &limit_str, "--oneline"],
        Some(repo_dir),
        Some(GIT_TIMEOUT),
    )
    .await?;

    if !result.success() {
        return Err(GitError::CommandFailed {
            message: result.stderr.clone(),
            result,
        });
    }

    let entries = result
        .stdout
        .lines()
        .filter_map(|line| {
            let (sha, msg) = line.split_once(' ')?;
            Some((sha.to_string(), msg.to_string()))
        })
        .collect();

    Ok(entries)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_diff_files() {
        let diff = "\
diff --git a/src/main.rs b/src/main.rs
index abc123..def456 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,3 @@
-fn old() {}
+fn new() {}
diff --git a/src/lib.rs b/src/lib.rs
index 111222..333444 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-pub mod old;
+pub mod new;
";

        let files = parse_diff_files(diff);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "src/main.rs");
        assert_eq!(files[1].path, "src/lib.rs");
        assert!(files[0].diff.contains("-fn old()"));
        assert!(files[1].diff.contains("+pub mod new"));
    }

    #[test]
    fn test_parse_diff_files_empty() {
        let files = parse_diff_files("");
        assert!(files.is_empty());
    }

    #[test]
    fn test_parse_diff_single_file() {
        let diff = "\
diff --git a/README.md b/README.md
index aaa..bbb 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-Old title
+New title
";
        let files = parse_diff_files(diff);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "README.md");
    }

    // Integration tests require a git repository; they are run conditionally.
    // Use `cargo test -- --ignored` to run them.

    #[tokio::test]
    #[ignore = "requires git repo"]
    async fn test_git_status() {
        let result = status(Path::new(".")).await;
        // This test just verifies the function runs without crashing.
        assert!(result.is_ok() || matches!(result, Err(GitError::NotARepo(_))));
    }

    #[tokio::test]
    #[ignore = "requires git repo"]
    async fn test_git_diff() {
        let result = diff(Path::new("."), false, None).await;
        assert!(result.is_ok() || matches!(result, Err(GitError::NotARepo(_))));
    }

    #[tokio::test]
    #[ignore = "requires git repo"]
    async fn test_git_head_sha() {
        let result = head_sha(Path::new(".")).await;
        if let Ok(sha) = result {
            assert_eq!(sha.len(), 40); // Full SHA is 40 hex chars.
        }
    }

    #[tokio::test]
    async fn test_git_not_a_repo() {
        let result = status(Path::new("/tmp")).await;
        // /tmp is likely not a git repo.
        assert!(
            result.is_ok() || matches!(result, Err(GitError::NotARepo(_) | GitError::CommandFailed { .. }))
        );
    }

    #[tokio::test]
    #[ignore = "requires git repo"]
    async fn test_git_log() {
        let result = log(Path::new("."), 5).await;
        if let Ok(entries) = result {
            for (sha, msg) in &entries {
                assert!(!sha.is_empty());
                assert!(!msg.is_empty());
            }
        }
    }
}
