//! Filesystem operations for the KAIRO-X agent.
//!
//! Provides a safe, sandboxed interface for file system interactions:
//! reading, writing, listing, searching, querying metadata, and deleting.
//!
//! **Security invariants:**
//! - All operations are confined to a `workspace_root` directory. Paths that
//!   resolve (via canonicalization) outside the workspace are rejected.
//! - Symlinks are rejected for read/write/delete operations to prevent
//!   symlink-based escapes.
//! - File reads are bounded by `MAX_FILE_SIZE` (50 MiB) to prevent OOM.
//! - Directories are created with mode 0o700 (owner-only) on Unix.
//! - TOCTOU pre-checks (`path.exists()`) are removed; errors come from
//!   the actual I/O syscall and are mapped via `from_io`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum file size we will read into memory (50 MiB).
const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from filesystem operations.
#[derive(Debug, Error)]
pub enum FsError {
    #[error("file not found: {0}")]
    NotFound(PathBuf),

    #[error("permission denied: {0}")]
    PermissionDenied(PathBuf),

    #[error("not a file: {0}")]
    NotAFile(PathBuf),

    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),

    #[error("path is outside allowed workspace: {0}")]
    OutsideWorkspace(PathBuf),

    #[error("symlinks are not allowed: {0}")]
    SymlinkNotAllowed(PathBuf),

    #[error("file too large: {path} is {size} bytes (max {max} bytes)")]
    FileTooLarge { path: PathBuf, size: u64, max: u64 },

    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("UTF-8 decoding error for {path}: {source}")]
    Utf8 {
        path: PathBuf,
        source: std::string::FromUtf8Error,
    },
}

impl FsError {
    /// Wrap a `std::io::Error` with path context, mapping common error kinds.
    fn from_io(path: &Path, err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => Self::NotFound(path.to_path_buf()),
            std::io::ErrorKind::PermissionDenied => Self::PermissionDenied(path.to_path_buf()),
            _ => Self::Io {
                path: path.to_path_buf(),
                source: err,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Sandbox helpers
// ---------------------------------------------------------------------------

/// Canonicalize `path` and verify it lives inside `workspace_root`.
///
/// Both the workspace root and the target path are canonicalized so that
/// `..` traversals and symlink tricks cannot escape the sandbox. For
/// paths that do not yet exist (e.g. a file about to be created), we
/// canonicalize the longest existing ancestor and append the remaining
/// components.
fn enforce_sandbox(path: &Path, workspace_root: &Path) -> Result<PathBuf, FsError> {
    let canon_root = workspace_root
        .canonicalize()
        .map_err(|e| FsError::from_io(workspace_root, e))?;

    let canon_path = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // Path does not exist yet. Walk up until we find an ancestor
            // that does exist, canonicalize it, then re-attach the tail.
            let mut existing = path.to_path_buf();
            let mut suffix_parts: Vec<std::ffi::OsString> = Vec::new();
            loop {
                if existing.exists() {
                    break;
                }
                match existing.file_name() {
                    Some(part) => {
                        suffix_parts.push(part.to_os_string());
                        existing = match existing.parent() {
                            Some(p) => p.to_path_buf(),
                            None => {
                                return Err(FsError::NotFound(path.to_path_buf()));
                            }
                        };
                    }
                    None => {
                        return Err(FsError::NotFound(path.to_path_buf()));
                    }
                }
            }
            let mut canon = existing
                .canonicalize()
                .map_err(|e| FsError::from_io(&existing, e))?;
            for part in suffix_parts.into_iter().rev() {
                canon.push(part);
            }
            canon
        }
    };

    if !canon_path.starts_with(&canon_root) {
        return Err(FsError::OutsideWorkspace(path.to_path_buf()));
    }

    Ok(canon_path)
}

/// Reject symlinks. Uses `symlink_metadata` (lstat) so it inspects the
/// link itself, not its target.
fn reject_symlink(path: &Path) -> Result<(), FsError> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            Err(FsError::SymlinkNotAllowed(path.to_path_buf()))
        }
        _ => Ok(()),
    }
}

/// Check that a file's size is within `MAX_FILE_SIZE`.
fn enforce_size_limit(path: &Path, meta: &std::fs::Metadata) -> Result<(), FsError> {
    let size = meta.len();
    if size > MAX_FILE_SIZE {
        return Err(FsError::FileTooLarge {
            path: path.to_path_buf(),
            size,
            max: MAX_FILE_SIZE,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FileInfo
// ---------------------------------------------------------------------------

/// Metadata about a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Last modification time.
    pub modified: DateTime<Utc>,
    /// Whether this is a directory (false = regular file or symlink).
    pub is_dir: bool,
    /// Whether this is a symbolic link.
    pub is_symlink: bool,
    /// Whether the file is read-only.
    pub readonly: bool,
}

/// Entry in a directory listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    /// File/directory name (not full path).
    pub name: String,
    /// Full path.
    pub path: PathBuf,
    /// Whether this is a directory.
    pub is_dir: bool,
    /// File size in bytes (0 for directories).
    pub size: u64,
}

/// A match from a text search within a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    /// Path to the file containing the match.
    pub path: PathBuf,
    /// 1-based line number.
    pub line_number: usize,
    /// The full text of the matching line.
    pub line: String,
}

// ---------------------------------------------------------------------------
// Directory creation helper (mode 0o700 on Unix)
// ---------------------------------------------------------------------------

/// Create directories with owner-only permissions on Unix.
fn create_dir_all_restricted(path: &Path) -> Result<(), FsError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)
            .map_err(|e| FsError::from_io(path, e))
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path).map_err(|e| FsError::from_io(path, e))
    }
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Read the entire contents of a file as a UTF-8 string.
///
/// The file must reside within `workspace_root`, must not be a symlink,
/// and must be smaller than `MAX_FILE_SIZE`.
pub fn read_file(path: &Path, workspace_root: &Path) -> Result<String, FsError> {
    debug!(path = %path.display(), "reading file");

    let canon = enforce_sandbox(path, workspace_root)?;
    reject_symlink(&canon)?;

    let meta = std::fs::symlink_metadata(&canon).map_err(|e| FsError::from_io(&canon, e))?;
    if meta.is_dir() {
        return Err(FsError::NotAFile(canon));
    }
    enforce_size_limit(&canon, &meta)?;

    let bytes = std::fs::read(&canon).map_err(|e| FsError::from_io(&canon, e))?;
    String::from_utf8(bytes).map_err(|e| FsError::Utf8 {
        path: canon,
        source: e,
    })
}

/// Read raw bytes from a file.
///
/// The file must reside within `workspace_root`, must not be a symlink,
/// and must be smaller than `MAX_FILE_SIZE`.
pub fn read_file_bytes(path: &Path, workspace_root: &Path) -> Result<Vec<u8>, FsError> {
    debug!(path = %path.display(), "reading file bytes");

    let canon = enforce_sandbox(path, workspace_root)?;
    reject_symlink(&canon)?;

    let meta = std::fs::symlink_metadata(&canon).map_err(|e| FsError::from_io(&canon, e))?;
    if meta.is_dir() {
        return Err(FsError::NotAFile(canon));
    }
    enforce_size_limit(&canon, &meta)?;

    std::fs::read(&canon).map_err(|e| FsError::from_io(&canon, e))
}

/// Write string content to a file, creating parent directories as needed.
///
/// If the file exists, it is overwritten. If it does not exist, it is created.
/// Parent directories are created with mode 0o700.
pub fn write_file(path: &Path, content: &str, workspace_root: &Path) -> Result<(), FsError> {
    debug!(path = %path.display(), len = content.len(), "writing file");

    let canon = enforce_sandbox(path, workspace_root)?;

    // Reject symlinks only if the target already exists on disk.
    if canon.exists() {
        reject_symlink(&canon)?;
    }

    if let Some(parent) = canon.parent() {
        create_dir_all_restricted(parent)?;
    }

    std::fs::write(&canon, content).map_err(|e| FsError::from_io(&canon, e))
}

/// Write raw bytes to a file, creating parent directories as needed.
///
/// Parent directories are created with mode 0o700.
pub fn write_file_bytes(
    path: &Path,
    content: &[u8],
    workspace_root: &Path,
) -> Result<(), FsError> {
    debug!(path = %path.display(), len = content.len(), "writing file bytes");

    let canon = enforce_sandbox(path, workspace_root)?;

    if canon.exists() {
        reject_symlink(&canon)?;
    }

    if let Some(parent) = canon.parent() {
        create_dir_all_restricted(parent)?;
    }

    std::fs::write(&canon, content).map_err(|e| FsError::from_io(&canon, e))
}

/// List the contents of a directory, sorted by name.
///
/// Returns entries for immediate children only (non-recursive).
/// Broken symlinks are skipped with a warning log rather than causing
/// a hard error.
pub fn list_directory(path: &Path, workspace_root: &Path) -> Result<Vec<DirEntry>, FsError> {
    debug!(path = %path.display(), "listing directory");

    let canon = enforce_sandbox(path, workspace_root)?;

    let read_dir = std::fs::read_dir(&canon).map_err(|e| {
        // Disambiguate "not found" from "not a directory" by checking the
        // error that the OS gave us, rather than doing a TOCTOU pre-check.
        FsError::from_io(&canon, e)
    })?;

    let mut entries = Vec::new();

    for entry_result in read_dir {
        let entry = entry_result.map_err(|e| FsError::from_io(&canon, e))?;
        let entry_path = entry.path();

        // Use symlink_metadata so we don't follow broken symlinks.
        let metadata = match std::fs::symlink_metadata(&entry_path) {
            Ok(m) => m,
            Err(e) => {
                warn!(
                    path = %entry_path.display(),
                    error = %e,
                    "skipping entry with unreadable metadata"
                );
                continue;
            }
        };

        // Skip broken symlinks gracefully.
        if metadata.file_type().is_symlink() {
            // Check if the symlink target is accessible.
            if std::fs::metadata(&entry_path).is_err() {
                warn!(
                    path = %entry_path.display(),
                    "skipping broken symlink"
                );
                continue;
            }
        }

        let name = entry.file_name().to_string_lossy().into_owned();

        // For symlinks that point to valid targets, resolve to get the
        // target's type/size. For everything else, use symlink_metadata.
        let (is_dir, size) = if metadata.file_type().is_symlink() {
            match std::fs::metadata(&entry_path) {
                Ok(target_meta) => (
                    target_meta.is_dir(),
                    if target_meta.is_file() {
                        target_meta.len()
                    } else {
                        0
                    },
                ),
                Err(_) => (false, 0),
            }
        } else {
            (
                metadata.is_dir(),
                if metadata.is_file() {
                    metadata.len()
                } else {
                    0
                },
            )
        };

        entries.push(DirEntry {
            name,
            path: entry_path,
            is_dir,
            size,
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    trace!(path = %canon.display(), count = entries.len(), "directory listed");
    Ok(entries)
}

/// Search for a text pattern in files under a directory (recursive).
///
/// Performs a simple line-by-line substring search (case-sensitive).
/// Skips binary files, hidden directories (those starting with `.`),
/// symlinks, and files larger than `MAX_FILE_SIZE`.
///
/// Returns up to `max_results` matches.
pub fn search_text(
    dir: &Path,
    pattern: &str,
    max_results: usize,
    workspace_root: &Path,
) -> Result<Vec<SearchMatch>, FsError> {
    debug!(
        dir = %dir.display(),
        pattern,
        max_results,
        "searching text"
    );

    let canon = enforce_sandbox(dir, workspace_root)?;

    let mut matches = Vec::new();
    search_text_recursive(&canon, pattern, max_results, &mut matches)?;
    Ok(matches)
}

/// Recursive helper for text search.
fn search_text_recursive(
    dir: &Path,
    pattern: &str,
    max_results: usize,
    matches: &mut Vec<SearchMatch>,
) -> Result<(), FsError> {
    if matches.len() >= max_results {
        return Ok(());
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            // If a directory became inaccessible mid-traversal (e.g. a
            // symlink loop target), log and skip instead of hard-failing.
            warn!(dir = %dir.display(), error = %e, "skipping inaccessible directory during search");
            return Ok(());
        }
    };

    for entry_result in entries {
        if matches.len() >= max_results {
            break;
        }

        let entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                warn!(dir = %dir.display(), error = %e, "skipping unreadable directory entry");
                continue;
            }
        };
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden directories and files.
        if name.starts_with('.') {
            continue;
        }

        // Use symlink_metadata so we can detect symlinks and avoid
        // following broken links or symlink loops.
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "skipping entry with unreadable metadata during search");
                continue;
            }
        };

        let ft = metadata.file_type();

        // Skip symlinks entirely (prevents loops and escapes).
        if ft.is_symlink() {
            trace!(path = %path.display(), "skipping symlink during search");
            continue;
        }

        if ft.is_dir() {
            search_text_recursive(&path, pattern, max_results, matches)?;
        } else if ft.is_file() {
            // Skip files exceeding the size limit.
            if metadata.len() > MAX_FILE_SIZE {
                trace!(
                    path = %path.display(),
                    size = metadata.len(),
                    "skipping oversized file during search"
                );
                continue;
            }

            // Attempt to read as UTF-8; skip binary files silently.
            if let Ok(content) = std::fs::read_to_string(&path) {
                for (line_idx, line) in content.lines().enumerate() {
                    if matches.len() >= max_results {
                        break;
                    }
                    if line.contains(pattern) {
                        matches.push(SearchMatch {
                            path: path.clone(),
                            line_number: line_idx + 1,
                            line: line.to_string(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Get metadata about a file or directory.
pub fn file_info(path: &Path, workspace_root: &Path) -> Result<FileInfo, FsError> {
    debug!(path = %path.display(), "querying file info");

    let canon = enforce_sandbox(path, workspace_root)?;

    let symlink_meta =
        std::fs::symlink_metadata(&canon).map_err(|e| FsError::from_io(&canon, e))?;

    // For symlinks, also resolve the target to get size/type info.
    // For non-symlinks, symlink_metadata == metadata.
    let resolved_meta = if symlink_meta.file_type().is_symlink() {
        std::fs::metadata(&canon).map_err(|e| FsError::from_io(&canon, e))?
    } else {
        symlink_meta.clone()
    };

    let modified = resolved_meta
        .modified()
        .map_err(|e| FsError::from_io(&canon, e))?;
    let modified_dt: DateTime<Utc> = modified.into();

    Ok(FileInfo {
        path: canon,
        size: resolved_meta.len(),
        modified: modified_dt,
        is_dir: resolved_meta.is_dir(),
        is_symlink: symlink_meta.file_type().is_symlink(),
        readonly: resolved_meta.permissions().readonly(),
    })
}

/// Delete a file. Returns an error if the path is a directory or a symlink.
pub fn delete_file(path: &Path, workspace_root: &Path) -> Result<(), FsError> {
    debug!(path = %path.display(), "deleting file");

    let canon = enforce_sandbox(path, workspace_root)?;
    reject_symlink(&canon)?;

    // Let remove_file report NotFound / NotAFile / PermissionDenied via
    // the OS error, avoiding TOCTOU pre-checks.
    std::fs::remove_file(&canon).map_err(|e| {
        // `remove_file` on a directory returns an OS-specific error; map it
        // to our `NotAFile` variant for a clear message.
        if canon.is_dir() {
            FsError::NotAFile(canon.clone())
        } else {
            FsError::from_io(&canon, e)
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temporary directory for test isolation. Returns `(dir, workspace)`
    /// where `workspace` is the canonicalized path suitable for use as
    /// `workspace_root`.
    fn temp_dir() -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("kairo_fs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let canon = dir.canonicalize().unwrap();
        (canon.clone(), canon)
    }

    #[test]
    fn test_read_write_file() {
        let (dir, ws) = temp_dir();
        let file = dir.join("hello.txt");

        write_file(&file, "Hello, KAIRO!", &ws).unwrap();
        let content = read_file(&file, &ws).unwrap();
        assert_eq!(content, "Hello, KAIRO!");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_nonexistent() {
        let (dir, ws) = temp_dir();
        let result = read_file(&dir.join("nonexistent.txt"), &ws);
        assert!(matches!(result, Err(FsError::NotFound(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_directory_as_file() {
        let (dir, ws) = temp_dir();
        let result = read_file(&dir, &ws);
        assert!(matches!(result, Err(FsError::NotAFile(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let (dir, ws) = temp_dir();
        let file = dir.join("a").join("b").join("c.txt");

        write_file(&file, "nested", &ws).unwrap();
        assert_eq!(read_file(&file, &ws).unwrap(), "nested");

        // Verify parent dir permissions on Unix.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let parent_meta = fs::metadata(file.parent().unwrap()).unwrap();
            assert_eq!(parent_meta.permissions().mode() & 0o777, 0o700);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_directory() {
        let (dir, ws) = temp_dir();
        fs::write(dir.join("alpha.txt"), "a").unwrap();
        fs::write(dir.join("beta.txt"), "b").unwrap();
        fs::create_dir(dir.join("gamma")).unwrap();

        let entries = list_directory(&dir, &ws).unwrap();
        assert_eq!(entries.len(), 3);
        // Should be sorted by name.
        assert_eq!(entries[0].name, "alpha.txt");
        assert_eq!(entries[1].name, "beta.txt");
        assert_eq!(entries[2].name, "gamma");
        assert!(!entries[0].is_dir);
        assert!(entries[2].is_dir);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_nonexistent_directory() {
        let (dir, ws) = temp_dir();
        let result = list_directory(&dir.join("nonexistent"), &ws);
        assert!(matches!(result, Err(FsError::NotFound(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_file_as_directory() {
        let (dir, ws) = temp_dir();
        let file = dir.join("file.txt");
        fs::write(&file, "not a dir").unwrap();

        let result = list_directory(&file, &ws);
        // `read_dir` on a file returns an IO error, which from_io maps.
        assert!(result.is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_search_text() {
        let (dir, ws) = temp_dir();
        fs::write(
            dir.join("a.rs"),
            "fn main() {\n    println!(\"hello\");\n}",
        )
        .unwrap();
        fs::write(
            dir.join("b.rs"),
            "fn other() {\n    println!(\"world\");\n}",
        )
        .unwrap();
        fs::write(dir.join("c.txt"), "no match here").unwrap();

        let matches = search_text(&dir, "println!", 100, &ws).unwrap();
        assert_eq!(matches.len(), 2);

        // All matches should be on line 2.
        for m in &matches {
            assert_eq!(m.line_number, 2);
            assert!(m.line.contains("println!"));
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_search_text_max_results() {
        let (dir, ws) = temp_dir();
        for i in 0..10 {
            fs::write(dir.join(format!("{i}.txt")), "needle").unwrap();
        }

        let matches = search_text(&dir, "needle", 3, &ws).unwrap();
        assert_eq!(matches.len(), 3);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_search_text_skips_hidden() {
        let (dir, ws) = temp_dir();
        let hidden = dir.join(".hidden");
        fs::create_dir(&hidden).unwrap();
        fs::write(hidden.join("secret.txt"), "needle").unwrap();
        fs::write(dir.join("visible.txt"), "needle").unwrap();

        let matches = search_text(&dir, "needle", 100, &ws).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].path.ends_with("visible.txt"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info() {
        let (dir, ws) = temp_dir();
        let file = dir.join("info.txt");
        fs::write(&file, "twelve chars").unwrap();

        let info = file_info(&file, &ws).unwrap();
        assert_eq!(info.size, 12);
        assert!(!info.is_dir);
        assert!(!info.is_symlink);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info_directory() {
        let (dir, ws) = temp_dir();
        let info = file_info(&dir, &ws).unwrap();
        assert!(info.is_dir);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info_nonexistent() {
        let (dir, ws) = temp_dir();
        let result = file_info(&dir.join("nonexistent.txt"), &ws);
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_file() {
        let (dir, ws) = temp_dir();
        let file = dir.join("delete_me.txt");
        fs::write(&file, "bye").unwrap();

        assert!(file.exists());
        delete_file(&file, &ws).unwrap();
        assert!(!file.exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_nonexistent() {
        let (dir, ws) = temp_dir();
        let result = delete_file(&dir.join("nonexistent.txt"), &ws);
        assert!(matches!(result, Err(FsError::NotFound(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_directory_fails() {
        let (dir, ws) = temp_dir();
        let result = delete_file(&dir, &ws);
        assert!(matches!(result, Err(FsError::NotAFile(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_overwrite() {
        let (dir, ws) = temp_dir();
        let file = dir.join("overwrite.txt");

        write_file(&file, "first", &ws).unwrap();
        write_file(&file, "second", &ws).unwrap();
        assert_eq!(read_file(&file, &ws).unwrap(), "second");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_write_bytes() {
        let (dir, ws) = temp_dir();
        let file = dir.join("binary.dat");

        let data: Vec<u8> = (0..=255).collect();
        write_file_bytes(&file, &data, &ws).unwrap();
        let read_back = read_file_bytes(&file, &ws).unwrap();
        assert_eq!(read_back, data);

        fs::remove_dir_all(&dir).ok();
    }

    // --- Sandbox enforcement tests ---

    #[test]
    fn test_sandbox_rejects_path_traversal() {
        let (dir, ws) = temp_dir();
        let escaped = dir.join("..").join("..").join("etc").join("passwd");
        let result = read_file(&escaped, &ws);
        assert!(
            matches!(result, Err(FsError::OutsideWorkspace(_))),
            "expected OutsideWorkspace, got: {:?}",
            result
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sandbox_allows_path_within_workspace() {
        let (dir, ws) = temp_dir();
        let nested = dir.join("sub").join("deep").join("file.txt");
        // Should not fail sandbox check (file will be created).
        write_file(&nested, "ok", &ws).unwrap();
        assert_eq!(read_file(&nested, &ws).unwrap(), "ok");
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_rejection_on_read() {
        let (dir, ws) = temp_dir();
        let real = dir.join("real.txt");
        let link = dir.join("link.txt");
        fs::write(&real, "secret").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = read_file(&link, &ws);
        assert!(
            matches!(result, Err(FsError::SymlinkNotAllowed(_))),
            "expected SymlinkNotAllowed, got: {:?}",
            result
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_rejection_on_delete() {
        let (dir, ws) = temp_dir();
        let real = dir.join("real.txt");
        let link = dir.join("link.txt");
        fs::write(&real, "data").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = delete_file(&link, &ws);
        assert!(
            matches!(result, Err(FsError::SymlinkNotAllowed(_))),
            "expected SymlinkNotAllowed, got: {:?}",
            result
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_list_directory_skips_broken_symlinks() {
        let (dir, ws) = temp_dir();
        fs::write(dir.join("good.txt"), "ok").unwrap();
        // Create a broken symlink pointing to a non-existent target.
        std::os::unix::fs::symlink(dir.join("nonexistent_target"), dir.join("broken_link"))
            .unwrap();

        let entries = list_directory(&dir, &ws).unwrap();
        // The broken symlink should be skipped.
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "good.txt");

        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_search_text_skips_symlinks() {
        let (dir, ws) = temp_dir();
        let real = dir.join("real.txt");
        fs::write(&real, "needle in real file").unwrap();
        std::os::unix::fs::symlink(&real, dir.join("link.txt")).unwrap();

        let matches = search_text(&dir, "needle", 100, &ws).unwrap();
        // Only the real file should match, not the symlink.
        assert_eq!(matches.len(), 1);
        assert!(matches[0].path.ends_with("real.txt"));

        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_dir_permissions_0o700() {
        let (dir, ws) = temp_dir();
        let nested = dir.join("restricted").join("inner");
        let file = nested.join("file.txt");
        write_file(&file, "content", &ws).unwrap();

        use std::os::unix::fs::PermissionsExt;
        let meta = fs::metadata(dir.join("restricted")).unwrap();
        assert_eq!(
            meta.permissions().mode() & 0o777,
            0o700,
            "directory should have mode 0o700"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
