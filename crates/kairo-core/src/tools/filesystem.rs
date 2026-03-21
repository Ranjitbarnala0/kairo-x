//! Filesystem operations for the KAIRO-X agent.
//!
//! Provides a safe, structured interface for file system interactions:
//! reading, writing, listing, searching, querying metadata, and deleting.
//! All operations return `Result` types with descriptive errors.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, trace};

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
// Operations
// ---------------------------------------------------------------------------

/// Read the entire contents of a file as a UTF-8 string.
pub fn read_file(path: &Path) -> Result<String, FsError> {
    debug!(path = %path.display(), "reading file");

    if !path.exists() {
        return Err(FsError::NotFound(path.to_path_buf()));
    }
    if path.is_dir() {
        return Err(FsError::NotAFile(path.to_path_buf()));
    }

    let bytes = std::fs::read(path).map_err(|e| FsError::from_io(path, e))?;
    String::from_utf8(bytes).map_err(|e| FsError::Utf8 {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read raw bytes from a file.
pub fn read_file_bytes(path: &Path) -> Result<Vec<u8>, FsError> {
    debug!(path = %path.display(), "reading file bytes");

    if !path.exists() {
        return Err(FsError::NotFound(path.to_path_buf()));
    }
    if path.is_dir() {
        return Err(FsError::NotAFile(path.to_path_buf()));
    }

    std::fs::read(path).map_err(|e| FsError::from_io(path, e))
}

/// Write string content to a file, creating parent directories as needed.
///
/// If the file exists, it is overwritten. If it does not exist, it is created.
pub fn write_file(path: &Path, content: &str) -> Result<(), FsError> {
    debug!(path = %path.display(), len = content.len(), "writing file");

    // Ensure parent directory exists.
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::from_io(parent, e))?;
        }
    }

    std::fs::write(path, content).map_err(|e| FsError::from_io(path, e))
}

/// Write raw bytes to a file, creating parent directories as needed.
pub fn write_file_bytes(path: &Path, content: &[u8]) -> Result<(), FsError> {
    debug!(path = %path.display(), len = content.len(), "writing file bytes");

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::from_io(parent, e))?;
        }
    }

    std::fs::write(path, content).map_err(|e| FsError::from_io(path, e))
}

/// List the contents of a directory, sorted by name.
///
/// Returns entries for immediate children only (non-recursive).
pub fn list_directory(path: &Path) -> Result<Vec<DirEntry>, FsError> {
    debug!(path = %path.display(), "listing directory");

    if !path.exists() {
        return Err(FsError::NotFound(path.to_path_buf()));
    }
    if !path.is_dir() {
        return Err(FsError::NotADirectory(path.to_path_buf()));
    }

    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(path).map_err(|e| FsError::from_io(path, e))?;

    for entry_result in read_dir {
        let entry = entry_result.map_err(|e| FsError::from_io(path, e))?;
        let metadata = entry.metadata().map_err(|e| FsError::from_io(&entry.path(), e))?;
        let name = entry
            .file_name()
            .to_string_lossy()
            .into_owned();

        entries.push(DirEntry {
            name,
            path: entry.path(),
            is_dir: metadata.is_dir(),
            size: if metadata.is_file() {
                metadata.len()
            } else {
                0
            },
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    trace!(path = %path.display(), count = entries.len(), "directory listed");
    Ok(entries)
}

/// Search for a text pattern in files under a directory (recursive).
///
/// Performs a simple line-by-line substring search (case-sensitive).
/// Skips binary files and hidden directories (those starting with `.`).
///
/// Returns up to `max_results` matches.
pub fn search_text(
    dir: &Path,
    pattern: &str,
    max_results: usize,
) -> Result<Vec<SearchMatch>, FsError> {
    debug!(
        dir = %dir.display(),
        pattern,
        max_results,
        "searching text"
    );

    if !dir.exists() {
        return Err(FsError::NotFound(dir.to_path_buf()));
    }

    let mut matches = Vec::new();
    search_text_recursive(dir, pattern, max_results, &mut matches)?;
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

    let entries = std::fs::read_dir(dir).map_err(|e| FsError::from_io(dir, e))?;

    for entry_result in entries {
        if matches.len() >= max_results {
            break;
        }

        let entry = entry_result.map_err(|e| FsError::from_io(dir, e))?;
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden directories and files.
        if name.starts_with('.') {
            continue;
        }

        let file_type = entry.file_type().map_err(|e| FsError::from_io(&path, e))?;

        if file_type.is_dir() {
            search_text_recursive(&path, pattern, max_results, matches)?;
        } else if file_type.is_file() {
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
pub fn file_info(path: &Path) -> Result<FileInfo, FsError> {
    debug!(path = %path.display(), "querying file info");

    let metadata = std::fs::metadata(path).map_err(|e| FsError::from_io(path, e))?;
    let symlink_metadata =
        std::fs::symlink_metadata(path).map_err(|e| FsError::from_io(path, e))?;

    let modified = metadata
        .modified()
        .map_err(|e| FsError::from_io(path, e))?;
    let modified_dt: DateTime<Utc> = modified.into();

    Ok(FileInfo {
        path: path.to_path_buf(),
        size: metadata.len(),
        modified: modified_dt,
        is_dir: metadata.is_dir(),
        is_symlink: symlink_metadata.file_type().is_symlink(),
        readonly: metadata.permissions().readonly(),
    })
}

/// Delete a file. Returns an error if the path is a directory.
pub fn delete_file(path: &Path) -> Result<(), FsError> {
    debug!(path = %path.display(), "deleting file");

    if !path.exists() {
        return Err(FsError::NotFound(path.to_path_buf()));
    }
    if path.is_dir() {
        return Err(FsError::NotAFile(path.to_path_buf()));
    }

    std::fs::remove_file(path).map_err(|e| FsError::from_io(path, e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temporary directory for test isolation.
    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kairo_fs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_read_write_file() {
        let dir = temp_dir();
        let file = dir.join("hello.txt");

        write_file(&file, "Hello, KAIRO!").unwrap();
        let content = read_file(&file).unwrap();
        assert_eq!(content, "Hello, KAIRO!");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_nonexistent() {
        let result = read_file(Path::new("/nonexistent/path/file.txt"));
        assert!(matches!(result, Err(FsError::NotFound(_))));
    }

    #[test]
    fn test_read_directory_as_file() {
        let dir = temp_dir();
        let result = read_file(&dir);
        assert!(matches!(result, Err(FsError::NotAFile(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = temp_dir();
        let file = dir.join("a").join("b").join("c.txt");

        write_file(&file, "nested").unwrap();
        assert_eq!(read_file(&file).unwrap(), "nested");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_directory() {
        let dir = temp_dir();
        fs::write(dir.join("alpha.txt"), "a").unwrap();
        fs::write(dir.join("beta.txt"), "b").unwrap();
        fs::create_dir(dir.join("gamma")).unwrap();

        let entries = list_directory(&dir).unwrap();
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
        let result = list_directory(Path::new("/nonexistent/directory"));
        assert!(matches!(result, Err(FsError::NotFound(_))));
    }

    #[test]
    fn test_list_file_as_directory() {
        let dir = temp_dir();
        let file = dir.join("file.txt");
        fs::write(&file, "not a dir").unwrap();

        let result = list_directory(&file);
        assert!(matches!(result, Err(FsError::NotADirectory(_))));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_search_text() {
        let dir = temp_dir();
        fs::write(dir.join("a.rs"), "fn main() {\n    println!(\"hello\");\n}").unwrap();
        fs::write(dir.join("b.rs"), "fn other() {\n    println!(\"world\");\n}").unwrap();
        fs::write(dir.join("c.txt"), "no match here").unwrap();

        let matches = search_text(&dir, "println!", 100).unwrap();
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
        let dir = temp_dir();
        for i in 0..10 {
            fs::write(dir.join(format!("{i}.txt")), "needle").unwrap();
        }

        let matches = search_text(&dir, "needle", 3).unwrap();
        assert_eq!(matches.len(), 3);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_search_text_skips_hidden() {
        let dir = temp_dir();
        let hidden = dir.join(".hidden");
        fs::create_dir(&hidden).unwrap();
        fs::write(hidden.join("secret.txt"), "needle").unwrap();
        fs::write(dir.join("visible.txt"), "needle").unwrap();

        let matches = search_text(&dir, "needle", 100).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].path.ends_with("visible.txt"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info() {
        let dir = temp_dir();
        let file = dir.join("info.txt");
        fs::write(&file, "twelve chars").unwrap();

        let info = file_info(&file).unwrap();
        assert_eq!(info.size, 12);
        assert!(!info.is_dir);
        assert!(!info.is_symlink);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info_directory() {
        let dir = temp_dir();
        let info = file_info(&dir).unwrap();
        assert!(info.is_dir);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_info_nonexistent() {
        let result = file_info(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_file() {
        let dir = temp_dir();
        let file = dir.join("delete_me.txt");
        fs::write(&file, "bye").unwrap();

        assert!(file.exists());
        delete_file(&file).unwrap();
        assert!(!file.exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_nonexistent() {
        let result = delete_file(Path::new("/nonexistent/file.txt"));
        assert!(matches!(result, Err(FsError::NotFound(_))));
    }

    #[test]
    fn test_delete_directory_fails() {
        let dir = temp_dir();
        let result = delete_file(&dir);
        assert!(matches!(result, Err(FsError::NotAFile(_))));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_overwrite() {
        let dir = temp_dir();
        let file = dir.join("overwrite.txt");

        write_file(&file, "first").unwrap();
        write_file(&file, "second").unwrap();
        assert_eq!(read_file(&file).unwrap(), "second");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_write_bytes() {
        let dir = temp_dir();
        let file = dir.join("binary.dat");

        let data: Vec<u8> = (0..=255).collect();
        write_file_bytes(&file, &data).unwrap();
        let read_back = read_file_bytes(&file).unwrap();
        assert_eq!(read_back, data);

        fs::remove_dir_all(&dir).ok();
    }
}
