//! Core project fingerprinting logic (§14).
//!
//! Walks the project root to detect language, package manager, build tools,
//! test/lint/typecheck commands, source layout, and monorepo structure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

use super::frameworks::detect_framework;

/// Type alias used throughout this module for brevity.
pub type Language = DetectedLanguage;

// ---------------------------------------------------------------------------
// Language enum — extended for fingerprinting (superset of classify::Language)
// ---------------------------------------------------------------------------

/// Programming language detected in the project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetectedLanguage {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    CSharp,
    Ruby,
    Cpp,
    C,
    Unknown,
}

impl DetectedLanguage {
    /// Infer language from a file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" | "pyx" => Some(Self::Python),
            "ts" | "tsx" | "mts" | "cts" => Some(Self::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "cs" => Some(Self::CSharp),
            "rb" => Some(Self::Ruby),
            "cpp" | "cxx" | "cc" | "hpp" | "hxx" => Some(Self::Cpp),
            "c" | "h" => Some(Self::C),
            _ => None,
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::TypeScript => "TypeScript",
            Self::JavaScript => "JavaScript",
            Self::Go => "Go",
            Self::Java => "Java",
            Self::CSharp => "C#",
            Self::Ruby => "Ruby",
            Self::Cpp => "C++",
            Self::C => "C",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for DetectedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// PackageManager enum
// ---------------------------------------------------------------------------

/// Package manager / build system detected in the project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageManager {
    Npm,
    Yarn,
    Pnpm,
    Cargo,
    GoMod,
    Pip,
    Poetry,
    Maven,
    Gradle,
}

impl PackageManager {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Npm => "npm",
            Self::Yarn => "yarn",
            Self::Pnpm => "pnpm",
            Self::Cargo => "cargo",
            Self::GoMod => "go mod",
            Self::Pip => "pip",
            Self::Poetry => "poetry",
            Self::Maven => "maven",
            Self::Gradle => "gradle",
        }
    }
}

impl std::fmt::Display for PackageManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// ProjectFingerprint — the output of fingerprinting
// ---------------------------------------------------------------------------

/// Complete fingerprint of a project's toolchain and layout.
///
/// Produced by [`fingerprint_project`] and consumed by the verification,
/// enforcement, and context subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFingerprint {
    /// Languages detected with file counts, sorted descending by count.
    pub languages: Vec<(DetectedLanguage, usize)>,
    /// The dominant language by file count.
    pub primary_language: DetectedLanguage,
    /// Detected package manager (if any).
    pub package_manager: Option<PackageManager>,
    /// Shell command to build the project.
    pub build_command: Option<String>,
    /// Shell command to run tests.
    pub test_command: Option<String>,
    /// Shell command to run the linter.
    pub lint_command: Option<String>,
    /// Shell command to run the type checker.
    pub typecheck_command: Option<String>,
    /// Detected framework (e.g., "Next.js", "Axum", "FastAPI").
    pub framework: Option<String>,
    /// Root directory containing source code (relative to project root).
    pub source_root: Option<PathBuf>,
    /// Root directory containing tests (relative to project root).
    pub test_root: Option<PathBuf>,
    /// Whether this project appears to be a monorepo.
    pub monorepo: bool,
}

impl ProjectFingerprint {
    /// Whether any build toolchain was detected at all.
    pub fn has_toolchain(&self) -> bool {
        self.build_command.is_some()
            || self.test_command.is_some()
            || self.lint_command.is_some()
    }

    /// Whether deterministic verification is possible (at least build or test).
    pub fn can_verify_deterministically(&self) -> bool {
        self.build_command.is_some() || self.test_command.is_some()
    }

    /// List of verification commands in standard chain order.
    /// Returns (step_name, command) pairs.
    pub fn verification_chain(&self) -> Vec<(&'static str, &str)> {
        let mut chain = Vec::new();
        if let Some(ref cmd) = self.build_command {
            chain.push(("build", cmd.as_str()));
        }
        if let Some(ref cmd) = self.lint_command {
            chain.push(("lint", cmd.as_str()));
        }
        if let Some(ref cmd) = self.typecheck_command {
            chain.push(("typecheck", cmd.as_str()));
        }
        if let Some(ref cmd) = self.test_command {
            chain.push(("test", cmd.as_str()));
        }
        chain
    }
}

// ---------------------------------------------------------------------------
// Fingerprinting entry point
// ---------------------------------------------------------------------------

/// Fingerprint a project rooted at the given path.
///
/// This is the primary entry point. It:
/// 1. Walks the directory tree to count files by language
/// 2. Detects manifest files (Cargo.toml, package.json, go.mod, etc.)
/// 3. Reads manifest files to extract scripts/commands
/// 4. Detects frameworks from dependencies
/// 5. Identifies source and test roots
/// 6. Checks for monorepo markers
///
/// Errors are logged but do not abort — partial fingerprints are still useful.
pub fn fingerprint_project(root: &Path) -> ProjectFingerprint {
    debug!("Fingerprinting project at: {}", root.display());

    // Step 1: Count files by language
    let lang_counts = count_languages(root);
    let mut languages: Vec<(DetectedLanguage, usize)> = lang_counts
        .into_iter()
        .filter(|(lang, _)| *lang != DetectedLanguage::Unknown)
        .collect();
    languages.sort_by(|a, b| b.1.cmp(&a.1));

    let primary_language = languages
        .first()
        .map(|(lang, _)| *lang)
        .unwrap_or(DetectedLanguage::Unknown);

    // Step 2-3: Detect toolchain from manifest files
    let toolchain = detect_toolchain(root);

    // Step 4: Detect framework
    let framework_info = detect_framework(root, primary_language, &toolchain);
    let framework = framework_info.map(|fi| fi.name);

    // Step 5: Detect source/test roots
    let source_root = detect_source_root(root, primary_language);
    let test_root = detect_test_root(root, primary_language);

    // Step 6: Detect monorepo
    let monorepo = detect_monorepo(root);

    ProjectFingerprint {
        languages,
        primary_language,
        package_manager: toolchain.package_manager,
        build_command: toolchain.build_command,
        test_command: toolchain.test_command,
        lint_command: toolchain.lint_command,
        typecheck_command: toolchain.typecheck_command,
        framework,
        source_root,
        test_root,
        monorepo,
    }
}

// ---------------------------------------------------------------------------
// Toolchain detection result (internal)
// ---------------------------------------------------------------------------

/// Intermediate result from manifest file analysis.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolchainInfo {
    pub package_manager: Option<PackageManager>,
    pub build_command: Option<String>,
    pub test_command: Option<String>,
    pub lint_command: Option<String>,
    pub typecheck_command: Option<String>,
    /// Raw dependencies extracted from manifest (for framework detection).
    pub dependencies: Vec<String>,
}

// ---------------------------------------------------------------------------
// Language counting
// ---------------------------------------------------------------------------

/// Walk the directory tree and count source files by language.
///
/// Skips hidden directories, node_modules, target, vendor, dist, build,
/// and other common non-source directories.
fn count_languages(root: &Path) -> HashMap<DetectedLanguage, usize> {
    let mut counts: HashMap<DetectedLanguage, usize> = HashMap::new();
    walk_for_languages(root, root, &mut counts, 0);
    counts
}

/// Maximum directory depth to traverse (prevents runaway on deep trees).
const MAX_WALK_DEPTH: u32 = 20;

/// Directories to skip during language counting.
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "vendor",
    "dist",
    "build",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    ".next",
    ".nuxt",
    "coverage",
    ".turbo",
    "out",
    "bin",
    "obj",
    ".gradle",
    ".idea",
    ".vscode",
];

fn walk_for_languages(
    dir: &Path,
    _root: &Path,
    counts: &mut HashMap<DetectedLanguage, usize>,
    depth: u32,
) {
    if depth > MAX_WALK_DEPTH {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            debug!("Cannot read directory {}: {}", dir.display(), e);
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden files and directories (starting with '.')
        if name.starts_with('.') {
            continue;
        }

        // Use symlink_metadata to avoid following symlinks. This prevents
        // infinite loops on circular symlinks and avoids double-counting
        // files that are symlinked from elsewhere in the tree.
        let file_type = match std::fs::symlink_metadata(&path) {
            Ok(m) => m.file_type(),
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_dir() {
            // Skip known non-source directories
            if SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            walk_for_languages(&path, _root, counts, depth + 1);
        } else if file_type.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if let Some(lang) = DetectedLanguage::from_extension(ext) {
                    *counts.entry(lang).or_insert(0) += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Toolchain detection from manifest files
// ---------------------------------------------------------------------------

/// Detect the build toolchain by checking for known manifest files.
///
/// Checks in priority order: Cargo.toml, package.json, go.mod,
/// pyproject.toml, setup.py, pom.xml, build.gradle.
fn detect_toolchain(root: &Path) -> ToolchainInfo {
    // Try each manifest file in turn. First match wins for package manager,
    // but we merge commands from multiple sources if present.

    if root.join("Cargo.toml").exists() {
        return detect_rust_toolchain(root);
    }

    if root.join("package.json").exists() {
        return detect_node_toolchain(root);
    }

    if root.join("go.mod").exists() {
        return detect_go_toolchain(root);
    }

    if root.join("pyproject.toml").exists() {
        return detect_python_toolchain_pyproject(root);
    }

    if root.join("setup.py").exists() || root.join("setup.cfg").exists() {
        return detect_python_toolchain_setup(root);
    }

    if root.join("pom.xml").exists() {
        return detect_maven_toolchain(root);
    }

    if root.join("build.gradle").exists() || root.join("build.gradle.kts").exists() {
        return detect_gradle_toolchain(root);
    }

    if root.join("Gemfile").exists() {
        return detect_ruby_toolchain(root);
    }

    ToolchainInfo::default()
}

// ---------------------------------------------------------------------------
// Rust toolchain
// ---------------------------------------------------------------------------

fn detect_rust_toolchain(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo {
        package_manager: Some(PackageManager::Cargo),
        build_command: Some("cargo build".to_string()),
        test_command: Some("cargo test".to_string()),
        lint_command: Some("cargo clippy -- -D warnings".to_string()),
        typecheck_command: Some("cargo check".to_string()),
        dependencies: Vec::new(),
    };

    // Parse Cargo.toml for dependencies (used in framework detection)
    if let Ok(content) = fs::read_to_string(root.join("Cargo.toml")) {
        if let Ok(parsed) = content.parse::<toml::Table>() {
            extract_cargo_dependencies(&parsed, &mut info.dependencies);

            // Check for workspace (monorepo-like)
            if parsed.get("workspace").is_some() {
                // Workspace: test all members
                info.test_command = Some("cargo test --workspace".to_string());
                info.build_command = Some("cargo build --workspace".to_string());
            }
        }
    }

    // Check for cargo-nextest
    if root.join(".config/nextest.toml").exists() {
        info.test_command = Some("cargo nextest run".to_string());
    }

    info
}

fn extract_cargo_dependencies(table: &toml::Table, deps: &mut Vec<String>) {
    for section in &["dependencies", "dev-dependencies"] {
        if let Some(toml::Value::Table(dep_table)) = table.get(*section) {
            for key in dep_table.keys() {
                deps.push(key.clone());
            }
        }
    }
    // Also check workspace dependencies
    if let Some(toml::Value::Table(ws)) = table.get("workspace") {
        if let Some(toml::Value::Table(ws_deps)) = ws.get("dependencies") {
            for key in ws_deps.keys() {
                deps.push(key.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Node.js (npm / yarn / pnpm) toolchain
// ---------------------------------------------------------------------------

fn detect_node_toolchain(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo::default();

    // Determine package manager from lock files
    if root.join("pnpm-lock.yaml").exists() {
        info.package_manager = Some(PackageManager::Pnpm);
    } else if root.join("yarn.lock").exists() {
        info.package_manager = Some(PackageManager::Yarn);
    } else {
        info.package_manager = Some(PackageManager::Npm);
    }

    let run_prefix = match info.package_manager {
        Some(PackageManager::Pnpm) => "pnpm",
        Some(PackageManager::Yarn) => "yarn",
        _ => "npm run",
    };

    // Parse package.json for scripts and dependencies
    let pkg_path = root.join("package.json");
    if let Ok(content) = fs::read_to_string(&pkg_path) {
        match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(pkg) => {
                // Extract scripts
                if let Some(scripts) = pkg.get("scripts").and_then(|s| s.as_object()) {
                    if scripts.contains_key("build") {
                        info.build_command = Some(format!("{run_prefix} build"));
                    }

                    if scripts.contains_key("test") {
                        info.test_command = Some(format!("{run_prefix} test"));
                    }

                    // Lint: try "lint", then "eslint"
                    if scripts.contains_key("lint") {
                        info.lint_command = Some(format!("{run_prefix} lint"));
                    } else if scripts.contains_key("eslint") {
                        info.lint_command = Some(format!("{run_prefix} eslint"));
                    }

                    // Typecheck: try "typecheck", "type-check", "tsc"
                    if scripts.contains_key("typecheck") {
                        info.typecheck_command = Some(format!("{run_prefix} typecheck"));
                    } else if scripts.contains_key("type-check") {
                        info.typecheck_command = Some(format!("{run_prefix} type-check"));
                    } else if scripts.contains_key("tsc") {
                        info.typecheck_command = Some(format!("{run_prefix} tsc"));
                    }
                }

                // Extract dependencies for framework detection
                for dep_key in &["dependencies", "devDependencies", "peerDependencies"] {
                    if let Some(deps) = pkg.get(*dep_key).and_then(|d| d.as_object()) {
                        for key in deps.keys() {
                            info.dependencies.push(key.clone());
                        }
                    }
                }

                // If no explicit typecheck but typescript is a dependency, add tsc
                if info.typecheck_command.is_none()
                    && info.dependencies.iter().any(|d| d == "typescript")
                {
                    info.typecheck_command = Some("npx tsc --noEmit".to_string());
                }
            }
            Err(e) => {
                warn!("Failed to parse package.json: {}", e);
            }
        }
    }

    info
}

// ---------------------------------------------------------------------------
// Go toolchain
// ---------------------------------------------------------------------------

fn detect_go_toolchain(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo {
        package_manager: Some(PackageManager::GoMod),
        build_command: Some("go build ./...".to_string()),
        test_command: Some("go test ./...".to_string()),
        lint_command: None,
        typecheck_command: Some("go vet ./...".to_string()),
        dependencies: Vec::new(),
    };

    // Check for golangci-lint config
    let lint_configs = [
        ".golangci.yml",
        ".golangci.yaml",
        ".golangci.toml",
        ".golangci.json",
    ];
    for cfg in &lint_configs {
        if root.join(cfg).exists() {
            info.lint_command = Some("golangci-lint run".to_string());
            break;
        }
    }

    // If no golangci-lint config, default to go vet (already set as typecheck)
    if info.lint_command.is_none() {
        info.lint_command = Some("go vet ./...".to_string());
    }

    // Parse go.mod for dependencies
    if let Ok(content) = fs::read_to_string(root.join("go.mod")) {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("require") || trimmed.is_empty() || trimmed == ")" {
                continue;
            }
            // Lines like: github.com/gin-gonic/gin v1.9.0
            if let Some(dep) = trimmed.split_whitespace().next() {
                if dep.contains('/') {
                    info.dependencies.push(dep.to_string());
                }
            }
        }
    }

    info
}

// ---------------------------------------------------------------------------
// Python toolchain (pyproject.toml)
// ---------------------------------------------------------------------------

fn detect_python_toolchain_pyproject(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo {
        package_manager: None,
        build_command: None,
        test_command: None,
        lint_command: None,
        typecheck_command: None,
        dependencies: Vec::new(),
    };

    if let Ok(content) = fs::read_to_string(root.join("pyproject.toml")) {
        if let Ok(parsed) = content.parse::<toml::Table>() {
            // Detect Poetry
            if parsed
                .get("tool")
                .and_then(|t| t.get("poetry"))
                .is_some()
            {
                info.package_manager = Some(PackageManager::Poetry);

                // Extract poetry dependencies
                if let Some(deps) = parsed
                    .get("tool")
                    .and_then(|t| t.get("poetry"))
                    .and_then(|p| p.get("dependencies"))
                    .and_then(|d| d.as_table())
                {
                    for key in deps.keys() {
                        info.dependencies.push(key.clone());
                    }
                }

                // Poetry dev dependencies
                if let Some(deps) = parsed
                    .get("tool")
                    .and_then(|t| t.get("poetry"))
                    .and_then(|p| p.get("group"))
                    .and_then(|g| g.get("dev"))
                    .and_then(|d| d.get("dependencies"))
                    .and_then(|d| d.as_table())
                {
                    for key in deps.keys() {
                        info.dependencies.push(key.clone());
                    }
                }
            } else {
                info.package_manager = Some(PackageManager::Pip);

                // PEP 621 project.dependencies
                if let Some(deps) = parsed
                    .get("project")
                    .and_then(|p| p.get("dependencies"))
                    .and_then(|d| d.as_array())
                {
                    for dep in deps {
                        if let Some(s) = dep.as_str() {
                            // Extract package name before version specifier
                            let name = s
                                .split(&['>', '<', '=', '~', '!', ';', '['][..])
                                .next()
                                .unwrap_or(s)
                                .trim();
                            info.dependencies.push(name.to_string());
                        }
                    }
                }

                // PEP 621 optional-dependencies (e.g., dev extras)
                if let Some(opt_deps) = parsed
                    .get("project")
                    .and_then(|p| p.get("optional-dependencies"))
                    .and_then(|d| d.as_table())
                {
                    for (_group, deps) in opt_deps {
                        if let Some(arr) = deps.as_array() {
                            for dep in arr {
                                if let Some(s) = dep.as_str() {
                                    let name = s
                                        .split(&['>', '<', '=', '~', '!', ';', '['][..])
                                        .next()
                                        .unwrap_or(s)
                                        .trim();
                                    info.dependencies.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Detect pytest configuration
            let has_pytest_section = parsed
                .get("tool")
                .and_then(|t| t.get("pytest"))
                .is_some();
            let has_pytest_dep = info.dependencies.iter().any(|d| d == "pytest");

            if has_pytest_section || has_pytest_dep {
                info.test_command = Some("pytest".to_string());
            } else {
                info.test_command = Some("python -m unittest discover".to_string());
            }

            // Detect ruff
            let has_ruff_section = parsed
                .get("tool")
                .and_then(|t| t.get("ruff"))
                .is_some();
            let has_ruff_dep = info.dependencies.iter().any(|d| d == "ruff");

            if has_ruff_section || has_ruff_dep {
                info.lint_command = Some("ruff check .".to_string());
            } else if info.dependencies.iter().any(|d| d == "flake8") {
                info.lint_command = Some("flake8 .".to_string());
            } else if info.dependencies.iter().any(|d| d == "pylint") {
                info.lint_command = Some("pylint .".to_string());
            }

            // Detect mypy / pyright
            let has_mypy_section = parsed
                .get("tool")
                .and_then(|t| t.get("mypy"))
                .is_some();
            let has_mypy_dep = info.dependencies.iter().any(|d| d == "mypy");

            if has_mypy_section || has_mypy_dep {
                info.typecheck_command = Some("mypy .".to_string());
            } else if info.dependencies.iter().any(|d| d == "pyright") {
                info.typecheck_command = Some("pyright".to_string());
            }
        }
    }

    // Python doesn't have a traditional "build" command, but we can check for
    // build backends
    if root.join("setup.py").exists() || root.join("setup.cfg").exists() {
        info.build_command = Some("python -m build".to_string());
    }

    info
}

// ---------------------------------------------------------------------------
// Python toolchain (setup.py / setup.cfg)
// ---------------------------------------------------------------------------

fn detect_python_toolchain_setup(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo {
        package_manager: Some(PackageManager::Pip),
        build_command: Some("python setup.py build".to_string()),
        test_command: None,
        lint_command: None,
        typecheck_command: None,
        dependencies: Vec::new(),
    };

    // Check for common test runners
    if root.join("pytest.ini").exists()
        || root.join("conftest.py").exists()
        || root.join("tests").is_dir()
    {
        info.test_command = Some("pytest".to_string());
    } else {
        info.test_command = Some("python -m unittest discover".to_string());
    }

    // Check for linter configs
    if root.join(".flake8").exists() || root.join("setup.cfg").exists() {
        info.lint_command = Some("flake8 .".to_string());
    }

    // Check for mypy config
    if root.join("mypy.ini").exists() || root.join(".mypy.ini").exists() {
        info.typecheck_command = Some("mypy .".to_string());
    }

    info
}

// ---------------------------------------------------------------------------
// Maven toolchain
// ---------------------------------------------------------------------------

fn detect_maven_toolchain(root: &Path) -> ToolchainInfo {
    let mvn = if root.join("mvnw").exists() {
        "./mvnw"
    } else {
        "mvn"
    };

    ToolchainInfo {
        package_manager: Some(PackageManager::Maven),
        build_command: Some(format!("{mvn} compile")),
        test_command: Some(format!("{mvn} test")),
        lint_command: Some(format!("{mvn} checkstyle:check")),
        typecheck_command: Some(format!("{mvn} compile")), // javac is the type checker
        dependencies: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Gradle toolchain
// ---------------------------------------------------------------------------

fn detect_gradle_toolchain(root: &Path) -> ToolchainInfo {
    let gradle = if root.join("gradlew").exists() {
        "./gradlew"
    } else {
        "gradle"
    };

    ToolchainInfo {
        package_manager: Some(PackageManager::Gradle),
        build_command: Some(format!("{gradle} build")),
        test_command: Some(format!("{gradle} test")),
        lint_command: Some(format!("{gradle} check")),
        typecheck_command: Some(format!("{gradle} compileJava")),
        dependencies: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Ruby toolchain
// ---------------------------------------------------------------------------

fn detect_ruby_toolchain(root: &Path) -> ToolchainInfo {
    let mut info = ToolchainInfo {
        package_manager: None, // Ruby uses Bundler but it's not in our enum
        build_command: None,
        test_command: None,
        lint_command: None,
        typecheck_command: None,
        dependencies: Vec::new(),
    };

    // Check for Rakefile
    if root.join("Rakefile").exists() {
        info.build_command = Some("bundle exec rake build".to_string());
        info.test_command = Some("bundle exec rake test".to_string());
    }

    // Check for RSpec
    if root.join("spec").is_dir() || root.join(".rspec").exists() {
        info.test_command = Some("bundle exec rspec".to_string());
    }

    // Check for RuboCop
    if root.join(".rubocop.yml").exists() {
        info.lint_command = Some("bundle exec rubocop".to_string());
    }

    info
}

// ---------------------------------------------------------------------------
// Source root detection
// ---------------------------------------------------------------------------

/// Detect the primary source code directory.
fn detect_source_root(root: &Path, language: DetectedLanguage) -> Option<PathBuf> {
    let candidates = match language {
        DetectedLanguage::Rust => vec!["src"],
        DetectedLanguage::Python => vec!["src", "lib"],
        DetectedLanguage::TypeScript | DetectedLanguage::JavaScript => vec!["src", "lib", "app", "pages"],
        DetectedLanguage::Go => vec!["cmd", "internal", "pkg"],
        DetectedLanguage::Java => vec!["src/main/java", "src/main"],
        DetectedLanguage::CSharp => vec!["src"],
        DetectedLanguage::Ruby => vec!["lib", "app"],
        DetectedLanguage::Cpp | DetectedLanguage::C => vec!["src", "lib", "include"],
        DetectedLanguage::Unknown => vec!["src", "lib"],
    };

    for candidate in candidates {
        let path = root.join(candidate);
        if path.is_dir() {
            return Some(PathBuf::from(candidate));
        }
    }

    // For Python, look for a package directory matching project name
    if language == DetectedLanguage::Python {
        if let Some(name) = root.file_name().and_then(|n| n.to_str()) {
            let pkg_name = name.replace('-', "_");
            let pkg_path = root.join(&pkg_name);
            if pkg_path.is_dir() {
                return Some(PathBuf::from(pkg_name));
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Test root detection
// ---------------------------------------------------------------------------

/// Detect the primary test directory.
fn detect_test_root(root: &Path, language: DetectedLanguage) -> Option<PathBuf> {
    let candidates = match language {
        DetectedLanguage::Rust => vec!["tests"],
        DetectedLanguage::Python => vec!["tests", "test"],
        DetectedLanguage::TypeScript | DetectedLanguage::JavaScript => {
            vec!["tests", "test", "__tests__", "spec"]
        }
        DetectedLanguage::Go => vec![], // Go tests live alongside source
        DetectedLanguage::Java => vec!["src/test/java", "src/test"],
        DetectedLanguage::CSharp => vec!["tests", "test"],
        DetectedLanguage::Ruby => vec!["spec", "test", "tests"],
        DetectedLanguage::Cpp | DetectedLanguage::C => vec!["tests", "test"],
        DetectedLanguage::Unknown => vec!["tests", "test"],
    };

    for candidate in candidates {
        let path = root.join(candidate);
        if path.is_dir() {
            return Some(PathBuf::from(candidate));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Monorepo detection
// ---------------------------------------------------------------------------

/// Detect whether this project is a monorepo.
///
/// Heuristics:
/// - Cargo workspace with members
/// - pnpm-workspace.yaml
/// - lerna.json
/// - nx.json
/// - turbo.json
/// - Multiple package.json files in subdirectories
/// - "packages/" or "apps/" directories
fn detect_monorepo(root: &Path) -> bool {
    // Direct monorepo config files
    let monorepo_markers = [
        "pnpm-workspace.yaml",
        "lerna.json",
        "nx.json",
        "turbo.json",
        "rush.json",
    ];

    for marker in &monorepo_markers {
        if root.join(marker).exists() {
            return true;
        }
    }

    // Cargo workspace
    if let Ok(content) = fs::read_to_string(root.join("Cargo.toml")) {
        if let Ok(parsed) = content.parse::<toml::Table>() {
            if let Some(ws) = parsed.get("workspace").and_then(|w| w.as_table()) {
                if ws.get("members").is_some() {
                    return true;
                }
            }
        }
    }

    // package.json with "workspaces" field
    if let Ok(content) = fs::read_to_string(root.join("package.json")) {
        if let Ok(pkg) = serde_json::from_str::<serde_json::Value>(&content) {
            if pkg.get("workspaces").is_some() {
                return true;
            }
        }
    }

    // Common monorepo directory structure
    let monorepo_dirs = ["packages", "apps", "libs", "services", "modules"];
    let subdir_count = monorepo_dirs
        .iter()
        .filter(|d| root.join(d).is_dir())
        .count();

    // If 2 or more of these directories exist, likely a monorepo
    subdir_count >= 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper to create a temp directory with specific files.
    #[allow(dead_code)]
    fn setup_project(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create temp dir");
        for file in files {
            let path = dir.path().join(file);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create parent dirs");
            }
            fs::write(&path, "").expect("create file");
        }
        dir
    }

    fn setup_project_with_content(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create temp dir");
        for (file, content) in files {
            let path = dir.path().join(file);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create parent dirs");
            }
            fs::write(&path, content).expect("create file");
        }
        dir
    }

    #[test]
    fn test_language_from_extension() {
        assert_eq!(DetectedLanguage::from_extension("rs"), Some(DetectedLanguage::Rust));
        assert_eq!(DetectedLanguage::from_extension("py"), Some(DetectedLanguage::Python));
        assert_eq!(DetectedLanguage::from_extension("ts"), Some(DetectedLanguage::TypeScript));
        assert_eq!(DetectedLanguage::from_extension("tsx"), Some(DetectedLanguage::TypeScript));
        assert_eq!(DetectedLanguage::from_extension("js"), Some(DetectedLanguage::JavaScript));
        assert_eq!(DetectedLanguage::from_extension("go"), Some(DetectedLanguage::Go));
        assert_eq!(DetectedLanguage::from_extension("java"), Some(DetectedLanguage::Java));
        assert_eq!(DetectedLanguage::from_extension("cs"), Some(DetectedLanguage::CSharp));
        assert_eq!(DetectedLanguage::from_extension("rb"), Some(DetectedLanguage::Ruby));
        assert_eq!(DetectedLanguage::from_extension("cpp"), Some(DetectedLanguage::Cpp));
        assert_eq!(DetectedLanguage::from_extension("c"), Some(DetectedLanguage::C));
        assert_eq!(DetectedLanguage::from_extension("xyz"), None);
    }

    #[test]
    fn test_fingerprint_rust_project() {
        let dir = setup_project_with_content(&[
            (
                "Cargo.toml",
                r#"
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1"
tokio = "1"
axum = "0.7"
"#,
            ),
            ("src/main.rs", "fn main() {}"),
            ("src/lib.rs", "pub fn hello() {}"),
            ("tests/integration.rs", "#[test] fn it_works() {}"),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Rust);
        assert_eq!(fp.package_manager, Some(PackageManager::Cargo));
        assert_eq!(fp.build_command.as_deref(), Some("cargo build"));
        assert_eq!(fp.test_command.as_deref(), Some("cargo test"));
        assert_eq!(
            fp.lint_command.as_deref(),
            Some("cargo clippy -- -D warnings")
        );
        assert_eq!(fp.typecheck_command.as_deref(), Some("cargo check"));
        assert_eq!(fp.source_root, Some(PathBuf::from("src")));
        assert_eq!(fp.test_root, Some(PathBuf::from("tests")));
        assert!(!fp.monorepo);
    }

    #[test]
    fn test_fingerprint_rust_workspace() {
        let dir = setup_project_with_content(&[
            (
                "Cargo.toml",
                r#"
[workspace]
members = ["crates/core", "crates/cli"]
"#,
            ),
            ("crates/core/src/lib.rs", ""),
            ("crates/cli/src/main.rs", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Rust);
        assert_eq!(fp.package_manager, Some(PackageManager::Cargo));
        assert_eq!(fp.build_command.as_deref(), Some("cargo build --workspace"));
        assert_eq!(fp.test_command.as_deref(), Some("cargo test --workspace"));
        assert!(fp.monorepo);
    }

    #[test]
    fn test_fingerprint_node_npm_project() {
        let dir = setup_project_with_content(&[
            (
                "package.json",
                r#"{
  "name": "myapp",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "lint": "eslint ."
  },
  "dependencies": {
    "react": "^18.0.0",
    "next": "^14.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "jest": "^29.0.0",
    "eslint": "^8.0.0"
  }
}"#,
            ),
            ("package-lock.json", "{}"),
            ("src/index.ts", ""),
            ("src/App.tsx", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::TypeScript);
        assert_eq!(fp.package_manager, Some(PackageManager::Npm));
        assert_eq!(fp.build_command.as_deref(), Some("npm run build"));
        assert_eq!(fp.test_command.as_deref(), Some("npm run test"));
        assert_eq!(fp.lint_command.as_deref(), Some("npm run lint"));
        assert_eq!(fp.source_root, Some(PathBuf::from("src")));
    }

    #[test]
    fn test_fingerprint_node_pnpm_project() {
        let dir = setup_project_with_content(&[
            (
                "package.json",
                r#"{
  "name": "myapp",
  "scripts": {
    "build": "vite build",
    "test": "vitest"
  }
}"#,
            ),
            ("pnpm-lock.yaml", ""),
            ("src/main.ts", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.package_manager, Some(PackageManager::Pnpm));
        assert_eq!(fp.build_command.as_deref(), Some("pnpm build"));
        assert_eq!(fp.test_command.as_deref(), Some("pnpm test"));
    }

    #[test]
    fn test_fingerprint_node_yarn_project() {
        let dir = setup_project_with_content(&[
            (
                "package.json",
                r#"{
  "name": "myapp",
  "scripts": {
    "build": "webpack",
    "test": "mocha"
  }
}"#,
            ),
            ("yarn.lock", ""),
            ("src/index.js", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.package_manager, Some(PackageManager::Yarn));
        assert_eq!(fp.build_command.as_deref(), Some("yarn build"));
    }

    #[test]
    fn test_fingerprint_go_project() {
        let dir = setup_project_with_content(&[
            (
                "go.mod",
                "module github.com/user/myapp\n\ngo 1.21\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.0\n)\n",
            ),
            ("cmd/server/main.go", "package main"),
            ("internal/handler.go", "package internal"),
            (".golangci.yml", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Go);
        assert_eq!(fp.package_manager, Some(PackageManager::GoMod));
        assert_eq!(fp.build_command.as_deref(), Some("go build ./..."));
        assert_eq!(fp.test_command.as_deref(), Some("go test ./..."));
        assert_eq!(fp.lint_command.as_deref(), Some("golangci-lint run"));
        assert_eq!(fp.typecheck_command.as_deref(), Some("go vet ./..."));
    }

    #[test]
    fn test_fingerprint_python_poetry_project() {
        let dir = setup_project_with_content(&[
            (
                "pyproject.toml",
                r#"
[tool.poetry]
name = "myapp"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
ruff = "^0.1"
mypy = "^1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88

[tool.mypy]
strict = true
"#,
            ),
            ("myapp/__init__.py", ""),
            ("myapp/main.py", ""),
            ("tests/test_main.py", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Python);
        assert_eq!(fp.package_manager, Some(PackageManager::Poetry));
        assert_eq!(fp.test_command.as_deref(), Some("pytest"));
        assert_eq!(fp.lint_command.as_deref(), Some("ruff check ."));
        assert_eq!(fp.typecheck_command.as_deref(), Some("mypy ."));
        assert_eq!(fp.test_root, Some(PathBuf::from("tests")));
    }

    #[test]
    fn test_fingerprint_python_pip_project() {
        let dir = setup_project_with_content(&[
            (
                "pyproject.toml",
                r#"
[project]
name = "myapp"
version = "0.1.0"
dependencies = [
    "flask>=2.0",
    "sqlalchemy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "flake8",
]
"#,
            ),
            ("src/app.py", ""),
            ("tests/test_app.py", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Python);
        assert_eq!(fp.package_manager, Some(PackageManager::Pip));
        assert_eq!(fp.test_command.as_deref(), Some("pytest"));
        assert_eq!(fp.lint_command.as_deref(), Some("flake8 ."));
    }

    #[test]
    fn test_fingerprint_maven_project() {
        let dir = setup_project_with_content(&[
            ("pom.xml", "<project></project>"),
            ("src/main/java/App.java", ""),
            ("src/test/java/AppTest.java", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Java);
        assert_eq!(fp.package_manager, Some(PackageManager::Maven));
        assert_eq!(fp.build_command.as_deref(), Some("mvn compile"));
        assert_eq!(fp.test_command.as_deref(), Some("mvn test"));
    }

    #[test]
    fn test_fingerprint_maven_wrapper() {
        let dir = setup_project_with_content(&[
            ("pom.xml", "<project></project>"),
            ("mvnw", "#!/bin/sh"),
            ("src/main/java/App.java", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.build_command.as_deref(), Some("./mvnw compile"));
    }

    #[test]
    fn test_fingerprint_gradle_project() {
        let dir = setup_project_with_content(&[
            ("build.gradle", "apply plugin: 'java'"),
            ("gradlew", "#!/bin/sh"),
            ("src/main/java/App.java", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.package_manager, Some(PackageManager::Gradle));
        assert_eq!(fp.build_command.as_deref(), Some("./gradlew build"));
        assert_eq!(fp.test_command.as_deref(), Some("./gradlew test"));
    }

    #[test]
    fn test_fingerprint_monorepo_lerna() {
        let dir = setup_project_with_content(&[
            ("lerna.json", "{}"),
            ("package.json", r#"{"name": "root"}"#),
            ("packages/a/package.json", "{}"),
            ("packages/b/package.json", "{}"),
        ]);

        let fp = fingerprint_project(dir.path());
        assert!(fp.monorepo);
    }

    #[test]
    fn test_fingerprint_monorepo_npm_workspaces() {
        let dir = setup_project_with_content(&[
            (
                "package.json",
                r#"{"name": "root", "workspaces": ["packages/*"]}"#,
            ),
            ("packages/a/index.js", ""),
        ]);

        let fp = fingerprint_project(dir.path());
        assert!(fp.monorepo);
    }

    #[test]
    fn test_fingerprint_empty_project() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let fp = fingerprint_project(dir.path());

        assert_eq!(fp.primary_language, DetectedLanguage::Unknown);
        assert!(fp.package_manager.is_none());
        assert!(fp.build_command.is_none());
        assert!(fp.test_command.is_none());
        assert!(fp.languages.is_empty());
        assert!(!fp.monorepo);
    }

    #[test]
    fn test_fingerprint_skips_node_modules() {
        let dir = setup_project_with_content(&[
            (
                "package.json",
                r#"{"name": "myapp", "scripts": {"build": "tsc"}}"#,
            ),
            ("src/index.ts", ""),
            ("node_modules/lodash/index.js", ""),
            ("node_modules/lodash/fp/index.js", ""),
            ("node_modules/react/index.js", ""),
        ]);

        let fp = fingerprint_project(dir.path());

        // Only the src/index.ts should count, not node_modules files
        let ts_count = fp
            .languages
            .iter()
            .find(|(l, _)| *l == DetectedLanguage::TypeScript)
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(ts_count, 1);

        // JS files from node_modules should NOT be counted
        let js_count = fp
            .languages
            .iter()
            .find(|(l, _)| *l == DetectedLanguage::JavaScript)
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(js_count, 0);
    }

    #[test]
    fn test_verification_chain() {
        let fp = ProjectFingerprint {
            languages: vec![(DetectedLanguage::Rust, 10)],
            primary_language: DetectedLanguage::Rust,
            package_manager: Some(PackageManager::Cargo),
            build_command: Some("cargo build".to_string()),
            test_command: Some("cargo test".to_string()),
            lint_command: Some("cargo clippy -- -D warnings".to_string()),
            typecheck_command: Some("cargo check".to_string()),
            framework: None,
            source_root: Some(PathBuf::from("src")),
            test_root: Some(PathBuf::from("tests")),
            monorepo: false,
        };

        let chain = fp.verification_chain();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0].0, "build");
        assert_eq!(chain[1].0, "lint");
        assert_eq!(chain[2].0, "typecheck");
        assert_eq!(chain[3].0, "test");
    }

    #[test]
    fn test_has_toolchain() {
        let mut fp = ProjectFingerprint {
            languages: vec![],
            primary_language: DetectedLanguage::Unknown,
            package_manager: None,
            build_command: None,
            test_command: None,
            lint_command: None,
            typecheck_command: None,
            framework: None,
            source_root: None,
            test_root: None,
            monorepo: false,
        };

        assert!(!fp.has_toolchain());

        fp.build_command = Some("make".to_string());
        assert!(fp.has_toolchain());
    }

    #[test]
    fn test_node_typecheck_inferred_from_typescript_dep() {
        let dir = setup_project_with_content(&[(
            "package.json",
            r#"{
  "name": "myapp",
  "scripts": {
    "build": "tsc"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}"#,
        )]);

        let fp = fingerprint_project(dir.path());
        assert!(fp.typecheck_command.is_some());
    }
}
