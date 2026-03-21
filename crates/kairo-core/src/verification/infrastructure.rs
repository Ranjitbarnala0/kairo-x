//! Verification infrastructure detection (§6.3).
//!
//! Before running verification, checks whether the project has the necessary
//! infrastructure (test framework, linter, type checker) and generates
//! prerequisite nodes to set them up if missing.
//!
//! These prerequisite nodes are inserted into the arena graph as dependencies
//! of the verification step, ensuring infrastructure is in place before any
//! automated checks run.

use crate::arena::node::Priority;
use crate::fingerprint::detector::{DetectedLanguage, ProjectFingerprint};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ---------------------------------------------------------------------------
// Prerequisite node
// ---------------------------------------------------------------------------

/// A prerequisite node that must be completed before verification can run.
///
/// These are generated when the project lacks expected infrastructure
/// (e.g., no test framework configured, no linter installed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrerequisiteNode {
    /// Title of the prerequisite task.
    pub title: String,
    /// Specification of what needs to be set up.
    pub spec: String,
    /// Priority — infrastructure setup is Critical because verification
    /// cannot proceed without it.
    pub priority: Priority,
    /// Which verification step this prerequisite enables.
    pub enables_step: String,
    /// Estimated files that will be created or modified.
    pub estimated_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// Prerequisite detection
// ---------------------------------------------------------------------------

/// Analyze a project fingerprint and generate prerequisite nodes for any
/// missing verification infrastructure.
///
/// Returns an empty vector if all expected infrastructure is present.
/// The `project_root` is used for additional filesystem checks beyond
/// what the fingerprint already captured.
pub fn plan_prerequisites(
    fingerprint: &ProjectFingerprint,
    _project_root: &Path,
) -> Vec<PrerequisiteNode> {
    let mut prerequisites = Vec::new();

    // Only generate prerequisites if we have *some* toolchain detected.
    // If there's no toolchain at all, fingerprinting didn't find a known project.
    if !fingerprint.has_toolchain()
        && fingerprint.primary_language == DetectedLanguage::Unknown
    {
        return prerequisites;
    }

    // Check for missing test infrastructure
    if fingerprint.test_command.is_none() {
        if let Some(node) = plan_test_infra(fingerprint) {
            prerequisites.push(node);
        }
    }

    // Check for missing lint infrastructure
    if fingerprint.lint_command.is_none() {
        if let Some(node) = plan_lint_infra(fingerprint) {
            prerequisites.push(node);
        }
    }

    // Check for missing type-check infrastructure
    // Only for languages where a separate type checker is meaningful
    if fingerprint.typecheck_command.is_none() {
        if let Some(node) = plan_typecheck_infra(fingerprint) {
            prerequisites.push(node);
        }
    }

    prerequisites
}

// ---------------------------------------------------------------------------
// Per-category planners
// ---------------------------------------------------------------------------

fn plan_test_infra(fp: &ProjectFingerprint) -> Option<PrerequisiteNode> {
    let (title, spec, files) = match fp.primary_language {
        DetectedLanguage::Rust => (
            "Set up Rust test infrastructure",
            "Configure test infrastructure for the Rust project:\n\
             1. Create tests/ directory with a basic integration test\n\
             2. Ensure #[cfg(test)] modules exist in key source files\n\
             3. Add dev-dependencies section to Cargo.toml if needed",
            vec![
                "tests/integration_test.rs".to_string(),
            ],
        ),
        DetectedLanguage::Python => (
            "Set up Python test infrastructure (pytest)",
            "Configure pytest for the Python project:\n\
             1. Add pytest to dev dependencies in pyproject.toml\n\
             2. Create conftest.py with common fixtures\n\
             3. Create tests/ directory with __init__.py\n\
             4. Add [tool.pytest.ini_options] to pyproject.toml\n\
             5. Create tests/test_smoke.py with a basic passing test",
            vec![
                "conftest.py".to_string(),
                "tests/__init__.py".to_string(),
                "tests/test_smoke.py".to_string(),
            ],
        ),
        DetectedLanguage::TypeScript => (
            "Set up TypeScript test infrastructure",
            "Configure a test runner for the TypeScript project:\n\
             1. Install vitest (or jest) as dev dependency\n\
             2. Add test script to package.json\n\
             3. Create vitest.config.ts with sensible defaults\n\
             4. Create tests/example.test.ts with a basic test\n\
             5. Ensure tsconfig includes test files",
            vec![
                "vitest.config.ts".to_string(),
                "tests/example.test.ts".to_string(),
            ],
        ),
        DetectedLanguage::JavaScript => (
            "Set up JavaScript test infrastructure",
            "Configure a test runner for the JavaScript project:\n\
             1. Install vitest (or jest) as dev dependency\n\
             2. Add test script to package.json\n\
             3. Create tests/example.test.js with a basic test",
            vec!["tests/example.test.js".to_string()],
        ),
        DetectedLanguage::Go => (
            "Set up Go test infrastructure",
            "Go has built-in testing. Create initial test files:\n\
             1. Create _test.go files alongside key source files\n\
             2. Create testdata/ directory for test fixtures if needed\n\
             3. Add testify to go.mod if assertion helpers are desired",
            vec!["main_test.go".to_string()],
        ),
        DetectedLanguage::Java => (
            "Set up Java test infrastructure (JUnit)",
            "Configure JUnit 5 for the Java project:\n\
             1. Add junit-jupiter dependency to pom.xml or build.gradle\n\
             2. Create src/test/java/ directory structure\n\
             3. Add a basic test class\n\
             4. Configure maven-surefire-plugin if using Maven",
            vec!["src/test/java/AppTest.java".to_string()],
        ),
        DetectedLanguage::Ruby => (
            "Set up Ruby test infrastructure (RSpec)",
            "Configure RSpec for the Ruby project:\n\
             1. Add rspec to Gemfile\n\
             2. Run bundle exec rspec --init\n\
             3. Create spec/spec_helper.rb\n\
             4. Create a basic spec file",
            vec![
                ".rspec".to_string(),
                "spec/spec_helper.rb".to_string(),
            ],
        ),
        _ => return None,
    };

    Some(PrerequisiteNode {
        title: title.to_string(),
        spec: spec.to_string(),
        priority: Priority::Critical,
        enables_step: "test".to_string(),
        estimated_files: files,
    })
}

fn plan_lint_infra(fp: &ProjectFingerprint) -> Option<PrerequisiteNode> {
    let (title, spec, files) = match fp.primary_language {
        DetectedLanguage::Rust => (
            "Configure Clippy linting for Rust",
            "Configure Clippy for the Rust project:\n\
             1. Create clippy.toml with project-specific rules\n\
             2. Add #![warn(clippy::all)] to lib.rs or main.rs\n\
             3. Configure deny list for critical lints",
            vec!["clippy.toml".to_string()],
        ),
        DetectedLanguage::Python => (
            "Set up Python linter (ruff)",
            "Configure ruff linter for the Python project:\n\
             1. Add ruff to dev dependencies\n\
             2. Add [tool.ruff] section to pyproject.toml\n\
             3. Configure line-length=88, select rules (E, F, W, I, N, UP)\n\
             4. Configure ruff format for consistent formatting",
            vec!["pyproject.toml".to_string()],
        ),
        DetectedLanguage::TypeScript | DetectedLanguage::JavaScript => (
            "Set up ESLint",
            "Configure ESLint for the project:\n\
             1. Install eslint and relevant plugins as dev dependencies\n\
             2. Create eslint.config.js (flat config format)\n\
             3. Add lint script to package.json\n\
             4. Configure TypeScript-aware rules if TypeScript is used",
            vec!["eslint.config.js".to_string()],
        ),
        DetectedLanguage::Go => (
            "Set up golangci-lint for Go",
            "Configure golangci-lint for the Go project:\n\
             1. Create .golangci.yml with recommended linters\n\
             2. Enable: errcheck, govet, staticcheck, unused, ineffassign\n\
             3. Configure timeout and exclude patterns for generated code",
            vec![".golangci.yml".to_string()],
        ),
        DetectedLanguage::Ruby => (
            "Set up RuboCop linter",
            "Configure RuboCop for the Ruby project:\n\
             1. Add rubocop to Gemfile\n\
             2. Create .rubocop.yml with sensible defaults\n\
             3. Enable recommended cops",
            vec![".rubocop.yml".to_string()],
        ),
        _ => return None,
    };

    Some(PrerequisiteNode {
        title: title.to_string(),
        spec: spec.to_string(),
        priority: Priority::Critical,
        enables_step: "lint".to_string(),
        estimated_files: files,
    })
}

fn plan_typecheck_infra(fp: &ProjectFingerprint) -> Option<PrerequisiteNode> {
    // Only generate type-check prerequisites for languages where a separate
    // type checker is meaningful and not already built into the compiler.
    let (title, spec, files) = match fp.primary_language {
        DetectedLanguage::Python => (
            "Set up Python type checking (mypy)",
            "Configure mypy for the Python project:\n\
             1. Add mypy to dev dependencies\n\
             2. Add [tool.mypy] section to pyproject.toml\n\
             3. Configure strict mode or incremental strictness\n\
             4. Add py.typed marker if this is a library",
            vec!["pyproject.toml".to_string()],
        ),
        DetectedLanguage::TypeScript => (
            "Configure TypeScript type checking",
            "Ensure TypeScript type checking is properly configured:\n\
             1. Create or update tsconfig.json with strict mode\n\
             2. Add typecheck script to package.json: tsc --noEmit\n\
             3. Ensure all source files are included in tsconfig",
            vec!["tsconfig.json".to_string()],
        ),
        DetectedLanguage::JavaScript => (
            "Add TypeScript type checking to JavaScript project",
            "Add optional type checking using TypeScript + JSDoc:\n\
             1. Install typescript as dev dependency\n\
             2. Create tsconfig.json with allowJs, checkJs, noEmit\n\
             3. Add typecheck script to package.json",
            vec!["tsconfig.json".to_string()],
        ),
        // Rust, Go, Java, C# have type checking built into compilation
        _ => return None,
    };

    Some(PrerequisiteNode {
        title: title.to_string(),
        spec: spec.to_string(),
        priority: Priority::Critical,
        enables_step: "typecheck".to_string(),
        estimated_files: files,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::detector::{DetectedLanguage, PackageManager};
    use std::path::PathBuf;

    fn make_fingerprint(
        language: DetectedLanguage,
        has_test: bool,
        has_lint: bool,
        has_typecheck: bool,
    ) -> ProjectFingerprint {
        ProjectFingerprint {
            languages: vec![(language, 10)],
            primary_language: language,
            package_manager: Some(PackageManager::Cargo),
            build_command: Some("cargo build".to_string()),
            test_command: if has_test {
                Some("cargo test".to_string())
            } else {
                None
            },
            lint_command: if has_lint {
                Some("cargo clippy".to_string())
            } else {
                None
            },
            typecheck_command: if has_typecheck {
                Some("cargo check".to_string())
            } else {
                None
            },
            framework: None,
            source_root: Some(PathBuf::from("src")),
            test_root: None,
            monorepo: false,
        }
    }

    #[test]
    fn test_no_prerequisites_when_complete() {
        let fp = make_fingerprint(DetectedLanguage::Rust, true, true, true);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.is_empty());
    }

    #[test]
    fn test_prerequisite_for_missing_test() {
        let fp = make_fingerprint(DetectedLanguage::Rust, false, true, true);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.iter().any(|p| p.enables_step == "test"));
    }

    #[test]
    fn test_prerequisite_for_missing_lint() {
        let fp = make_fingerprint(DetectedLanguage::Rust, true, false, true);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.iter().any(|p| p.enables_step == "lint"));
    }

    #[test]
    fn test_typecheck_prerequisite_for_typescript() {
        let mut fp = make_fingerprint(DetectedLanguage::TypeScript, true, true, false);
        fp.package_manager = Some(PackageManager::Npm);
        fp.build_command = Some("npm run build".to_string());
        fp.test_command = Some("npm run test".to_string());
        fp.lint_command = Some("npm run lint".to_string());
        fp.typecheck_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.iter().any(|p| p.enables_step == "typecheck"));
    }

    #[test]
    fn test_typecheck_prerequisite_for_python() {
        let mut fp = make_fingerprint(DetectedLanguage::Python, true, true, false);
        fp.package_manager = Some(PackageManager::Pip);
        fp.build_command = None;
        fp.test_command = Some("pytest".to_string());
        fp.lint_command = Some("ruff check .".to_string());
        fp.typecheck_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.iter().any(|p| p.enables_step == "typecheck"));
    }

    #[test]
    fn test_no_typecheck_prerequisite_for_go() {
        // Go doesn't need a separate type checker
        let mut fp = make_fingerprint(DetectedLanguage::Go, true, true, false);
        fp.package_manager = Some(PackageManager::GoMod);
        fp.build_command = Some("go build ./...".to_string());
        fp.test_command = Some("go test ./...".to_string());
        fp.lint_command = Some("golangci-lint run".to_string());
        fp.typecheck_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(!prereqs.iter().any(|p| p.enables_step == "typecheck"));
    }

    #[test]
    fn test_no_typecheck_prerequisite_for_rust() {
        // Rust type checking is built into cargo check
        let fp = make_fingerprint(DetectedLanguage::Rust, true, true, false);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(!prereqs.iter().any(|p| p.enables_step == "typecheck"));
    }

    #[test]
    fn test_prerequisite_priority_is_critical() {
        let fp = make_fingerprint(DetectedLanguage::Python, false, false, false);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        for prereq in &prereqs {
            assert_eq!(
                prereq.priority,
                Priority::Critical,
                "Infrastructure prerequisites must be Critical"
            );
        }
    }

    #[test]
    fn test_multiple_prerequisites() {
        let mut fp = make_fingerprint(DetectedLanguage::Python, false, false, false);
        fp.package_manager = Some(PackageManager::Pip);
        fp.build_command = Some("python -m build".to_string());
        fp.test_command = None;
        fp.lint_command = None;
        fp.typecheck_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        // Should have test, lint, and typecheck prerequisites
        assert!(prereqs.len() >= 3);
        assert!(prereqs.iter().any(|p| p.enables_step == "test"));
        assert!(prereqs.iter().any(|p| p.enables_step == "lint"));
        assert!(prereqs.iter().any(|p| p.enables_step == "typecheck"));
    }

    #[test]
    fn test_unknown_language_no_prerequisites() {
        let fp = ProjectFingerprint {
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

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        assert!(prereqs.is_empty());
    }

    #[test]
    fn test_prerequisite_has_estimated_files() {
        let fp = make_fingerprint(DetectedLanguage::TypeScript, false, false, false);
        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        for prereq in &prereqs {
            assert!(
                !prereq.estimated_files.is_empty(),
                "Prerequisite '{}' should have estimated files",
                prereq.title
            );
        }
    }

    #[test]
    fn test_go_test_prerequisite() {
        let mut fp = make_fingerprint(DetectedLanguage::Go, false, true, true);
        fp.package_manager = Some(PackageManager::GoMod);
        fp.build_command = Some("go build ./...".to_string());
        fp.lint_command = Some("golangci-lint run".to_string());
        fp.typecheck_command = Some("go vet ./...".to_string());
        fp.test_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        let test_prereq = prereqs.iter().find(|p| p.enables_step == "test");
        assert!(test_prereq.is_some());
        assert!(test_prereq.unwrap().title.contains("Go"));
    }

    #[test]
    fn test_java_test_prerequisite() {
        let mut fp = make_fingerprint(DetectedLanguage::Java, false, true, true);
        fp.package_manager = Some(PackageManager::Maven);
        fp.build_command = Some("mvn compile".to_string());
        fp.lint_command = Some("mvn checkstyle:check".to_string());
        fp.typecheck_command = Some("mvn compile".to_string());
        fp.test_command = None;

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        let test_prereq = prereqs.iter().find(|p| p.enables_step == "test");
        assert!(test_prereq.is_some());
        assert!(test_prereq.unwrap().title.contains("Java"));
    }

    #[test]
    fn test_ruby_lint_prerequisite() {
        let fp = ProjectFingerprint {
            languages: vec![(DetectedLanguage::Ruby, 5)],
            primary_language: DetectedLanguage::Ruby,
            package_manager: None,
            build_command: Some("bundle exec rake build".to_string()),
            test_command: Some("bundle exec rspec".to_string()),
            lint_command: None,
            typecheck_command: None,
            framework: None,
            source_root: None,
            test_root: None,
            monorepo: false,
        };

        let prereqs = plan_prerequisites(&fp, Path::new("/tmp"));
        let lint_prereq = prereqs.iter().find(|p| p.enables_step == "lint");
        assert!(lint_prereq.is_some());
        assert!(lint_prereq.unwrap().title.contains("RuboCop"));
    }
}
