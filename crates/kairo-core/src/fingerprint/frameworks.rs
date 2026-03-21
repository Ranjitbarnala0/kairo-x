//! Framework detection for project fingerprinting (§14).
//!
//! Detects popular frameworks by inspecting dependencies, configuration files,
//! and directory structure. Framework detection is language-aware: we only
//! check for frameworks that match the primary language.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::debug;

use super::detector::{DetectedLanguage, ToolchainInfo};

// ---------------------------------------------------------------------------
// FrameworkInfo
// ---------------------------------------------------------------------------

/// Information about a detected framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkInfo {
    /// Framework name (e.g., "Next.js", "Axum", "FastAPI").
    pub name: String,
    /// Framework category.
    pub category: FrameworkCategory,
    /// Version constraint string, if detectable.
    pub version: Option<String>,
}

/// Broad category of detected framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameworkCategory {
    /// Web framework (HTTP server, routing, etc.)
    Web,
    /// Frontend UI framework
    Frontend,
    /// Full-stack framework (server + client)
    FullStack,
    /// CLI framework
    Cli,
    /// Data / ML framework
    Data,
    /// Testing framework
    Testing,
    /// Other
    Other,
}

// ---------------------------------------------------------------------------
// Detection entry point
// ---------------------------------------------------------------------------

/// Detect the primary framework used in the project.
///
/// Returns `None` if no known framework is detected. When multiple frameworks
/// are present, returns the most specific / "dominant" one (e.g., Next.js
/// over bare React, Axum over bare Tokio).
pub(crate) fn detect_framework(
    root: &Path,
    language: DetectedLanguage,
    toolchain: &ToolchainInfo,
) -> Option<FrameworkInfo> {
    let deps = &toolchain.dependencies;

    let result = match language {
        DetectedLanguage::Rust => detect_rust_framework(root, deps),
        DetectedLanguage::TypeScript | DetectedLanguage::JavaScript => {
            detect_node_framework(root, deps)
        }
        DetectedLanguage::Python => detect_python_framework(root, deps),
        DetectedLanguage::Go => detect_go_framework(root, deps),
        DetectedLanguage::Java => detect_java_framework(root, deps),
        DetectedLanguage::Ruby => detect_ruby_framework(root, deps),
        DetectedLanguage::CSharp => detect_csharp_framework(root, deps),
        _ => None,
    };

    if let Some(ref fw) = result {
        debug!("Detected framework: {} ({:?})", fw.name, fw.category);
    }

    result
}

// ---------------------------------------------------------------------------
// Rust frameworks
// ---------------------------------------------------------------------------

fn detect_rust_framework(_root: &Path, deps: &[String]) -> Option<FrameworkInfo> {
    // Check most specific first

    // Actix-web
    if deps.iter().any(|d| d == "actix-web") {
        return Some(FrameworkInfo {
            name: "Actix Web".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Axum
    if deps.iter().any(|d| d == "axum") {
        return Some(FrameworkInfo {
            name: "Axum".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Rocket
    if deps.iter().any(|d| d == "rocket") {
        return Some(FrameworkInfo {
            name: "Rocket".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Warp
    if deps.iter().any(|d| d == "warp") {
        return Some(FrameworkInfo {
            name: "Warp".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Leptos (full-stack)
    if deps.iter().any(|d| d == "leptos") {
        return Some(FrameworkInfo {
            name: "Leptos".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Dioxus (frontend)
    if deps.iter().any(|d| d == "dioxus") {
        return Some(FrameworkInfo {
            name: "Dioxus".to_string(),
            category: FrameworkCategory::Frontend,
            version: None,
        });
    }

    // Yew (frontend)
    if deps.iter().any(|d| d == "yew") {
        return Some(FrameworkInfo {
            name: "Yew".to_string(),
            category: FrameworkCategory::Frontend,
            version: None,
        });
    }

    // Tauri (desktop)
    if deps.iter().any(|d| d == "tauri") {
        return Some(FrameworkInfo {
            name: "Tauri".to_string(),
            category: FrameworkCategory::Other,
            version: None,
        });
    }

    // Clap (CLI)
    if deps.iter().any(|d| d == "clap") {
        return Some(FrameworkInfo {
            name: "Clap".to_string(),
            category: FrameworkCategory::Cli,
            version: None,
        });
    }

    // Tonic (gRPC)
    if deps.iter().any(|d| d == "tonic") {
        return Some(FrameworkInfo {
            name: "Tonic".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Node.js / TypeScript frameworks
// ---------------------------------------------------------------------------

fn detect_node_framework(root: &Path, deps: &[String]) -> Option<FrameworkInfo> {
    // Check most specific / full-stack first

    // Next.js (full-stack React)
    if deps.iter().any(|d| d == "next") {
        return Some(FrameworkInfo {
            name: "Next.js".to_string(),
            category: FrameworkCategory::FullStack,
            version: find_dep_version(root, "next"),
        });
    }

    // Nuxt (full-stack Vue)
    if deps.iter().any(|d| d == "nuxt" || d == "nuxt3") {
        return Some(FrameworkInfo {
            name: "Nuxt".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // SvelteKit
    if deps.iter().any(|d| d == "@sveltejs/kit") {
        return Some(FrameworkInfo {
            name: "SvelteKit".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Remix
    if deps.iter().any(|d| d == "@remix-run/react" || d == "@remix-run/node") {
        return Some(FrameworkInfo {
            name: "Remix".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Astro
    if deps.iter().any(|d| d == "astro") {
        return Some(FrameworkInfo {
            name: "Astro".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Express.js (backend)
    if deps.iter().any(|d| d == "express") {
        return Some(FrameworkInfo {
            name: "Express".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Fastify
    if deps.iter().any(|d| d == "fastify") {
        return Some(FrameworkInfo {
            name: "Fastify".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // NestJS
    if deps.iter().any(|d| d == "@nestjs/core") {
        return Some(FrameworkInfo {
            name: "NestJS".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Hono
    if deps.iter().any(|d| d == "hono") {
        return Some(FrameworkInfo {
            name: "Hono".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // React (frontend only)
    if deps.iter().any(|d| d == "react") {
        return Some(FrameworkInfo {
            name: "React".to_string(),
            category: FrameworkCategory::Frontend,
            version: find_dep_version(root, "react"),
        });
    }

    // Vue.js
    if deps.iter().any(|d| d == "vue") {
        return Some(FrameworkInfo {
            name: "Vue.js".to_string(),
            category: FrameworkCategory::Frontend,
            version: None,
        });
    }

    // Svelte (without Kit)
    if deps.iter().any(|d| d == "svelte") {
        return Some(FrameworkInfo {
            name: "Svelte".to_string(),
            category: FrameworkCategory::Frontend,
            version: None,
        });
    }

    // Angular
    if deps.iter().any(|d| d == "@angular/core") {
        return Some(FrameworkInfo {
            name: "Angular".to_string(),
            category: FrameworkCategory::Frontend,
            version: None,
        });
    }

    // Electron
    if deps.iter().any(|d| d == "electron") {
        return Some(FrameworkInfo {
            name: "Electron".to_string(),
            category: FrameworkCategory::Other,
            version: None,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Python frameworks
// ---------------------------------------------------------------------------

fn detect_python_framework(_root: &Path, deps: &[String]) -> Option<FrameworkInfo> {
    // FastAPI
    if deps.iter().any(|d| d == "fastapi") {
        return Some(FrameworkInfo {
            name: "FastAPI".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Django
    if deps.iter().any(|d| d.to_lowercase() == "django") {
        return Some(FrameworkInfo {
            name: "Django".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Flask
    if deps.iter().any(|d| d.to_lowercase() == "flask") {
        return Some(FrameworkInfo {
            name: "Flask".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Starlette
    if deps.iter().any(|d| d == "starlette") {
        return Some(FrameworkInfo {
            name: "Starlette".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Litestar
    if deps.iter().any(|d| d == "litestar") {
        return Some(FrameworkInfo {
            name: "Litestar".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Streamlit
    if deps.iter().any(|d| d == "streamlit") {
        return Some(FrameworkInfo {
            name: "Streamlit".to_string(),
            category: FrameworkCategory::Data,
            version: None,
        });
    }

    // PyTorch
    if deps.iter().any(|d| d == "torch" || d == "pytorch") {
        return Some(FrameworkInfo {
            name: "PyTorch".to_string(),
            category: FrameworkCategory::Data,
            version: None,
        });
    }

    // TensorFlow
    if deps.iter().any(|d| d == "tensorflow" || d == "tf") {
        return Some(FrameworkInfo {
            name: "TensorFlow".to_string(),
            category: FrameworkCategory::Data,
            version: None,
        });
    }

    // Click (CLI)
    if deps.iter().any(|d| d == "click") {
        return Some(FrameworkInfo {
            name: "Click".to_string(),
            category: FrameworkCategory::Cli,
            version: None,
        });
    }

    // Typer (CLI)
    if deps.iter().any(|d| d == "typer") {
        return Some(FrameworkInfo {
            name: "Typer".to_string(),
            category: FrameworkCategory::Cli,
            version: None,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Go frameworks
// ---------------------------------------------------------------------------

fn detect_go_framework(_root: &Path, deps: &[String]) -> Option<FrameworkInfo> {
    // Gin
    if deps.iter().any(|d| d.contains("gin-gonic/gin")) {
        return Some(FrameworkInfo {
            name: "Gin".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Echo
    if deps.iter().any(|d| d.contains("labstack/echo")) {
        return Some(FrameworkInfo {
            name: "Echo".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Fiber
    if deps.iter().any(|d| d.contains("gofiber/fiber")) {
        return Some(FrameworkInfo {
            name: "Fiber".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Chi
    if deps.iter().any(|d| d.contains("go-chi/chi")) {
        return Some(FrameworkInfo {
            name: "Chi".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // gRPC
    if deps.iter().any(|d| d.contains("google.golang.org/grpc")) {
        return Some(FrameworkInfo {
            name: "gRPC-Go".to_string(),
            category: FrameworkCategory::Web,
            version: None,
        });
    }

    // Cobra (CLI)
    if deps.iter().any(|d| d.contains("spf13/cobra")) {
        return Some(FrameworkInfo {
            name: "Cobra".to_string(),
            category: FrameworkCategory::Cli,
            version: None,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Java frameworks
// ---------------------------------------------------------------------------

fn detect_java_framework(root: &Path, _deps: &[String]) -> Option<FrameworkInfo> {
    // Spring Boot — check for application.properties or application.yml
    if root.join("src/main/resources/application.properties").exists()
        || root.join("src/main/resources/application.yml").exists()
        || root.join("src/main/resources/application.yaml").exists()
    {
        return Some(FrameworkInfo {
            name: "Spring Boot".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Check pom.xml for spring-boot
    if let Ok(content) = fs::read_to_string(root.join("pom.xml")) {
        if content.contains("spring-boot") {
            return Some(FrameworkInfo {
                name: "Spring Boot".to_string(),
                category: FrameworkCategory::FullStack,
                version: None,
            });
        }
        if content.contains("quarkus") {
            return Some(FrameworkInfo {
                name: "Quarkus".to_string(),
                category: FrameworkCategory::Web,
                version: None,
            });
        }
        if content.contains("micronaut") {
            return Some(FrameworkInfo {
                name: "Micronaut".to_string(),
                category: FrameworkCategory::Web,
                version: None,
            });
        }
    }

    // Check build.gradle for spring-boot
    for gradle_file in &["build.gradle", "build.gradle.kts"] {
        if let Ok(content) = fs::read_to_string(root.join(gradle_file)) {
            if content.contains("spring-boot") || content.contains("org.springframework.boot") {
                return Some(FrameworkInfo {
                    name: "Spring Boot".to_string(),
                    category: FrameworkCategory::FullStack,
                    version: None,
                });
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Ruby frameworks
// ---------------------------------------------------------------------------

fn detect_ruby_framework(root: &Path, _deps: &[String]) -> Option<FrameworkInfo> {
    // Rails
    if root.join("config/routes.rb").exists() || root.join("config/application.rb").exists() {
        return Some(FrameworkInfo {
            name: "Ruby on Rails".to_string(),
            category: FrameworkCategory::FullStack,
            version: None,
        });
    }

    // Sinatra — check Gemfile
    if let Ok(content) = fs::read_to_string(root.join("Gemfile")) {
        if content.contains("sinatra") {
            return Some(FrameworkInfo {
                name: "Sinatra".to_string(),
                category: FrameworkCategory::Web,
                version: None,
            });
        }
        if content.contains("hanami") {
            return Some(FrameworkInfo {
                name: "Hanami".to_string(),
                category: FrameworkCategory::Web,
                version: None,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// C# frameworks
// ---------------------------------------------------------------------------

fn detect_csharp_framework(root: &Path, _deps: &[String]) -> Option<FrameworkInfo> {
    // Look for .csproj files
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("csproj") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if content.contains("Microsoft.AspNetCore") || content.contains("Microsoft.NET.Sdk.Web") {
                        return Some(FrameworkInfo {
                            name: "ASP.NET Core".to_string(),
                            category: FrameworkCategory::Web,
                            version: None,
                        });
                    }
                    if content.contains("Xamarin") {
                        return Some(FrameworkInfo {
                            name: "Xamarin".to_string(),
                            category: FrameworkCategory::Other,
                            version: None,
                        });
                    }
                    if content.contains("MAUI") {
                        return Some(FrameworkInfo {
                            name: ".NET MAUI".to_string(),
                            category: FrameworkCategory::Other,
                            version: None,
                        });
                    }
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to extract a dependency version from package.json.
fn find_dep_version(root: &Path, dep_name: &str) -> Option<String> {
    let pkg_path = root.join("package.json");
    let content = fs::read_to_string(pkg_path).ok()?;
    let pkg: serde_json::Value = serde_json::from_str(&content).ok()?;

    for section in &["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(version) = pkg
            .get(*section)
            .and_then(|s| s.get(dep_name))
            .and_then(|v| v.as_str())
        {
            return Some(version.to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn toolchain_with_deps(deps: Vec<String>) -> ToolchainInfo {
        ToolchainInfo {
            dependencies: deps,
            ..Default::default()
        }
    }

    fn setup_project(files: &[(&str, &str)]) -> tempfile::TempDir {
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
    fn test_detect_axum() {
        let tc = toolchain_with_deps(vec!["axum".to_string(), "tokio".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Rust, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Axum");
    }

    #[test]
    fn test_detect_actix_web() {
        let tc = toolchain_with_deps(vec!["actix-web".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Rust, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Actix Web");
    }

    #[test]
    fn test_detect_nextjs() {
        let dir = setup_project(&[(
            "package.json",
            r#"{"dependencies": {"next": "^14.0.0", "react": "^18.0.0"}}"#,
        )]);
        let tc = toolchain_with_deps(vec![
            "next".to_string(),
            "react".to_string(),
        ]);
        let fw = detect_framework(dir.path(), DetectedLanguage::TypeScript, &tc);
        assert!(fw.is_some());
        let fw = fw.unwrap();
        assert_eq!(fw.name, "Next.js");
        assert_eq!(fw.category, FrameworkCategory::FullStack);
        // Should extract version from package.json
        assert_eq!(fw.version.as_deref(), Some("^14.0.0"));
    }

    #[test]
    fn test_detect_react_without_next() {
        let tc = toolchain_with_deps(vec!["react".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::JavaScript, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "React");
    }

    #[test]
    fn test_detect_fastapi() {
        let tc = toolchain_with_deps(vec!["fastapi".to_string(), "uvicorn".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Python, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "FastAPI");
    }

    #[test]
    fn test_detect_django() {
        let tc = toolchain_with_deps(vec!["Django".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Python, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Django");
    }

    #[test]
    fn test_detect_gin() {
        let tc = toolchain_with_deps(vec!["github.com/gin-gonic/gin".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Go, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Gin");
    }

    #[test]
    fn test_detect_spring_boot_from_properties() {
        let dir = setup_project(&[(
            "src/main/resources/application.properties",
            "server.port=8080",
        )]);
        let tc = ToolchainInfo::default();
        let fw = detect_framework(dir.path(), DetectedLanguage::Java, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Spring Boot");
    }

    #[test]
    fn test_detect_rails() {
        let dir = setup_project(&[
            ("config/routes.rb", "Rails.application.routes.draw {}"),
            ("config/application.rb", ""),
        ]);
        let tc = ToolchainInfo::default();
        let fw = detect_framework(dir.path(), DetectedLanguage::Ruby, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Ruby on Rails");
    }

    #[test]
    fn test_detect_aspnet_core() {
        let dir = setup_project(&[(
            "MyApp.csproj",
            r#"<Project Sdk="Microsoft.NET.Sdk.Web"></Project>"#,
        )]);
        let tc = ToolchainInfo::default();
        let fw = detect_framework(dir.path(), DetectedLanguage::CSharp, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "ASP.NET Core");
    }

    #[test]
    fn test_no_framework_detected() {
        let tc = ToolchainInfo::default();
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Rust, &tc);
        assert!(fw.is_none());
    }

    #[test]
    fn test_unknown_language_no_framework() {
        let tc = toolchain_with_deps(vec!["react".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Unknown, &tc);
        assert!(fw.is_none());
    }

    #[test]
    fn test_nextjs_preferred_over_react() {
        // When both next and react are present, Next.js should win
        let tc = toolchain_with_deps(vec![
            "react".to_string(),
            "next".to_string(),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::TypeScript, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "Next.js");
    }

    #[test]
    fn test_nestjs_detection() {
        let tc = toolchain_with_deps(vec!["@nestjs/core".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::TypeScript, &tc);
        assert!(fw.is_some());
        assert_eq!(fw.unwrap().name, "NestJS");
    }

    #[test]
    fn test_clap_detection() {
        let tc = toolchain_with_deps(vec!["clap".to_string()]);
        let dir = tempfile::tempdir().unwrap();
        let fw = detect_framework(dir.path(), DetectedLanguage::Rust, &tc);
        assert!(fw.is_some());
        let fw = fw.unwrap();
        assert_eq!(fw.name, "Clap");
        assert_eq!(fw.category, FrameworkCategory::Cli);
    }
}
