//! Context engine — main coordinator (§9).
//!
//! The `ContextEngine` orchestrates context assembly for every LLM call:
//!
//! 1. **Gather** candidate context items from multiple sources.
//! 2. **Score** each candidate via its feature vector.
//! 3. **Pack** the highest-scoring candidates into a token budget.
//! 4. **Render** the packed context into a string for the call assembler.
//!
//! Session-aware mode (§9.5): when continuing an existing LLM session, the
//! engine tracks which content hashes have already been sent and only includes
//! *new* candidates in the incremental context package.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::arena::{fnv_hash, Arena};
use crate::context::candidates::{
    estimate_tokens, ContextCandidate, ContextFeatures, ContextSource,
};
use crate::context::import_graph::{ImportGraph, Language};
use crate::context::packer::{pack_context, pack_context_incremental, PackedContext};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during context assembly.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("node {0} not found in arena")]
    NodeNotFound(u32),

    #[error("no spec found for node {0}")]
    NoSpec(u32),

    #[error("I/O error reading file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

// ---------------------------------------------------------------------------
// ContextEngine
// ---------------------------------------------------------------------------

/// Main context engine that assembles context packages for LLM calls.
///
/// Holds a read-only view of the project file system (in-memory), the import
/// graph, and the set of content hashes already sent in each session.
pub struct ContextEngine {
    /// Import graph for the project.
    import_graph: ImportGraph,

    /// In-memory file cache: file path → content.
    /// Populated by the runtime when files are read or written.
    file_cache: HashMap<String, String>,

    /// Per-session tracking of content hashes that have already been sent.
    /// session_id → set of content_hashes.
    sent_hashes: HashMap<u32, HashSet<u64>>,

    /// Default token budget for a full context package.
    default_budget: u32,
}

impl ContextEngine {
    /// Create a new context engine with the given default token budget.
    pub fn new(default_budget: u32) -> Self {
        Self {
            import_graph: ImportGraph::new(),
            file_cache: HashMap::new(),
            sent_hashes: HashMap::new(),
            default_budget,
        }
    }

    /// Get a reference to the import graph.
    pub fn import_graph(&self) -> &ImportGraph {
        &self.import_graph
    }

    /// Get a mutable reference to the import graph.
    pub fn import_graph_mut(&mut self) -> &mut ImportGraph {
        &mut self.import_graph
    }

    /// Register a file in the in-memory cache. Returns its FNV hash.
    pub fn register_file(&mut self, path: &str, content: String) -> u64 {
        let hash = fnv_hash(path);
        self.file_cache.insert(path.to_string(), content);
        hash
    }

    /// Remove a file from the cache.
    pub fn remove_file(&mut self, path: &str) {
        self.file_cache.remove(path);
    }

    /// Get the content of a cached file.
    pub fn get_file(&self, path: &str) -> Option<&str> {
        self.file_cache.get(path).map(|s| s.as_str())
    }

    /// Number of files in the cache.
    pub fn cached_file_count(&self) -> usize {
        self.file_cache.len()
    }

    /// Rebuild the import graph from all cached files.
    pub fn rebuild_import_graph(&mut self) {
        let files: Vec<(String, String)> = self
            .file_cache
            .iter()
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect();

        self.import_graph.rebuild_from_files(&files, fnv_hash);
    }

    /// Update the import graph for a single file that was just written/modified.
    ///
    /// Borrows the file cache to build the known-files list for import
    /// resolution. Only file paths are cloned (needed for the
    /// `resolve_import_path` lifetime); content strings are NOT cloned since
    /// `resolve_import_path` only inspects paths. The target file's content
    /// is cloned once to break the shared borrow on `self.file_cache`.
    pub fn update_import_graph_for_file(&mut self, path: &str) {
        if let Some(content) = self.file_cache.get(path).cloned() {
            // resolve_import_path only uses the path component of each pair;
            // the content field is never read. Pass empty strings to avoid
            // cloning every file's content.
            let files: Vec<(String, String)> = self
                .file_cache
                .keys()
                .map(|p| (p.clone(), String::new()))
                .collect();

            self.import_graph.update_file(path, &content, &files, fnv_hash);
        }
    }

    // -----------------------------------------------------------------------
    // Session-aware tracking
    // -----------------------------------------------------------------------

    /// Record that a set of content hashes have been sent in a session.
    pub fn record_sent(&mut self, session_id: u32, hashes: impl IntoIterator<Item = u64>) {
        let entry = self.sent_hashes.entry(session_id).or_default();
        for h in hashes {
            entry.insert(h);
        }
    }

    /// Get the set of content hashes already sent in a session.
    pub fn sent_for_session(&self, session_id: u32) -> Option<&HashSet<u64>> {
        self.sent_hashes.get(&session_id)
    }

    /// Clear tracking for a closed session.
    pub fn clear_session(&mut self, session_id: u32) {
        self.sent_hashes.remove(&session_id);
    }

    // -----------------------------------------------------------------------
    // Main API: assemble context for a node
    // -----------------------------------------------------------------------

    /// Assemble a full context package for a graph node.
    ///
    /// This is the entry point called by the runtime before each LLM call.
    ///
    /// `node_idx`: the arena node to build context for.
    /// `arena`: the execution graph.
    /// `budget`: optional override for the token budget.
    ///
    /// Returns `(packed, fingerprint)` where fingerprint is an XXH3 hash of
    /// the sorted content hashes for cache invalidation.
    pub fn assemble_context(
        &self,
        node_idx: u32,
        arena: &Arena,
        budget: Option<u32>,
    ) -> Result<(PackedContext, u64), ContextError> {
        let budget = budget.unwrap_or(self.default_budget);

        let candidates = self.gather_candidates(node_idx, arena)?;
        let packed = pack_context(candidates, budget);
        let fingerprint = compute_fingerprint(&packed);

        Ok((packed, fingerprint))
    }

    /// Assemble an incremental context package for a session continuation.
    ///
    /// Only includes candidates not yet sent in `session_id`.
    pub fn assemble_incremental(
        &self,
        node_idx: u32,
        arena: &Arena,
        session_id: u32,
        budget: Option<u32>,
    ) -> Result<(PackedContext, u64), ContextError> {
        let budget = budget.unwrap_or(self.default_budget);

        let candidates = self.gather_candidates(node_idx, arena)?;
        let sent = self.sent_hashes.get(&session_id);
        let empty_set = HashSet::new();
        let sent_ref = sent.unwrap_or(&empty_set);

        let packed = pack_context_incremental(candidates, budget, sent_ref);
        let fingerprint = compute_fingerprint(&packed);

        Ok((packed, fingerprint))
    }

    // -----------------------------------------------------------------------
    // Candidate gathering (§9.3)
    // -----------------------------------------------------------------------

    /// Gather all context candidates for a node from multiple sources.
    ///
    /// Sources:
    /// 1. Files directly referenced in the node spec.
    /// 2. Files in the same directory as the target files.
    /// 3. Files imported by / importing target files (via import graph).
    /// 4. Type definition files.
    /// 5. Related test files.
    /// 6. Previous implementations from dependency nodes.
    /// 7. The node spec itself.
    ///
    /// Deduplicates by `content_hash`.
    fn gather_candidates(
        &self,
        node_idx: u32,
        arena: &Arena,
    ) -> Result<Vec<ContextCandidate>, ContextError> {
        let node = arena.get(node_idx)
            .ok_or(ContextError::NodeNotFound(node_idx))?;

        let spec = arena.get_spec(node_idx).unwrap_or("");

        // Resolve target file paths
        let target_paths: Vec<String> = node
            .impl_files
            .iter()
            .filter_map(|&hash| arena.resolve_file_path(hash))
            .map(|s| s.to_string())
            .collect();

        let target_hashes: Vec<u64> = node.impl_files.iter().copied().collect();

        // Collect target directories for same-directory matching
        let target_dirs: HashSet<String> = target_paths
            .iter()
            .filter_map(|p| {
                Path::new(p)
                    .parent()
                    .and_then(|d| d.to_str())
                    .map(|s| s.to_string())
            })
            .collect();

        let mut seen_hashes: HashSet<u64> = HashSet::new();
        let mut candidates: Vec<ContextCandidate> = Vec::new();

        // ---- Source 1: Node spec ----
        if !spec.is_empty() {
            let c = ContextCandidate::new(
                ContextSource::NodeSpec,
                format!("spec:{}", node.title),
                spec,
                ContextFeatures {
                    mentioned_in_spec: true,
                    ..Default::default()
                },
            );
            seen_hashes.insert(c.content_hash);
            candidates.push(c);
        }

        // ---- Source 2: Files referenced in spec ----
        for (path, content) in &self.file_cache {
            // Check if the filename (without directory) appears in the spec
            let filename = Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if !filename.is_empty() && spec.contains(filename) {
                let features = self.compute_features(
                    path,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    content,
                );

                let c = ContextCandidate::new(
                    ContextSource::File,
                    path.as_str(),
                    content.as_str(),
                    ContextFeatures {
                        mentioned_in_spec: true,
                        ..features
                    },
                );

                if seen_hashes.insert(c.content_hash) {
                    candidates.push(c);
                }
            }
        }

        // ---- Source 3: Same directory files ----
        for (path, content) in &self.file_cache {
            if target_paths.contains(path) {
                continue; // skip the target files themselves
            }

            let file_dir = Path::new(path)
                .parent()
                .and_then(|d| d.to_str())
                .unwrap_or("");

            if target_dirs.contains(file_dir) {
                let features = self.compute_features(
                    path,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    content,
                );

                let c = ContextCandidate::new(
                    ContextSource::File,
                    path.as_str(),
                    content.as_str(),
                    features,
                );

                if seen_hashes.insert(c.content_hash) {
                    candidates.push(c);
                }
            }
        }

        // ---- Source 4: Import graph neighbors ----
        for &target_hash in &target_hashes {
            let neighbors = self.import_graph.neighbors(target_hash);
            for neighbor_hash in neighbors {
                // Find the path for this hash
                if let Some(path) = arena.resolve_file_path(neighbor_hash) {
                    if let Some(content) = self.file_cache.get(path) {
                        let features = self.compute_features(
                            path,
                            &target_paths,
                            &target_hashes,
                            &target_dirs,
                            spec,
                            content,
                        );

                        let c = ContextCandidate::new(
                            ContextSource::File,
                            path,
                            content.as_str(),
                            features,
                        );

                        if seen_hashes.insert(c.content_hash) {
                            candidates.push(c);
                        }
                    }
                }
            }
        }

        // Also check import graph for files registered only in the cache
        // (not yet registered in arena file_paths)
        for &target_hash in &target_hashes {
            // Files imported by the target
            for &imported_hash in self.import_graph.imports_of(target_hash) {
                self.try_add_from_hash(
                    imported_hash,
                    arena,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    &mut seen_hashes,
                    &mut candidates,
                );
            }

            // Files that import the target
            for &importer_hash in self.import_graph.importers_of(target_hash) {
                self.try_add_from_hash(
                    importer_hash,
                    arena,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    &mut seen_hashes,
                    &mut candidates,
                );
            }
        }

        // ---- Source 5: Type definition files ----
        for (path, content) in &self.file_cache {
            if is_type_definition_file(path) {
                let features = self.compute_features(
                    path,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    content,
                );

                let c = ContextCandidate::new(
                    ContextSource::File,
                    path.as_str(),
                    content.as_str(),
                    ContextFeatures {
                        type_definition: true,
                        ..features
                    },
                );

                if seen_hashes.insert(c.content_hash) {
                    candidates.push(c);
                }
            }
        }

        // ---- Source 6: Related test files ----
        for target_path in &target_paths {
            let test_candidates = find_related_test_files(target_path, &self.file_cache);
            for (test_path, test_content) in test_candidates {
                let features = self.compute_features(
                    &test_path,
                    &target_paths,
                    &target_hashes,
                    &target_dirs,
                    spec,
                    &test_content,
                );

                let c = ContextCandidate::new(
                    ContextSource::TestFile,
                    test_path,
                    test_content,
                    ContextFeatures {
                        related_test: true,
                        ..features
                    },
                );

                if seen_hashes.insert(c.content_hash) {
                    candidates.push(c);
                }
            }
        }

        // ---- Source 7: Dependency node outputs ----
        for &dep_idx in &node.dependencies {
            let dep_node = match arena.get(dep_idx) {
                Some(n) => n,
                None => continue,
            };
            if dep_node.status == crate::arena::node::NodeStatus::Verified {
                // Include the dependency's implementation files
                for &file_hash in &dep_node.impl_files {
                    self.try_add_from_hash(
                        file_hash,
                        arena,
                        &target_paths,
                        &target_hashes,
                        &target_dirs,
                        spec,
                        &mut seen_hashes,
                        &mut candidates,
                    );
                }

                // Include the dependency's spec if available
                if let Some(dep_spec) = arena.get_spec(dep_idx) {
                    let c = ContextCandidate::new(
                        ContextSource::NodeSpec,
                        format!("dep-spec:{}", dep_node.title),
                        dep_spec,
                        ContextFeatures {
                            dependency_of_target: true,
                            ..Default::default()
                        },
                    );

                    if seen_hashes.insert(c.content_hash) {
                        candidates.push(c);
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Helper to add a candidate from a file hash, if the file is in the cache.
    #[allow(clippy::too_many_arguments)]
    fn try_add_from_hash(
        &self,
        file_hash: u64,
        arena: &Arena,
        target_paths: &[String],
        target_hashes: &[u64],
        target_dirs: &HashSet<String>,
        spec: &str,
        seen_hashes: &mut HashSet<u64>,
        candidates: &mut Vec<ContextCandidate>,
    ) {
        if let Some(path) = arena.resolve_file_path(file_hash) {
            if let Some(content) = self.file_cache.get(path) {
                let features = self.compute_features(
                    path,
                    target_paths,
                    target_hashes,
                    target_dirs,
                    spec,
                    content,
                );

                let c = ContextCandidate::new(
                    ContextSource::File,
                    path,
                    content.as_str(),
                    features,
                );

                if seen_hashes.insert(c.content_hash) {
                    candidates.push(c);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Feature computation
    // -----------------------------------------------------------------------

    /// Compute the feature vector for a candidate file.
    fn compute_features(
        &self,
        candidate_path: &str,
        target_paths: &[String],
        target_hashes: &[u64],
        target_dirs: &HashSet<String>,
        spec: &str,
        content: &str,
    ) -> ContextFeatures {
        let candidate_hash = fnv_hash(candidate_path);
        let candidate_dir = Path::new(candidate_path)
            .parent()
            .and_then(|d| d.to_str())
            .unwrap_or("");
        let candidate_filename = Path::new(candidate_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // same_directory
        let same_directory = target_dirs.contains(candidate_dir);

        // imported_by_target: any target file imports this candidate
        let imported_by_target = target_hashes
            .iter()
            .any(|&th| self.import_graph.imports_of(th).contains(&candidate_hash));

        // imports_target: this candidate imports any target file
        let imports_target = {
            let candidate_imports = self.import_graph.imports_of(candidate_hash);
            target_hashes.iter().any(|th| candidate_imports.contains(th))
        };

        // same_language
        let same_language = target_paths
            .first()
            .map(|tp| Language::same_family(tp, candidate_path))
            .unwrap_or(false);

        // mentioned_in_spec
        let mentioned_in_spec = !candidate_filename.is_empty() && spec.contains(candidate_filename);

        // related_test
        let related_test = is_test_file(candidate_path);

        // similar_name: require minimum stem length to avoid spurious substring
        // matches on short names like "a", "io", "db", etc.
        let similar_name = target_paths.iter().any(|tp| {
            let target_stem = Path::new(tp)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let candidate_stem = Path::new(candidate_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if candidate_stem.len() < 3 || target_stem.len() < 3 {
                return false; // too short for substring matching
            }
            target_stem.contains(candidate_stem) || candidate_stem.contains(target_stem)
        });

        // type_definition
        let type_definition = is_type_definition_file(candidate_path)
            || content_has_type_definitions(content, candidate_path);

        // dependency_of_target: checked at the call site (dependency node outputs)
        // recently_modified: not tracked in this module (set by caller)
        // in_current_session: not tracked in this module (set by caller)

        let estimated = estimate_tokens(content);

        ContextFeatures {
            same_directory,
            imported_by_target,
            imports_target,
            same_language,
            recently_modified: false,
            mentioned_in_spec,
            related_test,
            dependency_of_target: false,
            similar_name,
            in_current_session: false,
            file_size_bucket: crate::context::candidates::token_bucket(estimated),
            type_definition,
        }
    }
}

impl Default for ContextEngine {
    fn default() -> Self {
        // Default budget: ~16k tokens (reasonable for most models)
        Self::new(16_000)
    }
}

// ---------------------------------------------------------------------------
// Fingerprinting
// ---------------------------------------------------------------------------

/// Compute an XXH3 fingerprint of a packed context for cache invalidation.
///
/// The fingerprint is based on the sorted content hashes of all included items.
fn compute_fingerprint(packed: &PackedContext) -> u64 {
    let mut hashes: Vec<u64> = packed.items.iter().map(|item| item.content_hash).collect();
    hashes.sort_unstable();

    // Hash the sorted hash list
    let bytes: Vec<u8> = hashes.iter().flat_map(|h| h.to_le_bytes()).collect();
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

// ---------------------------------------------------------------------------
// File classification helpers
// ---------------------------------------------------------------------------

/// Check if a file is a type definition file based on its name/extension.
fn is_type_definition_file(path: &str) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    // TypeScript declaration files
    filename.ends_with(".d.ts")
        // Common naming patterns (matched against filename only)
        || filename.starts_with("types.")
        || filename.starts_with("type.")
        || filename == "types" // extensionless
        || filename.starts_with("interfaces.")
        || filename.starts_with("models.")
        // C/C++ headers (often contain type definitions)
        || filename.ends_with(".h")
        || filename.ends_with(".hpp")
        || filename.ends_with(".hxx")
        // Python type stubs
        || filename.ends_with(".pyi")
}

/// Heuristic check of file content for type definition patterns.
///
/// Uses line-start matching to avoid false positives from comments, strings,
/// or variable names that happen to contain keywords like "type" or "class".
fn content_has_type_definitions(content: &str, path: &str) -> bool {
    let lang = Language::from_path(path);
    match lang {
        Language::TypeScript | Language::JavaScript => {
            content.lines().any(|line| {
                let t = line.trim_start();
                t.starts_with("interface ")
                    || t.starts_with("export interface ")
                    || t.starts_with("type ")
                    || t.starts_with("export type ")
                    || t.starts_with("enum ")
                    || t.starts_with("export enum ")
            })
        }
        Language::Rust => {
            content.lines().any(|line| {
                let t = line.trim_start();
                t.starts_with("pub struct ")
                    || t.starts_with("pub enum ")
                    || t.starts_with("pub trait ")
                    || t.starts_with("struct ")
                    || t.starts_with("enum ")
                    || t.starts_with("trait ")
            })
        }
        Language::Python => {
            content.lines().any(|line| {
                let t = line.trim_start();
                // Top-level or indented class definitions
                t.starts_with("class ") && t.contains(':')
            })
        }
        Language::Go => {
            content.lines().any(|line| {
                let t = line.trim_start();
                t.starts_with("type ") && (t.contains("struct {") || t.contains("interface {"))
            })
        }
        _ => false,
    }
}

/// Check if a file path looks like a test file.
///
/// Uses word-boundary-aware matching on the filename to avoid false positives
/// (e.g. "contest.rs" or "spectral.ts" should not match).
fn is_test_file(path: &str) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    // Check common test file naming patterns
    filename.starts_with("test_") || filename.starts_with("test.")
        || filename.ends_with("_test.") || filename.contains("_test.")
        || filename.ends_with(".test.") || filename.contains(".test.")
        || filename.starts_with("spec_") || filename.starts_with("spec.")
        || filename.ends_with("_spec.") || filename.contains("_spec.")
        || filename.ends_with(".spec.") || filename.contains(".spec.")
        // Also check directory components for test directories
        || path.contains("/test/") || path.contains("/tests/")
        || path.contains("/spec/") || path.contains("/specs/")
        || path.contains("/__tests__/") || path.contains("/__test__/")
}

/// Find test files related to a given implementation file.
fn find_related_test_files(
    impl_path: &str,
    file_cache: &HashMap<String, String>,
) -> Vec<(String, String)> {
    let stem = Path::new(impl_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    if stem.is_empty() {
        return Vec::new();
    }

    let impl_dir = Path::new(impl_path)
        .parent()
        .and_then(|d| d.to_str())
        .unwrap_or("");

    let lang = Language::from_path(impl_path);

    // Build candidate test file names
    let mut test_patterns = Vec::new();

    match lang {
        Language::JavaScript | Language::TypeScript => {
            let ext = Path::new(impl_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("ts");
            test_patterns.push(format!("{stem}.test.{ext}"));
            test_patterns.push(format!("{stem}.spec.{ext}"));
            test_patterns.push(format!("{stem}_test.{ext}"));
        }
        Language::Rust => {
            // In Rust, tests are often in the same file or in a tests/ directory
            test_patterns.push(format!("{stem}_test.rs"));
            test_patterns.push("tests.rs".to_string());
        }
        Language::Python => {
            test_patterns.push(format!("test_{stem}.py"));
            test_patterns.push(format!("{stem}_test.py"));
        }
        Language::Go => {
            test_patterns.push(format!("{stem}_test.go"));
        }
        _ => {}
    }

    let mut results = Vec::new();

    for (path, content) in file_cache {
        let file_name = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Check for test pattern match
        let matches_pattern = test_patterns.iter().any(|pat| file_name == pat);

        // Also check if the test file is in the same directory or a __tests__/tests/ subdirectory
        let file_dir = Path::new(path)
            .parent()
            .and_then(|d| d.to_str())
            .unwrap_or("");

        let in_related_dir = file_dir == impl_dir
            || file_dir == format!("{impl_dir}/__tests__")
            || file_dir == format!("{impl_dir}/tests")
            || file_dir == format!("{impl_dir}/test");

        if matches_pattern && in_related_dir {
            results.push((path.clone(), content.clone()));
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::{Node, Priority};

    fn setup_arena_and_engine() -> (Arena, ContextEngine) {
        let mut arena = Arena::new();
        let mut engine = ContextEngine::new(8000);

        // Register some files
        let auth_hash = arena.register_file_path("src/auth.ts");
        let _utils_hash = arena.register_file_path("src/utils.ts");
        let _types_hash = arena.register_file_path("src/types.ts");
        let _auth_test_hash = arena.register_file_path("src/auth.test.ts");

        // Populate cache
        engine.register_file(
            "src/auth.ts",
            "import { User } from './types';\nexport function login(user: User) { return true; }"
                .to_string(),
        );
        engine.register_file(
            "src/utils.ts",
            "export function hash(s: string): string { return s; }".to_string(),
        );
        engine.register_file(
            "src/types.ts",
            "export interface User { id: number; name: string; }\nexport type Role = 'admin' | 'user';"
                .to_string(),
        );
        engine.register_file(
            "src/auth.test.ts",
            "import { login } from './auth';\ndescribe('login', () => { it('works', () => {}); });"
                .to_string(),
        );

        // Create a node for auth implementation
        let mut node = Node::new("Auth login".to_string(), Priority::Standard);
        node.impl_files.push(auth_hash);
        let node_idx = arena.alloc(node);
        arena.set_spec(node_idx, "Implement the login function in auth.ts. Use the User type from types.ts.".to_string());

        // Build import graph
        engine.rebuild_import_graph();

        (arena, engine)
    }

    #[test]
    fn test_engine_creation() {
        let engine = ContextEngine::new(10000);
        assert_eq!(engine.cached_file_count(), 0);
    }

    #[test]
    fn test_register_and_get_file() {
        let mut engine = ContextEngine::new(10000);
        engine.register_file("src/foo.rs", "fn foo() {}".to_string());

        assert_eq!(engine.get_file("src/foo.rs"), Some("fn foo() {}"));
        assert_eq!(engine.cached_file_count(), 1);
    }

    #[test]
    fn test_remove_file() {
        let mut engine = ContextEngine::new(10000);
        engine.register_file("src/foo.rs", "fn foo() {}".to_string());
        engine.remove_file("src/foo.rs");

        assert!(engine.get_file("src/foo.rs").is_none());
        assert_eq!(engine.cached_file_count(), 0);
    }

    #[test]
    fn test_assemble_context_basic() {
        let (arena, engine) = setup_arena_and_engine();

        // Node index 1 (first alloc after root)
        let result = engine.assemble_context(1, &arena, None);
        assert!(result.is_ok());

        let (packed, fingerprint) = result.unwrap();
        assert!(!packed.is_empty());
        assert!(fingerprint != 0);

        // Should include the spec
        let labels: Vec<&str> = packed.items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.contains("spec:")));
    }

    #[test]
    fn test_assemble_context_includes_mentioned_files() {
        let (arena, engine) = setup_arena_and_engine();

        let (packed, _) = engine.assemble_context(1, &arena, None).unwrap();

        let labels: Vec<&str> = packed.items.iter().map(|i| i.label.as_str()).collect();
        // types.ts is mentioned in the spec
        assert!(
            labels.contains(&"src/types.ts"),
            "Should include types.ts (mentioned in spec). Labels: {:?}",
            labels
        );
    }

    #[test]
    fn test_assemble_context_includes_same_directory() {
        let (arena, engine) = setup_arena_and_engine();

        let (packed, _) = engine.assemble_context(1, &arena, None).unwrap();

        let labels: Vec<&str> = packed.items.iter().map(|i| i.label.as_str()).collect();
        // utils.ts is in the same directory
        assert!(
            labels.contains(&"src/utils.ts"),
            "Should include utils.ts (same directory). Labels: {:?}",
            labels
        );
    }

    #[test]
    fn test_assemble_context_includes_test_file() {
        let (arena, engine) = setup_arena_and_engine();

        let (packed, _) = engine.assemble_context(1, &arena, None).unwrap();

        let labels: Vec<&str> = packed.items.iter().map(|i| i.label.as_str()).collect();
        assert!(
            labels.contains(&"src/auth.test.ts"),
            "Should include auth.test.ts (related test). Labels: {:?}",
            labels
        );
    }

    #[test]
    fn test_assemble_context_deduplicates() {
        let (arena, engine) = setup_arena_and_engine();

        let (packed, _) = engine.assemble_context(1, &arena, None).unwrap();

        // Check no duplicate content hashes
        let hashes: Vec<u64> = packed.items.iter().map(|i| i.content_hash).collect();
        let unique: HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(hashes.len(), unique.len(), "Should have no duplicate content");
    }

    #[test]
    fn test_assemble_context_respects_budget() {
        let (arena, engine) = setup_arena_and_engine();

        // Tiny budget
        let (packed, _) = engine.assemble_context(1, &arena, Some(50)).unwrap();
        assert!(packed.total_tokens <= 50);
    }

    #[test]
    fn test_incremental_assembly() {
        let (arena, mut engine) = setup_arena_and_engine();

        // First assembly
        let (packed1, _) = engine.assemble_context(1, &arena, None).unwrap();
        let sent: Vec<u64> = packed1.items.iter().map(|i| i.content_hash).collect();

        // Record as sent in session 1
        engine.record_sent(1, sent.clone());

        // Incremental assembly should return no items (all already sent)
        let (packed2, _) = engine.assemble_incremental(1, &arena, 1, None).unwrap();
        assert!(
            packed2.is_empty(),
            "Incremental should be empty when all candidates already sent. Got {} items.",
            packed2.len()
        );
    }

    #[test]
    fn test_clear_session() {
        let mut engine = ContextEngine::new(10000);
        engine.record_sent(1, vec![100, 200, 300]);
        assert!(engine.sent_for_session(1).is_some());

        engine.clear_session(1);
        assert!(engine.sent_for_session(1).is_none());
    }

    #[test]
    fn test_deallocated_node_returns_error() {
        let (mut arena, engine) = setup_arena_and_engine();
        arena.mark_complete(1);
        arena.dealloc(1);

        let result = engine.assemble_context(1, &arena, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fingerprint_changes_with_content() {
        let (arena, engine) = setup_arena_and_engine();

        let (_, fp1) = engine.assemble_context(1, &arena, None).unwrap();
        let (_, fp2) = engine.assemble_context(1, &arena, None).unwrap();

        // Same inputs should produce same fingerprint
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_is_type_definition_file() {
        assert!(is_type_definition_file("src/types.ts"));
        assert!(is_type_definition_file("src/models.py"));
        assert!(is_type_definition_file("include/header.h"));
        assert!(is_type_definition_file("src/User.d.ts"));
        assert!(!is_type_definition_file("src/main.ts"));
        assert!(!is_type_definition_file("src/utils.rs"));
    }

    #[test]
    fn test_is_test_file() {
        assert!(is_test_file("src/auth.test.ts"));
        assert!(is_test_file("src/auth.spec.js"));
        assert!(is_test_file("src/auth_test.go"));
        assert!(is_test_file("tests/test_auth.py"));
        assert!(!is_test_file("src/auth.ts"));
        assert!(!is_test_file("src/utils.rs"));
    }

    #[test]
    fn test_content_has_type_definitions() {
        assert!(content_has_type_definitions(
            "export interface User { id: number; }",
            "types.ts"
        ));
        assert!(content_has_type_definitions(
            "pub struct Config { pub name: String }",
            "config.rs"
        ));
        assert!(content_has_type_definitions(
            "class User:\n    pass",
            "models.py"
        ));
        assert!(!content_has_type_definitions(
            "console.log('hello');",
            "main.ts"
        ));
    }

    #[test]
    fn test_find_related_test_files() {
        let mut cache = HashMap::new();
        cache.insert("src/auth.ts".to_string(), "export function login() {}".to_string());
        cache.insert(
            "src/auth.test.ts".to_string(),
            "describe('auth', () => {});".to_string(),
        );
        cache.insert("src/utils.ts".to_string(), "export function hash() {}".to_string());

        let tests = find_related_test_files("src/auth.ts", &cache);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].0, "src/auth.test.ts");
    }

    #[test]
    fn test_find_related_test_files_none() {
        let mut cache = HashMap::new();
        cache.insert("src/auth.ts".to_string(), "export function login() {}".to_string());
        cache.insert("src/utils.ts".to_string(), "export function hash() {}".to_string());

        let tests = find_related_test_files("src/auth.ts", &cache);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_compute_fingerprint_deterministic() {
        let c1 = ContextCandidate::new(
            ContextSource::File,
            "a.rs",
            "content a",
            ContextFeatures::default(),
        );
        let c2 = ContextCandidate::new(
            ContextSource::File,
            "b.rs",
            "content b",
            ContextFeatures::default(),
        );

        let packed1 = pack_context(vec![c1.clone(), c2.clone()], 10000);
        let packed2 = pack_context(vec![c2, c1], 10000);

        let fp1 = compute_fingerprint(&packed1);
        let fp2 = compute_fingerprint(&packed2);

        // Fingerprints should be equal because they sort hashes before hashing
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_dependency_node_context() {
        let mut arena = Arena::new();
        let mut engine = ContextEngine::new(8000);

        // Register files
        let dep_file_hash = arena.register_file_path("src/database.ts");
        let main_file_hash = arena.register_file_path("src/service.ts");

        engine.register_file(
            "src/database.ts",
            "export class Database { connect() {} }".to_string(),
        );
        engine.register_file("src/service.ts", "// service implementation".to_string());

        // Create dependency node (database)
        let dep_node = Node::new("Database layer".to_string(), Priority::Critical);
        let dep_idx = arena.alloc(dep_node);
        arena.get_mut(dep_idx).expect("just-allocated node").impl_files.push(dep_file_hash);
        arena.set_spec(dep_idx, "Implement the database connection layer.".to_string());
        arena.mark_complete(dep_idx);

        // Create main node (service) that depends on database
        let mut main_node = Node::new("Service layer".to_string(), Priority::Standard);
        main_node.impl_files.push(main_file_hash);
        let main_idx = arena.alloc(main_node);
        arena.set_spec(main_idx, "Implement the service using the database.".to_string());
        arena.add_dependency(main_idx, dep_idx).expect("no cycle in test setup");

        let (packed, _) = engine.assemble_context(main_idx, &arena, None).unwrap();

        let labels: Vec<&str> = packed.items.iter().map(|i| i.label.as_str()).collect();
        // Should include the dependency's spec
        assert!(
            labels.iter().any(|l| l.contains("dep-spec:")),
            "Should include dependency node spec. Labels: {:?}",
            labels
        );
    }
}
