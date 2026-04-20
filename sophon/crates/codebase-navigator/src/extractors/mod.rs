//! Language-aware symbol extractors.
//!
//! Each extractor knows how to pull top-level declarations (functions,
//! classes, structs, etc.) out of a single source file. The default
//! implementations are deterministic regex-based — good enough for
//! 90 % of clean code, honest about the edge cases where they fail:
//!
//! - **Multi-line signatures** (e.g. a Rust function with a generics
//!   clause that wraps over 3 lines) are recognised by the first line
//!   only; the `signature` field captures that first line.
//! - **Keywords inside comments or string literals** are not special-
//!   cased, so a comment like `// fn foo()` in Rust will produce a
//!   spurious symbol. The extractors strip `//` / `#` / `--` line
//!   comments before matching to mitigate the most common case.
//! - **Nested / inner declarations** are not walked — we only emit
//!   top-level symbols so that the digest stays small.
//!
//! When any of these limitations bite, the user should build with
//! `--features tree-sitter` and swap in the AST-backed extractor
//! defined in [`tree_sitter_backend`] (stub for now).

pub mod c_cpp;
pub mod generic;
pub mod go;
pub mod java;
pub mod javascript;
pub mod kotlin;
pub mod php;
pub mod plugin;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod swift;

// Legacy single-file stub kept for backwards compatibility with the
// pre-tree-sitter feature flag snapshot. The real backend now lives in
// `tree_sitter/` as a directory module.
#[cfg(feature = "tree-sitter")]
pub mod tree_sitter;

use std::path::Path;

use crate::types::Symbol;

/// A language-aware symbol extractor. Implementations are expected to
/// be pure functions: same input, same output, no I/O side effects.
pub trait SymbolExtractor: Send + Sync {
    /// Human-readable name of the language. Used in diagnostics.
    fn language(&self) -> &'static str;

    /// File extensions this extractor claims (lower-cased, no leading dot).
    fn extensions(&self) -> &'static [&'static str];

    /// Extract symbols from a file's raw source.
    fn extract(&self, source: &str) -> Vec<Symbol>;
}

/// Registry that picks an extractor based on a file's extension.
pub struct ExtractorRegistry {
    extractors: Vec<Box<dyn SymbolExtractor>>,
}

impl Default for ExtractorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ExtractorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractorRegistry")
            .field("count", &self.extractors.len())
            .finish()
    }
}

impl ExtractorRegistry {
    /// Build the default registry. When the `tree-sitter` Cargo
    /// feature is enabled, we populate it with the AST-backed
    /// extractors (multi-line signatures, string/comment-aware,
    /// method-vs-function accuracy). Without the feature it falls
    /// back to the deterministic regex extractors so the default
    /// build stays small and ML/C-free.
    pub fn new() -> Self {
        #[cfg(feature = "tree-sitter")]
        {
            return Self::new_tree_sitter();
        }
        #[cfg(not(feature = "tree-sitter"))]
        {
            Self::new_regex()
        }
    }

    /// Regex backend — always available. Covers 10 languages:
    /// Rust, Python, JavaScript/TypeScript, Go, Java, Kotlin, Swift,
    /// C/C++, Ruby, PHP.
    pub fn new_regex() -> Self {
        Self {
            extractors: vec![
                Box::new(rust::RustExtractor),
                Box::new(python::PythonExtractor),
                Box::new(javascript::JavaScriptExtractor),
                Box::new(go::GoExtractor),
                Box::new(java::JavaExtractor),
                Box::new(kotlin::KotlinExtractor),
                Box::new(swift::SwiftExtractor),
                Box::new(c_cpp::CCppExtractor),
                Box::new(ruby::RubyExtractor),
                Box::new(php::PhpExtractor),
            ],
        }
    }

    /// Tree-sitter AST backend — only compiled when the
    /// `tree-sitter` feature is on.
    ///
    /// The registry is a **superset**, not a replacement for the
    /// regex backend:
    ///
    /// - Languages with a tree-sitter grammar wrapper (Rust, Python,
    ///   JavaScript, TypeScript + TSX, Go) get an AST-backed
    ///   extractor wrapped in [`FallbackExtractor`] so a parse
    ///   failure silently retries with the regex backend.
    /// - Languages *without* a tree-sitter wrapper (Java, Kotlin,
    ///   Swift, C/C++, Ruby, PHP) keep their regex extractor
    ///   registered as-is. Without this, enabling the feature would
    ///   silently drop coverage for 6 languages — a bug reported by
    ///   the § 7.7 bench (sinatra went to 0 symbols).
    #[cfg(feature = "tree-sitter")]
    pub fn new_tree_sitter() -> Self {
        Self {
            extractors: vec![
                // AST-backed wrappers for the 5 languages we have grammars for.
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::rust_ts::new()),
                    Box::new(rust::RustExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::python_ts::new()),
                    Box::new(python::PythonExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::javascript_ts::new()),
                    Box::new(javascript::JavaScriptExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::typescript_ts::new_ts()),
                    Box::new(javascript::JavaScriptExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::typescript_ts::new_tsx()),
                    Box::new(javascript::JavaScriptExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::go_ts::new()),
                    Box::new(go::GoExtractor),
                )),
                // AST-backed wrappers for the remaining languages —
                // every one of them still falls back to its regex
                // extractor on parse failure via FallbackExtractor.
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::ruby_ts::new()),
                    Box::new(ruby::RubyExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::java_ts::new()),
                    Box::new(java::JavaExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::cpp_ts::new()),
                    Box::new(c_cpp::CCppExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::php_ts::new()),
                    Box::new(php::PhpExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::kotlin_ts::new()),
                    Box::new(kotlin::KotlinExtractor),
                )),
                Box::new(FallbackExtractor::new(
                    Box::new(tree_sitter::swift_ts::new()),
                    Box::new(swift::SwiftExtractor),
                )),
            ],
        }
    }

    /// Find the extractor that claims a given file path. Returns
    /// `None` if the extension is unrecognised — callers should skip
    /// the file rather than fall back to a useless generic extractor,
    /// because the generic output would pollute the ranker with junk
    /// references.
    pub fn for_path(&self, path: &Path) -> Option<&dyn SymbolExtractor> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())?;
        for extractor in &self.extractors {
            if extractor.extensions().iter().any(|e| *e == ext) {
                return Some(extractor.as_ref());
            }
        }
        None
    }

    pub fn supported_extensions(&self) -> Vec<&'static str> {
        self.extractors
            .iter()
            .flat_map(|e| e.extensions().iter().copied())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.extractors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.extractors.is_empty()
    }

    /// Scan `dir` for `.toml` plugin files, load each as a
    /// [`plugin::PluginExtractor`], and register it. Returns the list
    /// of language names that were successfully loaded (skips files
    /// that fail to parse with a warning on stderr).
    pub fn load_plugins(&mut self, dir: &Path) -> Vec<String> {
        let mut loaded = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return loaded,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            match plugin::PluginExtractor::from_file(&path) {
                Ok(ext) => {
                    loaded.push(ext.language().to_string());
                    self.extractors.push(Box::new(ext));
                }
                Err(e) => {
                    tracing::warn!(path = ?path, error = %e, "failed to load extractor plugin");
                }
            }
        }
        loaded
    }
}

/// Extractor wrapper that runs `primary` first and, on an empty
/// result, retries with `secondary`. Used by the tree-sitter build
/// to fall back to the regex backend when a file fails to parse.
pub struct FallbackExtractor {
    primary: Box<dyn SymbolExtractor>,
    secondary: Box<dyn SymbolExtractor>,
}

impl FallbackExtractor {
    pub fn new(primary: Box<dyn SymbolExtractor>, secondary: Box<dyn SymbolExtractor>) -> Self {
        Self { primary, secondary }
    }
}

impl SymbolExtractor for FallbackExtractor {
    fn language(&self) -> &'static str {
        self.primary.language()
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.primary.extensions()
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let out = self.primary.extract(source);
        if out.is_empty() && !source.trim().is_empty() {
            return self.secondary.extract(source);
        }
        out
    }
}
