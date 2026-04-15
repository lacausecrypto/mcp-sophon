//! Sophon codebase-navigator — give an LLM agent a compact map of a
//! repository instead of forcing it to read every file.
//!
//! Inspired by Aider's repomap and token-savior's `codebase_*` tools.
//! Closes the third gap identified in the v2 plan: after input
//! compression (`compress_prompt`, `compress_history`, …) and output
//! compression (`compress_output`), there's still the "the agent has
//! no idea where the symbols it needs actually live" problem.
//!
//! The pipeline is:
//!
//! 1. [`Navigator::scan`] walks a directory, skipping the usual build
//!    artefacts and `.git/`, and applies a language-aware
//!    [`SymbolExtractor`] per file to pull function / class / struct /
//!    trait signatures out.
//! 2. [`Navigator::rank`] builds a "file A contains a token that is a
//!    symbol defined in file B" reference graph and runs a
//!    personalised PageRank-lite against an optional query.
//! 3. [`Navigator::digest`] serialises the top-ranked files and
//!    symbols into a token-budgeted text digest that the LLM can
//!    consume directly.
//!
//! Positioning constraint: **no tree-sitter in the default build**.
//! The extractors are deterministic regex-based (ctags-style). A
//! `tree-sitter` Cargo feature reserves the slot for a future
//! AST-level backend — but shipping ~15 MB of compiled grammars in the
//! default binary would break the "~10 MB, zero C deps" promise we
//! made in BENCHMARK.md, so we don't.

pub mod digest;
pub mod extractors;
pub mod navigator;
pub mod ranker;
pub mod scanner;
pub mod types;

pub use digest::{Digest, DigestConfig, FileDigest, SymbolLine};
pub use extractors::{ExtractorRegistry, SymbolExtractor};
pub use navigator::{Navigator, NavigatorConfig, NavigatorError};
pub use ranker::{rank_files, RankResult};
pub use scanner::{scan_directory, FileRecord};
pub use types::{Symbol, SymbolKind};
