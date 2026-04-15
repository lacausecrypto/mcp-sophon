//! Tree-sitter backend. Gated by the `tree-sitter` Cargo feature.
//!
//! Each per-language extractor loads its grammar, parses the source,
//! and runs a declarative tag query (an `.scm` file embedded via
//! `include_str!`). Captures from the query are mapped to the same
//! [`Symbol`](crate::types::Symbol) shape the regex extractors return,
//! so the rest of the crate (scanner, ranker, digest) is backend-
//! agnostic.
//!
//! Tag queries live in `queries/<language>.scm`. The query format is
//! the one used by tree-sitter itself (`(function_item name: …) @name`);
//! the capture group suffix (`@function`, `@struct`, …) tells the
//! runner which [`SymbolKind`] to emit.

#![cfg(feature = "tree-sitter")]

pub mod cpp_ts;
pub mod go_ts;
pub mod java_ts;
pub mod javascript_ts;
pub mod kotlin_ts;
pub mod php_ts;
pub mod python_ts;
pub mod ruby_ts;
pub mod rust_ts;
pub mod swift_ts;
pub mod typescript_ts;

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use crate::extractors::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

/// Apply a tag query to source and emit [`Symbol`]s. Used by every
/// per-language extractor in this module; the only variance between
/// languages is the `Language` and the `.scm` text.
///
/// Returns an empty vec on parser failure (malformed source, internal
/// error) — the caller should fall back to the regex extractor.
pub fn extract_with_query(
    source: &str,
    language: Language,
    query_text: &str,
) -> Vec<Symbol> {
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let Ok(query) = Query::new(&language, query_text) else {
        return Vec::new();
    };

    let capture_names: Vec<&str> = query.capture_names().iter().map(|s| *s).collect();
    let mut cursor = QueryCursor::new();
    let mut out: Vec<Symbol> = Vec::new();

    let source_bytes = source.as_bytes();

    // tree-sitter 0.23+ returns a `StreamingIterator` from `matches`;
    // iterate with `while let Some(m) = it.next()` rather than `for`.
    let mut matches_iter = cursor.matches(&query, tree.root_node(), source_bytes);
    while let Some(qmatch) = matches_iter.next() {
        // A match can contain several captures: the name (`@name`) and
        // the kind (`@function`, `@method`, etc.). We look for exactly
        // one `@name` and one kind-capture per match.
        let mut name: Option<(String, u32)> = None;
        let mut kind: Option<SymbolKind> = None;

        for capture in qmatch.captures {
            let cap_name = capture_names
                .get(capture.index as usize)
                .copied()
                .unwrap_or("");
            if cap_name == "name" {
                let text = capture
                    .node
                    .utf8_text(source_bytes)
                    .unwrap_or("")
                    .to_string();
                let line = capture.node.start_position().row as u32 + 1;
                name = Some((text, line));
            } else if let Some(k) = capture_name_to_kind(cap_name) {
                kind = Some(k);
            }
        }

        if let (Some((name, line)), Some(kind)) = (name, kind) {
            // Pull the full line for the signature field — matches the
            // shape the regex backends produce.
            let signature = source
                .lines()
                .nth((line - 1) as usize)
                .map(|l| l.trim().to_string())
                .unwrap_or_default();
            out.push(Symbol::new(name, kind, line, signature));
        }
    }
    out
}

fn capture_name_to_kind(name: &str) -> Option<SymbolKind> {
    match name {
        "function" => Some(SymbolKind::Function),
        "method" => Some(SymbolKind::Method),
        "class" => Some(SymbolKind::Class),
        "struct" => Some(SymbolKind::Struct),
        "enum" => Some(SymbolKind::Enum),
        "trait" => Some(SymbolKind::Trait),
        "interface" => Some(SymbolKind::Interface),
        "type" => Some(SymbolKind::TypeAlias),
        "const" => Some(SymbolKind::Const),
        "module" => Some(SymbolKind::Module),
        _ => None,
    }
}

/// Wrapper type so that the language-specific extractors can share
/// storage behind `dyn SymbolExtractor`.
pub struct TreeSitterBackend {
    language_name: &'static str,
    extensions: &'static [&'static str],
    language: Language,
    query_text: &'static str,
}

impl TreeSitterBackend {
    pub fn new(
        language_name: &'static str,
        extensions: &'static [&'static str],
        language: Language,
        query_text: &'static str,
    ) -> Self {
        Self { language_name, extensions, language, query_text }
    }
}

impl SymbolExtractor for TreeSitterBackend {
    fn language(&self) -> &'static str {
        self.language_name
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.extensions
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        extract_with_query(source, self.language.clone(), self.query_text)
    }
}
