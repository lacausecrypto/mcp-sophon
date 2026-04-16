//! TOML-based plugin extractor.
//!
//! Allows adding new language support without writing Rust code.
//! Each `.toml` file defines a language name, file extensions, and
//! a set of regex patterns that map to [`SymbolKind`] variants.

use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

// ── TOML schema ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct PluginToml {
    extractor: ExtractorMeta,
    patterns: Vec<PatternDef>,
}

#[derive(Deserialize)]
struct ExtractorMeta {
    name: String,
    extensions: Vec<String>,
}

#[derive(Deserialize)]
struct PatternDef {
    kind: String,
    pattern: String,
    /// 1-based capture group for the symbol name.
    name_group: usize,
    /// Optional 1-based capture group whose match is appended to the
    /// signature. If omitted the whole match is used as the signature.
    #[serde(default)]
    signature_group: Option<usize>,
}

// ── Compiled runtime representation ─────────────────────────────

struct CompiledPattern {
    regex: Regex,
    kind: SymbolKind,
    name_group: usize,
    signature_group: Option<usize>,
}

/// A symbol extractor driven by a TOML configuration file.
pub struct PluginExtractor {
    /// Leaked so we can return `&'static str` from [`SymbolExtractor::language`].
    language: &'static str,
    /// Leaked so we can return `&'static [&'static str]` from [`SymbolExtractor::extensions`].
    extensions: &'static [&'static str],
    patterns: Vec<CompiledPattern>,
}

impl PluginExtractor {
    /// Load a plugin extractor from a TOML file on disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_toml(&text)
    }

    /// Parse a plugin extractor from a TOML string (useful for tests).
    pub fn from_toml(toml_text: &str) -> Result<Self> {
        let def: PluginToml = toml::from_str(toml_text).context("parsing plugin TOML")?;

        // Leak the language name to get a &'static str.
        let language: &'static str = Box::leak(def.extractor.name.into_boxed_str());

        // Leak the extensions slice.
        let ext_strs: Vec<&'static str> = def
            .extractor
            .extensions
            .into_iter()
            .map(|s| &*Box::leak(s.into_boxed_str()))
            .collect();
        let extensions: &'static [&'static str] = Box::leak(ext_strs.into_boxed_slice());

        let mut patterns = Vec::with_capacity(def.patterns.len());
        for p in def.patterns {
            let regex = Regex::new(&p.pattern)
                .with_context(|| format!("compiling pattern for kind '{}'", p.kind))?;
            let kind =
                parse_kind(&p.kind).with_context(|| format!("unknown symbol kind '{}'", p.kind))?;
            patterns.push(CompiledPattern {
                regex,
                kind,
                name_group: p.name_group,
                signature_group: p.signature_group,
            });
        }

        Ok(Self {
            language,
            extensions,
            patterns,
        })
    }
}

// SAFETY: all fields are `Send + Sync` — the leaked strs are
// immutable and the compiled Regexes are thread-safe.
unsafe impl Send for PluginExtractor {}
unsafe impl Sync for PluginExtractor {}

impl SymbolExtractor for PluginExtractor {
    fn language(&self) -> &'static str {
        self.language
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.extensions
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        for cp in &self.patterns {
            for caps in cp.regex.captures_iter(source) {
                let name = match caps.get(cp.name_group) {
                    Some(m) => m.as_str().to_string(),
                    None => continue,
                };

                // Determine line number (1-indexed).
                let match_start = caps.get(0).unwrap().start();
                let line = source[..match_start].matches('\n').count() as u32 + 1;

                let signature = if let Some(sg) = cp.signature_group {
                    caps.get(sg)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| caps.get(0).unwrap().as_str().to_string())
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                };

                symbols.push(Symbol::new(name, cp.kind, line, signature));
            }
        }
        // Sort by line number so output is stable.
        symbols.sort_by_key(|s| s.line);
        symbols
    }
}

fn parse_kind(s: &str) -> Option<SymbolKind> {
    match s.to_ascii_lowercase().as_str() {
        "function" | "fn" => Some(SymbolKind::Function),
        "method" => Some(SymbolKind::Method),
        "class" => Some(SymbolKind::Class),
        "struct" => Some(SymbolKind::Struct),
        "enum" => Some(SymbolKind::Enum),
        "trait" => Some(SymbolKind::Trait),
        "interface" => Some(SymbolKind::Interface),
        "type" | "type_alias" | "typealias" => Some(SymbolKind::TypeAlias),
        "const" | "constant" => Some(SymbolKind::Const),
        "module" | "mod" => Some(SymbolKind::Module),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn plugin_extracts_haskell_symbols() {
        let toml_text = r#"
[extractor]
name = "Haskell"
extensions = ["hs"]

[[patterns]]
kind = "function"
pattern = '(?m)^(\w+)\s*::\s*(.+)$'
name_group = 1

[[patterns]]
kind = "class"
pattern = '(?m)^data\s+(\w+)'
name_group = 1
"#;

        let source = r#"
module Main where

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

data Tree a = Leaf a | Branch (Tree a) (Tree a)

double :: Int -> Int
double x = x * 2
"#;

        let ext = PluginExtractor::from_toml(toml_text).expect("parse plugin TOML");
        assert_eq!(ext.language(), "Haskell");
        assert_eq!(ext.extensions(), &["hs"]);

        let symbols = ext.extract(source);
        assert_eq!(symbols.len(), 3);

        assert_eq!(symbols[0].name, "factorial");
        assert_eq!(symbols[0].kind, SymbolKind::Function);

        assert_eq!(symbols[1].name, "Tree");
        assert_eq!(symbols[1].kind, SymbolKind::Class);

        assert_eq!(symbols[2].name, "double");
        assert_eq!(symbols[2].kind, SymbolKind::Function);
    }

    #[test]
    fn plugin_from_file_roundtrip() {
        let toml_text = r#"
[extractor]
name = "TestLang"
extensions = ["tl"]

[[patterns]]
kind = "function"
pattern = '(?m)^def\s+(\w+)'
name_group = 1
"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testlang.toml");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(toml_text.as_bytes()).unwrap();
        }

        let ext = PluginExtractor::from_file(&path).expect("load from file");
        assert_eq!(ext.language(), "TestLang");

        let syms = ext.extract("def hello\n  puts 'hi'\nend\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "hello");
    }

    #[test]
    fn registry_load_plugins() {
        use super::super::ExtractorRegistry;

        let toml_text = r#"
[extractor]
name = "Demo"
extensions = ["demo"]

[[patterns]]
kind = "function"
pattern = '(?m)^func\s+(\w+)'
name_group = 1
"#;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("demo.toml"), toml_text).unwrap();
        // non-toml file should be ignored
        std::fs::write(dir.path().join("readme.txt"), "not a plugin").unwrap();

        let mut registry = ExtractorRegistry::new_regex();
        let before = registry.len();
        let loaded = registry.load_plugins(dir.path());
        assert_eq!(loaded, vec!["Demo".to_string()]);
        assert_eq!(registry.len(), before + 1);

        // Verify the new extractor works via the registry.
        let ext = registry.for_path(Path::new("test.demo")).unwrap();
        let syms = ext.extract("func myFunc\n  return 1\nend\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "myFunc");
    }
}
