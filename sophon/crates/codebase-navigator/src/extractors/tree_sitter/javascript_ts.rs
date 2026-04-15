//! JavaScript tree-sitter backend (covers `.js`, `.jsx`, `.mjs`, `.cjs`).
//!
//! TypeScript uses a separate grammar; see `typescript_ts.rs`.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/javascript.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "javascript (ts)",
        &["js", "jsx", "mjs", "cjs"],
        tree_sitter_javascript::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_function_and_class() {
        let ext = new();
        let src = "function foo() {}\nclass Bar {}\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "foo" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn captures_arrow_function_const() {
        let ext = new();
        let src = "export const handler = async (req) => { return 42 }\nconst add = (a,b) => a+b\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "handler" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "add" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn ignores_keywords_in_string() {
        let ext = new();
        let src = "const doc = \"function ghost() {}\"\nfunction real() {}\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
