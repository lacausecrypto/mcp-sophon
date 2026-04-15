//! Python tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/python.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "python (ts)",
        &["py", "pyi"],
        tree_sitter_python::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_top_level_function_and_class() {
        let ext = new();
        let src = "def foo():\n    pass\n\nclass Bar:\n    def baz(self): pass\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "foo" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "Bar" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "baz" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn skips_nested_inner_function() {
        let ext = new();
        let src = "def outer():\n    def inner():\n        return 1\n    return inner\n";
        let syms = ext.extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "outer");
    }

    #[test]
    fn ignores_keywords_in_string_literal() {
        let ext = new();
        let src = "msg = \"def ghost():\"\ndef real():\n    pass\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
