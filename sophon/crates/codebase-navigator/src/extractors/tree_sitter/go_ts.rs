//! Go tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/go.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "go (ts)",
        &["go"],
        tree_sitter_go::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_func_method_struct_interface() {
        let ext = new();
        let src = "package foo\n\nfunc Hello() {}\n\nfunc (s *Server) Serve() {}\n\ntype User struct {}\n\ntype Greeter interface { Hi() }\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "Hello" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "Serve" && s.kind == SymbolKind::Method));
        assert!(syms.iter().any(|s| s.name == "User" && s.kind == SymbolKind::Struct));
        assert!(syms.iter().any(|s| s.name == "Greeter" && s.kind == SymbolKind::Interface));
    }

    #[test]
    fn ignores_keywords_in_string() {
        let ext = new();
        let src = "package main\n\nconst doc = \"func ghost() {}\"\n\nfunc real() {}\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
