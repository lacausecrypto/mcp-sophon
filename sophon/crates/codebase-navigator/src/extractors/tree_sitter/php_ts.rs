//! PHP tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/php.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "php (ts)",
        &["php", "phtml", "php3", "php4", "php5", "phps"],
        tree_sitter_php::LANGUAGE_PHP.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_class_and_method() {
        let ext = new();
        let src = "<?php\nclass User {\n  public function login() { return true; }\n}\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "login" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_interface_trait_enum() {
        let ext = new();
        let src = "<?php\ninterface I {}\ntrait T {}\nenum Color { case Red; case Green; }\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "I" && s.kind == SymbolKind::Interface));
        assert!(syms.iter().any(|s| s.name == "T" && s.kind == SymbolKind::Interface));
        assert!(syms.iter().any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
    }

    #[test]
    fn captures_namespace_and_function() {
        let ext = new();
        let src = "<?php\nnamespace App\\Http;\nfunction handle() { return 1; }\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.kind == SymbolKind::Module));
        assert!(syms.iter().any(|s| s.name == "handle" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn ignores_keyword_in_string_literal() {
        let ext = new();
        let src = "<?php\n$doc = \"class Ghost {}\";\nfunction real() { return 42; }\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "Ghost"));
    }
}
