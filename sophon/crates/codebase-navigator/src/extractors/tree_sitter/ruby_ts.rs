//! Ruby tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/ruby.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "ruby (ts)",
        &["rb", "rake", "gemspec"],
        tree_sitter_ruby::LANGUAGE.into(),
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
        let src = "class User\n  def initialize(name)\n    @name = name\n  end\nend\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "initialize" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_module_and_nested_method() {
        let ext = new();
        let src = "module Authentication\n  def self.sign_in(user)\n    user.touch\n  end\nend\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "Authentication" && s.kind == SymbolKind::Module));
        assert!(syms.iter().any(|s| s.name == "sign_in" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn ignores_keyword_in_string() {
        let ext = new();
        let src = "msg = \"def ghost; end\"\ndef real\n  42\nend\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
