//! Swift tree-sitter backend. Re-enabled with tree-sitter 0.25 —
//! `tree-sitter-swift` 0.7's `LANGUAGE` now converts cleanly to a
//! `tree_sitter::Language` because the core crate understands the
//! shared `LanguageFn` type.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/swift.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "swift (ts)",
        &["swift"],
        tree_sitter_swift::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_class_and_function() {
        let ext = new();
        let src = "class User {\n  func greet() -> String { return \"hi\" }\n}\nfunc main() {}\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "main" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn captures_protocol_and_typealias() {
        let ext = new();
        let src = "protocol Drawable { func draw() }\ntypealias Id = Int\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Drawable" && s.kind == SymbolKind::Interface));
        assert!(syms
            .iter()
            .any(|s| s.name == "Id" && s.kind == SymbolKind::TypeAlias));
    }
}
