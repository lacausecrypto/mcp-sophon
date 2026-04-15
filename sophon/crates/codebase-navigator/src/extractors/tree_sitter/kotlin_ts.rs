//! Kotlin tree-sitter backend. Uses `tree-sitter-kotlin-ng`, the
//! actively maintained fork of the community grammar. Still not an
//! official grammar, so the `FallbackExtractor` wrapper remains in
//! place to catch parse failures.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/kotlin.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "kotlin (ts)",
        &["kt", "kts"],
        tree_sitter_kotlin_ng::LANGUAGE.into(),
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
        let src = "class User(val name: String) {\n  fun greet(): String = \"hi, $name\"\n}\nfun main() {}\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "greet"));
        assert!(syms
            .iter()
            .any(|s| s.name == "main" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn captures_object_and_typealias() {
        let ext = new();
        let src = "object Singleton { fun tick() {} }\ntypealias Id = Long\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Singleton" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "Id" && s.kind == SymbolKind::TypeAlias));
    }
}
