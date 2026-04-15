//! C / C++ tree-sitter backend.
//!
//! The C++ grammar is a superset of C, so we route both `.c`/`.h`
//! and `.cpp`/`.cxx`/`.cc`/`.hpp`/`.hxx` files through it. Users
//! who want strict C parsing can build with the `tree-sitter-c`
//! grammar instead — it's already a dep — but in practice the C++
//! grammar handles plain C cleanly for tag-query purposes.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/cpp.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "c/c++ (ts)",
        &["c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "c++", "h++"],
        tree_sitter_cpp::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_function_and_struct() {
        let ext = new();
        let src = "\
#include <stdio.h>

static int add(int a, int b) { return a + b; }
struct Point { int x; int y; };
";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "add" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "Point" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn captures_class_and_namespace() {
        let ext = new();
        let src = "namespace foo { class Widget { public: void render(); }; }";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "foo" && s.kind == SymbolKind::Module));
        assert!(syms.iter().any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn ignores_keyword_in_string_literal() {
        let ext = new();
        let src = "const char* doc = \"class Ghost {};\"; int real() { return 42; }";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "Ghost"));
    }

    #[test]
    fn captures_enum_and_using() {
        let ext = new();
        let src = "enum class Color { Red, Green }; using Callback = void (*)(int);";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
        assert!(syms.iter().any(|s| s.name == "Callback" && s.kind == SymbolKind::TypeAlias));
    }
}
