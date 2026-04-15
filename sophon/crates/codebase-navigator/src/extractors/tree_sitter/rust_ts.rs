//! Rust tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/rust.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "rust (ts)",
        &["rs"],
        tree_sitter_rust::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_functions_and_struct() {
        let ext = new();
        let src = "pub fn foo(x: i32) -> i32 { x }\n\npub struct Bar;\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "foo" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn captures_methods_in_impl() {
        let ext = new();
        let src = "
struct Widget;

impl Widget {
    pub fn render(&self) {}
    pub fn area(&self) -> f64 { 0.0 }
}
";
        let syms = ext.extract(src);
        assert!(
            syms.iter()
                .any(|s| s.name == "render" && s.kind == SymbolKind::Method),
            "methods not recognised: {:?}",
            syms
        );
        assert!(syms
            .iter()
            .any(|s| s.name == "area" && s.kind == SymbolKind::Method));
        assert!(syms
            .iter()
            .any(|s| s.name == "Widget" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn ignores_keywords_inside_string_literals() {
        // Regex backend hits this wrongly; tree-sitter should not.
        let ext = new();
        let src = r#"
const GREETING: &str = "pub fn ghost() -> i32";
pub fn real() {}
"#;
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(
            !syms.iter().any(|s| s.name == "ghost"),
            "false positive: {:?}",
            syms
        );
        assert!(syms.iter().any(|s| s.name == "GREETING"));
    }

    #[test]
    fn captures_multi_line_signature() {
        let ext = new();
        let src = "pub fn huge<\n    T: Clone,\n    U: Copy,\n>(\n    a: T,\n    b: U,\n) -> bool { true }\n";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "huge"));
    }
}
