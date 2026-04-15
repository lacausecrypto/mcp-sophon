//! Java tree-sitter backend.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/java.scm");

pub fn new() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "java (ts)",
        &["java"],
        tree_sitter_java::LANGUAGE.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_class_interface_enum_record() {
        let ext = new();
        let src = "\
package com.example;

public class Foo { public void bar() {} }
public interface Repo { void save(); }
public enum Color { RED, GREEN }
public record Point(int x, int y) {}
";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "Foo" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "Repo" && s.kind == SymbolKind::Interface));
        assert!(syms.iter().any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
        assert!(syms.iter().any(|s| s.name == "Point" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn captures_method_inside_class() {
        let ext = new();
        let src = "public class Svc { public void start() {} private int count(int a) { return a; } }";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "start" && s.kind == SymbolKind::Method));
        assert!(syms.iter().any(|s| s.name == "count" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn ignores_keyword_in_string_literal() {
        let ext = new();
        let src = "public class C { String doc = \"class Ghost {}\"; public void real() {} }";
        let syms = ext.extract(src);
        assert!(syms.iter().any(|s| s.name == "C"));
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "Ghost"));
    }
}
