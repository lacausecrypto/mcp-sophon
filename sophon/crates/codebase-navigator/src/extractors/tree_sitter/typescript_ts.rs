//! TypeScript tree-sitter backend — covers `.ts` and `.tsx`.
//!
//! The `tree_sitter_typescript` crate exposes two languages: plain
//! `language_typescript()` for `.ts` and `language_tsx()` for `.tsx`.
//! We instantiate a separate [`TreeSitterBackend`] for each because
//! the grammar (and therefore the valid query AST) differs subtly.

#![cfg(feature = "tree-sitter")]

use super::TreeSitterBackend;
use crate::extractors::SymbolExtractor;

const QUERY: &str = include_str!("queries/typescript.scm");

pub fn new_ts() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "typescript (ts)",
        &["ts"],
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        QUERY,
    )
}

pub fn new_tsx() -> impl SymbolExtractor {
    TreeSitterBackend::new(
        "typescript-tsx (ts)",
        &["tsx"],
        tree_sitter_typescript::LANGUAGE_TSX.into(),
        QUERY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SymbolKind;

    #[test]
    fn captures_interface_and_type() {
        let ext = new_ts();
        let src = "export interface User { name: string }\nexport type Age = number\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "User" && s.kind == SymbolKind::Interface));
        assert!(syms
            .iter()
            .any(|s| s.name == "Age" && s.kind == SymbolKind::TypeAlias));
    }

    #[test]
    fn captures_functions_and_class() {
        let ext = new_ts();
        let src = "export function foo(): void {}\nexport class Widget {}\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "foo" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn tsx_captures_component() {
        let ext = new_tsx();
        let src =
            "export const Button = (props: {label: string}) => <button>{props.label}</button>\n";
        let syms = ext.extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Button" && s.kind == SymbolKind::Function));
    }
}
