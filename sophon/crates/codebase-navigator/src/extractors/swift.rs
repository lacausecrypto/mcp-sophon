//! Swift symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct SwiftExtractor;

/// `(public|private|internal|fileprivate|open)? (final|indirect)* (class|struct|enum|protocol|actor|extension) Name`
static SW_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal|fileprivate|open)\s+)?
        (?:(?:final|indirect)\s+)*
        (?P<kw>class|struct|enum|protocol|actor|extension)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .unwrap()
});

/// `(public|…)? (static|class|mutating|nonmutating|override|open|final)* func Name(`
static SW_FUNC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal|fileprivate|open)\s+)?
        (?:(?:static|class|mutating|nonmutating|override|open|final|required|convenience)\s+)*
        func
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*(?:<[^>]*>)?
        \s*\(
        ",
    )
    .unwrap()
});

/// `typealias Name = ...`
static SW_TYPEALIAS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*(?:public\s+)?typealias\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=").unwrap());

/// Top-level `let` / `var` with a capitalized name (heuristic for
/// module-level constants). Lowercased ones are usually local.
static SW_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal|fileprivate)\s+)?
        (?:let|var)
        \s+
        (?P<name>[A-Z][A-Za-z0-9_]*)
        \s*(?::|=)
        ",
    )
    .unwrap()
});

impl SymbolExtractor for SwiftExtractor {
    fn language(&self) -> &'static str {
        "swift"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["swift"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        let mut class_indents: Vec<usize> = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;
            let trimmed = line.trim().to_string();
            let indent_len = line.len() - line.trim_start().len();

            while let Some(&top) = class_indents.last() {
                if indent_len <= top && !trimmed.is_empty() && !trimmed.starts_with('}') {
                    class_indents.pop();
                } else {
                    break;
                }
            }

            if let Some(caps) = SW_TYPE_RE.captures(line) {
                let kw = &caps["kw"];
                let name = caps["name"].to_string();
                let kind = match kw {
                    "class" | "actor" | "extension" => SymbolKind::Class,
                    "struct" => SymbolKind::Struct,
                    "enum" => SymbolKind::Enum,
                    "protocol" => SymbolKind::Interface,
                    _ => continue,
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
                class_indents.push(indent_len);
                continue;
            }

            if let Some(caps) = SW_TYPEALIAS_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::TypeAlias, line_no, trimmed));
                continue;
            }

            if let Some(caps) = SW_FUNC_RE.captures(line) {
                let name = caps["name"].to_string();
                let kind = if !class_indents.is_empty() && indent_len > 0 {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
                continue;
            }

            if indent_len == 0 {
                if let Some(caps) = SW_CONST_RE.captures(line) {
                    out.push(Symbol::new(&caps["name"], SymbolKind::Const, line_no, trimmed));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        SwiftExtractor.extract(src)
    }

    #[test]
    fn captures_class_struct_enum_protocol_actor() {
        let src = "import Foundation\n\npublic class Widget {}\nstruct Point { var x: Int }\nenum Color { case red }\nprotocol Greeter { func hello() }\nactor Counter {}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "Point" && s.kind == SymbolKind::Struct));
        assert!(syms.iter().any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
        assert!(syms.iter().any(|s| s.name == "Greeter" && s.kind == SymbolKind::Interface));
        assert!(syms.iter().any(|s| s.name == "Counter" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn captures_top_level_func_and_method() {
        let src = "func greet(_ name: String) {}\n\npublic class Widget {\n    func render() {}\n    static func make() -> Widget { Widget() }\n}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "greet" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "render" && s.kind == SymbolKind::Method));
        assert!(syms.iter().any(|s| s.name == "make" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_typealias_and_top_let() {
        let src = "public typealias Callback = (String) -> Void\nlet MaxRetries = 5\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "Callback" && s.kind == SymbolKind::TypeAlias));
        assert!(syms.iter().any(|s| s.name == "MaxRetries" && s.kind == SymbolKind::Const));
    }

    #[test]
    fn ignores_commented_func() {
        let src = "// func ghost() {}\nfunc real() {}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }

    #[test]
    fn captures_extension_as_class() {
        let src = "extension String {\n    func shout() -> String { self.uppercased() }\n}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "String" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "shout" && s.kind == SymbolKind::Method));
    }
}
