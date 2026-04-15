//! PHP symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct PhpExtractor;

/// `function name(` at the top of the line. Accepts optional access
/// modifiers and `static` / `abstract` / `final` for methods.
static PHP_FUNC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|protected)\s+)?
        (?:(?:static|abstract|final)\s+)?
        (?:(?:public|private|protected)\s+)?
        function
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*\(
        ",
    )
    .unwrap()
});

/// `class|interface|trait|enum Name`
static PHP_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:abstract|final|readonly)\s+)?
        (?P<kw>class|interface|trait|enum)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .unwrap()
});

/// `const NAME = ...` at class or namespace level (all-caps heuristic).
static PHP_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|protected)\s+)?
        const
        \s+
        (?P<name>[A-Z_][A-Z0-9_]*)
        \s*=
        ",
    )
    .unwrap()
});

/// `namespace Foo\Bar;`
static PHP_NAMESPACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*namespace\s+(?P<name>[A-Za-z_][A-Za-z0-9_\\]*)").unwrap());

impl SymbolExtractor for PhpExtractor {
    fn language(&self) -> &'static str {
        "php"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["php", "phtml"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        let mut class_indents: Vec<usize> = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            // Strip `//` and `#` line comments.
            let line = {
                let c1 = raw.split("//").next().unwrap_or(raw);
                c1.split('#').next().unwrap_or(c1)
            };
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

            if let Some(caps) = PHP_NAMESPACE_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Module,
                    line_no,
                    trimmed,
                ));
                continue;
            }

            if let Some(caps) = PHP_TYPE_RE.captures(line) {
                let kw = &caps["kw"];
                let kind = match kw {
                    "class" => SymbolKind::Class,
                    "interface" => SymbolKind::Interface,
                    "trait" => SymbolKind::Class,
                    "enum" => SymbolKind::Enum,
                    _ => continue,
                };
                out.push(Symbol::new(&caps["name"], kind, line_no, trimmed));
                class_indents.push(indent_len);
                continue;
            }

            if let Some(caps) = PHP_CONST_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Const,
                    line_no,
                    trimmed,
                ));
                continue;
            }

            if let Some(caps) = PHP_FUNC_RE.captures(line) {
                let name = caps["name"].to_string();
                let kind = if !class_indents.is_empty() && indent_len > 0 {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        PhpExtractor.extract(src)
    }

    #[test]
    fn captures_class_interface_trait_enum() {
        let src = "<?php\n\nclass User {}\ninterface Repo {}\ntrait Timestamps {}\nenum Status { case Draft; }\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "Repo" && s.kind == SymbolKind::Interface));
        assert!(syms
            .iter()
            .any(|s| s.name == "Timestamps" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "Status" && s.kind == SymbolKind::Enum));
    }

    #[test]
    fn captures_top_level_and_class_methods() {
        let src = "<?php\n\nfunction greet($name) {}\n\nclass Service {\n    public function start() {}\n    private static function internal() {}\n}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "greet" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "start" && s.kind == SymbolKind::Method));
        assert!(syms
            .iter()
            .any(|s| s.name == "internal" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_namespace_and_const() {
        let src =
            "<?php\n\nnamespace App\\Services;\n\nclass Config {\n    const MAX_RETRIES = 5;\n}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "App\\Services" && s.kind == SymbolKind::Module));
        assert!(syms
            .iter()
            .any(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Const));
    }

    #[test]
    fn ignores_commented_declaration() {
        let src = "<?php\n// function ghost() {}\n# function ghost_too() {}\nfunction real() {}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
        assert!(!syms.iter().any(|s| s.name == "ghost_too"));
    }

    #[test]
    fn abstract_final_modifiers_are_ignored_as_noise() {
        let src = "<?php\n\nabstract class Base {}\nfinal class Widget {}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Base" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
    }
}
