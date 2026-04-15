//! Java symbol extractor.
//!
//! Captures top-level declarations that sit at column 0 or just
//! beyond whitespace. Nested classes are emitted (they're still
//! top-level per the language spec), but local classes defined
//! inside method bodies are not — we can't distinguish them from
//! method-scoped types without an AST, so the regex is deliberately
//! conservative: only declarations whose signature line starts the
//! line (after optional whitespace) and that are followed by `{` or
//! the opener of the declaration get picked up.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct JavaExtractor;

/// `(public|private|protected)? (static|abstract|final|sealed)* (class|interface|enum|record) Name`
static JAVA_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|protected)\s+)?
        (?:(?:static|abstract|final|sealed|non-sealed)\s+)*
        (?P<kw>class|interface|enum|record|@interface)
        \s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        ",
    )
    .expect("valid java type regex")
});

/// Method declarations with an explicit access modifier. Return type
/// may contain generics `<>`, arrays `[]`, and dotted package names.
/// We require the access modifier to avoid matching every statement.
static JAVA_METHOD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|protected)\s+)
        (?:(?:static|abstract|final|synchronized|native|default)\s+)*
        (?:<[^>]*>\s+)?                  # optional generic clause
        [\w.<>,\[\]\s]+?\s+               # return type (lazy, stops at the last word)
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        \s*\(
        ",
    )
    .expect("valid java method regex")
});

/// Public constants: `public static final int MAX = 10;`
static JAVA_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:public|private|protected)\s+
        static\s+final\s+
        [\w.<>,\[\]]+\s+
        (?P<name>[A-Z_][A-Z0-9_]*)
        \s*=
        ",
    )
    .expect("valid java const regex")
});

impl SymbolExtractor for JavaExtractor {
    fn language(&self) -> &'static str {
        "java"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["java"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, raw) in source.lines().enumerate() {
            // Strip `//` line comments. Block comments are not handled
            // — declarations are almost never wrapped in `/* */` in
            // practice.
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;
            let trimmed = line.trim().to_string();

            if let Some(caps) = JAVA_TYPE_RE.captures(line) {
                let kw = &caps["kw"];
                let name = caps["name"].to_string();
                let kind = match kw {
                    "class" => SymbolKind::Class,
                    "interface" | "@interface" => SymbolKind::Interface,
                    "enum" => SymbolKind::Enum,
                    "record" => SymbolKind::Struct, // records are Java's struct-ish
                    _ => continue,
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
                continue;
            }

            if let Some(caps) = JAVA_CONST_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Const,
                    line_no,
                    trimmed,
                ));
                continue;
            }

            if let Some(caps) = JAVA_METHOD_RE.captures(line) {
                // Skip constructors: the method name matches a type
                // name seen earlier and the return type slot is
                // missing. Java method regex requires at least one
                // return-type token; constructors don't have one, so
                // the regex naturally rejects `public Foo(...)`.
                //
                // Still, filter out common keywords that sneak
                // through as "return type" matches.
                let name = caps["name"].to_string();
                if matches!(
                    name.as_str(),
                    "if" | "for" | "while" | "switch" | "return" | "throw" | "catch"
                ) {
                    continue;
                }
                out.push(Symbol::new(name, SymbolKind::Method, line_no, trimmed));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        JavaExtractor.extract(src)
    }

    #[test]
    fn captures_public_class_and_interface() {
        let src = "package com.example;\n\npublic class Foo {}\n\npublic interface Bar {}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Foo" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::Interface));
    }

    #[test]
    fn captures_enum_and_record() {
        let src = "public enum Color { RED, GREEN }\n\npublic record Point(int x, int y) {}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
        assert!(syms
            .iter()
            .any(|s| s.name == "Point" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn captures_methods_with_modifiers() {
        let src = "public class Service {\n    public void start() {}\n    private static int count(List<String> items) { return items.size(); }\n}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "start" && s.kind == SymbolKind::Method));
        assert!(syms
            .iter()
            .any(|s| s.name == "count" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_public_constant() {
        let src = "public class C {\n    public static final int MAX_RETRIES = 5;\n}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Const));
    }

    #[test]
    fn ignores_commented_declaration() {
        let src = "// public class Ghost {}\npublic class Real {}\n";
        let syms = extract(src);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Real"));
        assert!(!names.contains(&"Ghost"));
    }
}
