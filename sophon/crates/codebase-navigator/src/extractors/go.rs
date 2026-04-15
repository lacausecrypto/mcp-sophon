//! Go symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct GoExtractor;

/// `func Name(...)` or `func (r *T) Name(...)`. The receiver clause
/// is optional and, when present, makes this a method.
static GO_FUNC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*func\s+
        (?P<recv>\([^)]*\)\s+)?
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*\(
        ",
    )
    .unwrap()
});

static GO_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*type\s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s+
        (?P<kw>struct|interface|[A-Za-z_][A-Za-z0-9_\[\]]*)
        ",
    )
    .unwrap()
});

static GO_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*(?:const|var)\s+
        (?P<name>[A-Z][A-Za-z0-9_]*)   # exported only
        \s*(?:=|\s[A-Za-z_])
        ",
    )
    .unwrap()
});

impl SymbolExtractor for GoExtractor {
    fn language(&self) -> &'static str {
        "go"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["go"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, raw) in source.lines().enumerate() {
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;

            if let Some(caps) = GO_FUNC_RE.captures(line) {
                let kind = if caps.name("recv").is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                out.push(Symbol::new(&caps["name"], kind, line_no, line.trim()));
                continue;
            }

            if let Some(caps) = GO_TYPE_RE.captures(line) {
                let kw = &caps["kw"];
                let kind = match kw {
                    "struct" => SymbolKind::Struct,
                    "interface" => SymbolKind::Interface,
                    _ => SymbolKind::TypeAlias,
                };
                out.push(Symbol::new(&caps["name"], kind, line_no, line.trim()));
                continue;
            }

            if let Some(caps) = GO_CONST_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::Const, line_no, line.trim()));
                continue;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        GoExtractor.extract(src)
    }

    #[test]
    fn captures_func_and_method() {
        let src = "package foo\n\nfunc Hello() string { return \"hi\" }\n\nfunc (r *Server) Serve() {}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "Hello" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "Serve" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_struct_and_interface() {
        let src = "type User struct {\n  Name string\n}\n\ntype Greeter interface {\n  Hello()\n}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "User" && s.kind == SymbolKind::Struct));
        assert!(syms.iter().any(|s| s.name == "Greeter" && s.kind == SymbolKind::Interface));
    }

    #[test]
    fn captures_exported_const_and_var() {
        let src = "const MaxRetries = 5\nvar DefaultClient = newClient()\nvar internal = 1\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "MaxRetries"));
        assert!(syms.iter().any(|s| s.name == "DefaultClient"));
        assert!(!syms.iter().any(|s| s.name == "internal"));
    }

    #[test]
    fn ignores_comment_lines() {
        let src = "// func ghost() {}\nfunc real() {}\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "real");
    }
}
