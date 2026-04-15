//! Kotlin symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct KotlinExtractor;

/// `(public|private|internal|protected)? (open|abstract|final|sealed|data|value|enum|inline)* (class|interface|object) Name`
static KT_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal|protected)\s+)?
        (?:(?:open|abstract|final|sealed|data|value|enum|inline|annotation|companion)\s+)*
        (?P<kw>class|interface|object)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .expect("valid kotlin type regex")
});

/// `(public|private|internal|protected)? (suspend|inline|infix|operator|override|open|abstract|final)* fun (<...>)? Name(`
static KT_FUN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal|protected)\s+)?
        (?:(?:suspend|inline|infix|operator|override|open|abstract|final|tailrec|external)\s+)*
        fun
        \s+
        (?:<[^>]*>\s+)?                    # optional generics
        (?:[A-Za-z_][A-Za-z0-9_.<>]*\.)?    # optional receiver type
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*\(
        ",
    )
    .expect("valid kotlin fun regex")
});

/// `typealias Name = ...`
static KT_TYPEALIAS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\s*(?:public|private|internal)?\s*typealias\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=")
        .unwrap()
});

/// Top-level `val`/`var` with screaming-snake or PascalCase name.
/// We skip camelCase ones because there are too many of them in
/// practice.
static KT_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:public|private|internal)\s+)?
        (?:const\s+)?
        (?:val|var)
        \s+
        (?P<name>[A-Z][A-Za-z0-9_]*)
        \s*(?::|=)
        ",
    )
    .unwrap()
});

impl SymbolExtractor for KotlinExtractor {
    fn language(&self) -> &'static str {
        "kotlin"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["kt", "kts"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        // Track class/object nesting via indentation — same trick as
        // Python's extractor. A `fun` whose indent is > 0 and inside
        // a class is a method; otherwise it's a function.
        let mut class_indents: Vec<usize> = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;
            let trimmed = line.trim().to_string();
            let indent_len = line.len() - line.trim_start().len();

            // Pop class stack when we dedent out of a class.
            while let Some(&top) = class_indents.last() {
                if indent_len <= top && !trimmed.is_empty() && !trimmed.starts_with('}') {
                    class_indents.pop();
                } else {
                    break;
                }
            }

            if let Some(caps) = KT_TYPE_RE.captures(line) {
                let kw = &caps["kw"];
                let name = caps["name"].to_string();
                let kind = match kw {
                    "class" => SymbolKind::Class,
                    "interface" => SymbolKind::Interface,
                    "object" => SymbolKind::Class,
                    _ => continue,
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
                class_indents.push(indent_len);
                continue;
            }

            if let Some(caps) = KT_TYPEALIAS_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::TypeAlias, line_no, trimmed));
                continue;
            }

            if let Some(caps) = KT_FUN_RE.captures(line) {
                let name = caps["name"].to_string();
                let kind = if !class_indents.is_empty() && indent_len > 0 {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                out.push(Symbol::new(name, kind, line_no, trimmed));
                continue;
            }

            // Only catch TOP-LEVEL constants (indent 0).
            if indent_len == 0 {
                if let Some(caps) = KT_CONST_RE.captures(line) {
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
        KotlinExtractor.extract(src)
    }

    #[test]
    fn captures_classes_objects_interfaces() {
        let src = "data class User(val name: String)\nobject Singleton\ninterface Repo\nsealed class Result\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "Singleton" && s.kind == SymbolKind::Class));
        assert!(syms.iter().any(|s| s.name == "Repo" && s.kind == SymbolKind::Interface));
        assert!(syms.iter().any(|s| s.name == "Result" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn captures_top_level_and_suspend_fun() {
        let src = "fun greet(name: String) {}\nsuspend fun fetch(): String = \"\"\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "greet" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "fetch" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn captures_methods_inside_class() {
        let src = "class Service {\n    fun start() {}\n    suspend fun handle(req: Request) = Unit\n}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "start" && s.kind == SymbolKind::Method));
        assert!(syms.iter().any(|s| s.name == "handle" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_typealias_and_top_const() {
        let src = "typealias Callback = (String) -> Unit\nconst val MAX_RETRIES = 5\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "Callback" && s.kind == SymbolKind::TypeAlias));
        assert!(syms.iter().any(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Const));
    }

    #[test]
    fn ignores_commented_fun() {
        let src = "// fun ghost() {}\nfun real() {}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
