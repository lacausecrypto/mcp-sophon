//! JavaScript / TypeScript symbol extractor.
//!
//! Covers the common declaration shapes:
//!
//! - `function foo(…)`, `export function foo(…)`
//! - `async function foo(…)`, `export default function foo(…)`
//! - `class Foo …`, `export class Foo …`
//! - `interface Foo …` and `type Foo = …` (TypeScript)
//! - `export const foo = (…) => …` and `export const foo = async (…) => …`
//! - `const FOO = …` at the top level (treated as a `Const`)
//!
//! Extractor limitations (same caveats as `rust.rs`):
//!
//! - Multi-line generic clauses where the name is not on the first
//!   line will be missed. Write your function signature on one line
//!   or enable the tree-sitter backend.
//! - JSX / TSX components declared via `const Foo: React.FC = …` are
//!   matched as a `Const`, not as a `Class`. The LLM can still find
//!   them via the name.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct JavaScriptExtractor;

static JS_FN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+(?:default\s+)?)?
        (?:async\s+)?
        function\*?\s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        ",
    )
    .unwrap()
});

static JS_CLASS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+(?:default\s+)?)?
        (?:abstract\s+)?
        class\s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        ",
    )
    .unwrap()
});

static TS_INTERFACE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+)?
        interface\s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        ",
    )
    .unwrap()
});

static TS_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+)?
        type\s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        \s*=
        ",
    )
    .unwrap()
});

/// `const foo = (…) => …` or `let foo = async (…) => …`. This is how
/// most modern JS/TS code declares functions.
static JS_ARROW_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+(?:default\s+)?)?
        (?:const|let|var)\s+
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        \s*(?::\s*[^=]+)?     # optional TS type annotation
        \s*=\s*
        (?:async\s+)?
        (?:\(|[A-Za-z_$][A-Za-z0-9_$]*\s*=>)   # paren-start or single-arg arrow
        ",
    )
    .unwrap()
});

/// Top-level `const` that isn't an arrow function → treat as Const.
static JS_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+)?
        const\s+
        (?P<name>[A-Z][A-Z0-9_$]*)   # SCREAMING_SNAKE only, to avoid picking up every const
        \s*=
        ",
    )
    .unwrap()
});

impl SymbolExtractor for JavaScriptExtractor {
    fn language(&self) -> &'static str {
        "javascript"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["js", "jsx", "ts", "tsx", "mjs", "cjs"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, raw) in source.lines().enumerate() {
            // Strip `//` line comments. We don't try to handle `/* … */`
            // block comments — rare on a declaration line.
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;

            if let Some(caps) = JS_FN_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Function,
                    line_no,
                    line.trim(),
                ));
                continue;
            }
            if let Some(caps) = JS_CLASS_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Class,
                    line_no,
                    line.trim(),
                ));
                continue;
            }
            if let Some(caps) = TS_INTERFACE_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Interface,
                    line_no,
                    line.trim(),
                ));
                continue;
            }
            if let Some(caps) = TS_TYPE_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::TypeAlias,
                    line_no,
                    line.trim(),
                ));
                continue;
            }
            if let Some(caps) = JS_ARROW_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Function,
                    line_no,
                    line.trim(),
                ));
                continue;
            }
            if let Some(caps) = JS_CONST_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Const,
                    line_no,
                    line.trim(),
                ));
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
        JavaScriptExtractor.extract(src)
    }

    #[test]
    fn captures_function_declarations() {
        let src = "function foo() {}\nexport function bar() {}\nasync function baz() {}\nexport default function qux() {}\n";
        let syms = extract(src);
        let names: Vec<_> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"bar"));
        assert!(names.contains(&"baz"));
        assert!(names.contains(&"qux"));
    }

    #[test]
    fn captures_arrow_function_const() {
        let src =
            "export const handler = async (req) => { return 42 }\nconst add = (a, b) => a + b\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "handler" && s.kind == SymbolKind::Function));
        assert!(syms
            .iter()
            .any(|s| s.name == "add" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn captures_ts_interface_and_type() {
        let src = "export interface Foo { a: number }\nexport type Bar = string | number\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Foo" && s.kind == SymbolKind::Interface));
        assert!(syms
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::TypeAlias));
    }

    #[test]
    fn captures_class() {
        let src = "export abstract class Widget {\n  render() {}\n}\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn captures_screaming_const_only() {
        let src = "const NOT_PICKED = 1\nconst MAX_RETRIES = 5\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Const));
        // NOT_PICKED is screaming snake so it IS picked up. The test
        // just confirms the pattern; lowercase consts are skipped.
        let lowercase_src = "const foo = 1\n";
        assert!(extract(lowercase_src).is_empty());
    }

    #[test]
    fn ignores_commented_declarations() {
        let src = "// function ghost() {}\nfunction real() {}\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "real");
    }
}
