//! Python symbol extractor.
//!
//! Captures top-level `def`, `async def`, and `class` declarations.
//! Nested (inner) functions are intentionally not emitted — they would
//! bloat the digest without giving the caller useful navigation info.
//! Indentation level is what we use to decide "top-level".

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct PythonExtractor;

static PY_DEF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^(?P<indent>\s*)
        (?P<kw>async\s+def|def|class)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .expect("valid python regex")
});

impl SymbolExtractor for PythonExtractor {
    fn language(&self) -> &'static str {
        "python"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["py", "pyi"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        // Stack of open class indents (their own indent level). A def
        // whose indent is strictly deeper than a class on the stack is
        // a method; otherwise it's a top-level function.
        let mut class_indents: Vec<usize> = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            // Strip `# ...` comments.
            let line = strip_py_comment(raw);

            if let Some(caps) = PY_DEF_RE.captures(line) {
                let indent_len = caps.name("indent").map(|m| m.as_str().len()).unwrap_or(0);
                // Pop class stack down to anything at a shallower-or-equal indent.
                while let Some(&top) = class_indents.last() {
                    if indent_len <= top {
                        class_indents.pop();
                    } else {
                        break;
                    }
                }

                let kw = &caps["kw"];
                let name = caps["name"].to_string();
                let signature = line.trim().to_string();

                let kind = if kw == "class" {
                    class_indents.push(indent_len);
                    SymbolKind::Class
                } else if !class_indents.is_empty() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };

                // Skip non-top-level functions: anything nested inside
                // another def (i.e. indentation > 0 and no class on
                // stack at this indent).
                if matches!(kind, SymbolKind::Function) && indent_len > 0 {
                    continue;
                }

                out.push(Symbol::new(name, kind, (idx + 1) as u32, signature));
            }
        }
        out
    }
}

fn strip_py_comment(line: &str) -> &str {
    // Very conservative: only strip if `#` appears and is not inside
    // a preceding string literal. We don't track strings precisely;
    // the heuristic is "first `#` wins" because most code doesn't use
    // `#` inside strings on def/class lines.
    if let Some(idx) = line.find('#') {
        &line[..idx]
    } else {
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        PythonExtractor.extract(src)
    }

    #[test]
    fn captures_top_level_def_and_async() {
        let src = "def foo():\n    pass\n\nasync def bar():\n    pass\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[1].name, "bar");
        assert_eq!(syms[0].kind, SymbolKind::Function);
    }

    #[test]
    fn captures_class_with_methods() {
        let src = "\nclass Greeter:\n    def __init__(self, name):\n        self.name = name\n    def hello(self):\n        return f'hi {self.name}'\n";
        let syms = extract(src);
        let names: Vec<_> = syms.iter().map(|s| (s.name.as_str(), s.kind)).collect();
        assert!(names.contains(&("Greeter", SymbolKind::Class)));
        assert!(names.contains(&("__init__", SymbolKind::Method)));
        assert!(names.contains(&("hello", SymbolKind::Method)));
    }

    #[test]
    fn skips_nested_inner_function() {
        let src = "def outer():\n    def inner():\n        return 1\n    return inner\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "outer");
    }

    #[test]
    fn strips_inline_comments() {
        let src = "# def ghost():\ndef real():\n    pass\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "real");
    }

    #[test]
    fn class_stack_pops_on_dedent() {
        let src = "class A:\n    def a(self): pass\nclass B:\n    def b(self): pass\ndef top():\n    pass\n";
        let syms = extract(src);
        let top = syms.iter().find(|s| s.name == "top").unwrap();
        assert_eq!(top.kind, SymbolKind::Function);
    }
}
