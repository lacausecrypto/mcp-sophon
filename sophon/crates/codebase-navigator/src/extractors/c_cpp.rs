//! C and C++ symbol extractor.
//!
//! C/C++ is the hardest language for regex extraction because:
//!
//! - function signatures can span multiple lines
//! - return types can contain generics (C++ templates), pointers,
//!   references, const/volatile, namespaces (`std::unique_ptr<Foo>`)
//! - macros can make anything look like anything
//!
//! We go for conservative-but-useful: signatures that fit on one
//! line with a recognisable return type and a parameter list. False
//! positives are minimised by requiring a `{` or `;` on the same
//! line or by anchoring to well-known declaration keywords.
//!
//! This extractor shares one implementation for both languages; the
//! only difference is the registered extension list.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct CCppExtractor;

/// `struct Name` / `class Name` / `union Name` / `enum class Name` /
/// `namespace Name`
static CPP_TYPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:template\s*<[^>]*>\s*)?
        (?P<kw>struct|class|union|enum(?:\s+class)?|namespace)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .unwrap()
});

/// `typedef ... Name;`
static CPP_TYPEDEF_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*typedef\s+[^;{]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;").unwrap());

/// C++11 `using Name = ...;`
static CPP_USING_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:template\s*<[^>]*>\s*)?
        using\s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*=
        ",
    )
    .unwrap()
});

/// `#define NAME ...`
static CPP_DEFINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*#\s*define\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)").unwrap());

/// Function declaration / definition with a recognisable return
/// type. Requires the line to contain `(` and end in either `{`, `;`,
/// `const {`, `noexcept`, or similar — enough to tell a function
/// line from a statement.
///
/// Return type matchers: either a well-known keyword (`void`, `int`,
/// `static`, etc.) or a qualified name, possibly pointer/reference.
static CPP_FUNC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:(?:static|inline|extern|virtual|explicit|constexpr|friend)\s+)*
        (?:(?:const|volatile)\s+)?
        (?P<ret>[A-Za-z_][\w:<>,\s\*&]*?)
        \s+
        (?:[A-Za-z_][\w:]*::)?               # optional class qualifier
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*\(
        [^;{]*?
        \)
        \s*
        (?:const|noexcept|override|final|=\s*0)?
        \s*
        (?:[{;]|->)
        ",
    )
    .unwrap()
});

/// Keywords that should never be tagged as function names even when
/// the regex happens to match — usually control-flow statements.
const RESERVED_FUNCTION_NAMES: &[&str] = &[
    "if", "for", "while", "switch", "return", "throw", "catch", "do", "else", "goto", "break",
    "continue", "case",
];

impl SymbolExtractor for CCppExtractor {
    fn language(&self) -> &'static str {
        "c/c++"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "c++", "h++"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, raw) in source.lines().enumerate() {
            // Strip `//` line comments. Block comments left alone.
            let line = raw.split("//").next().unwrap_or(raw);
            let line_no = (idx + 1) as u32;
            let trimmed = line.trim().to_string();

            if let Some(caps) = CPP_DEFINE_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::Const, line_no, trimmed));
                continue;
            }

            if let Some(caps) = CPP_TYPE_RE.captures(line) {
                let kw = caps["kw"].to_string();
                let kind = if kw.starts_with("struct") {
                    SymbolKind::Struct
                } else if kw.starts_with("class") {
                    SymbolKind::Class
                } else if kw.starts_with("union") {
                    SymbolKind::Struct
                } else if kw.starts_with("enum") {
                    SymbolKind::Enum
                } else if kw == "namespace" {
                    SymbolKind::Module
                } else {
                    continue;
                };
                out.push(Symbol::new(&caps["name"], kind, line_no, trimmed));
                continue;
            }

            if let Some(caps) = CPP_TYPEDEF_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::TypeAlias, line_no, trimmed));
                continue;
            }

            if let Some(caps) = CPP_USING_RE.captures(line) {
                out.push(Symbol::new(&caps["name"], SymbolKind::TypeAlias, line_no, trimmed));
                continue;
            }

            if let Some(caps) = CPP_FUNC_RE.captures(line) {
                let name = caps["name"].to_string();
                if RESERVED_FUNCTION_NAMES.contains(&name.as_str()) {
                    continue;
                }
                // Also skip obvious variable declarations like
                // `int x = foo();` where the matched `name` ends up
                // being the variable. We guarded with the `(…)` +
                // `{|;|->` pattern but an additional check on
                // `caps["ret"]` ensures it isn't empty.
                if caps.name("ret").map(|m| m.as_str().trim().is_empty()).unwrap_or(true) {
                    continue;
                }
                out.push(Symbol::new(name, SymbolKind::Function, line_no, trimmed));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        CCppExtractor.extract(src)
    }

    #[test]
    fn captures_c_functions_and_struct() {
        let src = "#include <stdio.h>\n\nstatic int add(int a, int b) { return a + b; }\nvoid greet(const char *name);\nstruct Point { int x; int y; };\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "add" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "greet" && s.kind == SymbolKind::Function));
        assert!(syms.iter().any(|s| s.name == "Point" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn captures_cpp_class_and_namespace() {
        let src = "namespace foo {\n\nclass Widget {\npublic:\n    void render();\n};\n\n}\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "foo" && s.kind == SymbolKind::Module));
        assert!(syms.iter().any(|s| s.name == "Widget" && s.kind == SymbolKind::Class));
    }

    #[test]
    fn captures_typedef_using_and_define() {
        let src = "typedef unsigned long u64;\nusing Callback = void (*)(int);\n#define MAX_LEN 1024\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "u64" && s.kind == SymbolKind::TypeAlias));
        assert!(syms.iter().any(|s| s.name == "Callback" && s.kind == SymbolKind::TypeAlias));
        assert!(syms.iter().any(|s| s.name == "MAX_LEN" && s.kind == SymbolKind::Const));
    }

    #[test]
    fn skips_control_flow_statements() {
        let src = "int main() {\n    if (x) { do_it(); }\n    while (y) { step(); }\n    return 0;\n}\n";
        let syms = extract(src);
        let names: Vec<_> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"main"));
        assert!(!names.contains(&"if"));
        assert!(!names.contains(&"while"));
    }

    #[test]
    fn captures_enum_class_and_template() {
        let src = "enum class Color { Red, Green };\ntemplate<typename T>\nclass Holder {};\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
        assert!(syms.iter().any(|s| s.name == "Holder" && s.kind == SymbolKind::Class));
    }
}
