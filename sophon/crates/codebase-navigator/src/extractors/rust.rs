//! Rust symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct RustExtractor;

/// Match `pub? (pub(crate)?)? (async)? fn NAME` etc. at the start of a
/// line, tolerating leading whitespace inside an `impl` block (which
/// we treat as a top-level method). We scrub `//` line comments from
/// each line before matching to avoid spurious hits.
static RUST_DECL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:pub(?:\([^)]*\))?\s+)?
        (?:async\s+)?
        (?:unsafe\s+)?
        (?:const\s+)?
        (?P<kw>fn|struct|enum|trait|type|const|static|mod|impl)
        \s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        ",
    )
    .expect("valid rust regex")
});

impl SymbolExtractor for RustExtractor {
    fn language(&self) -> &'static str {
        "rust"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        let mut in_impl_depth: i32 = 0;

        for (idx, raw) in source.lines().enumerate() {
            // Strip line comments conservatively so `// fn foo()`
            // doesn't produce a ghost symbol. We split on `//` only
            // outside of string literals — but tracking strings
            // accurately needs a lexer. For v1 we just split on `//`
            // which produces the correct result for the common case.
            let line = raw.split("//").next().unwrap_or(raw);

            if let Some(caps) = RUST_DECL_RE.captures(line) {
                let kw = &caps["kw"];
                let name = caps["name"].to_string();
                let signature = line.trim().to_string();
                let kind = match kw {
                    "fn" => {
                        // Heuristic: a `fn` inside an impl block is a method.
                        if in_impl_depth > 0 {
                            SymbolKind::Method
                        } else {
                            SymbolKind::Function
                        }
                    }
                    "struct" => SymbolKind::Struct,
                    "enum" => SymbolKind::Enum,
                    "trait" => SymbolKind::Trait,
                    "type" => SymbolKind::TypeAlias,
                    "const" | "static" => SymbolKind::Const,
                    "mod" => SymbolKind::Module,
                    "impl" => {
                        in_impl_depth += 1;
                        continue;
                    }
                    _ => continue,
                };
                out.push(Symbol::new(name, kind, (idx + 1) as u32, signature));
            }

            // Crude brace tracking for impl blocks. Again: good
            // enough for the common case, fails on weird formatting.
            for ch in line.chars() {
                if ch == '{' && in_impl_depth > 0 {
                    // already inside, another nested block
                } else if ch == '}' && in_impl_depth > 0 {
                    in_impl_depth -= 1;
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
        RustExtractor.extract(src)
    }

    #[test]
    fn captures_pub_fn() {
        let src = "pub fn hello(x: i32) -> i32 { x + 1 }";
        let syms = extract(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "hello");
        assert_eq!(syms[0].kind, SymbolKind::Function);
        assert_eq!(syms[0].line, 1);
    }

    #[test]
    fn captures_struct_enum_trait() {
        let src = "\npub struct Foo;\nenum Bar { A, B }\ntrait Baz {}\n";
        let syms = extract(src);
        let kinds: Vec<_> = syms.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SymbolKind::Struct));
        assert!(kinds.contains(&SymbolKind::Enum));
        assert!(kinds.contains(&SymbolKind::Trait));
    }

    #[test]
    fn captures_async_and_unsafe_fn() {
        let src = "pub async fn fetch() {}\nunsafe fn boom() {}\n";
        let syms = extract(src);
        assert_eq!(syms.len(), 2);
        assert!(syms.iter().any(|s| s.name == "fetch"));
        assert!(syms.iter().any(|s| s.name == "boom"));
    }

    #[test]
    fn ignores_keyword_in_comment() {
        let src = "// pub fn ghost() {}\npub fn real() {}";
        let syms = extract(src);
        let names: Vec<_> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"real"));
        assert!(!names.contains(&"ghost"));
    }

    #[test]
    fn captures_const_and_type() {
        let src = "pub const MAX: usize = 10;\npub type Alias = Vec<u8>;";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "MAX" && s.kind == SymbolKind::Const));
        assert!(syms.iter().any(|s| s.name == "Alias" && s.kind == SymbolKind::TypeAlias));
    }
}
