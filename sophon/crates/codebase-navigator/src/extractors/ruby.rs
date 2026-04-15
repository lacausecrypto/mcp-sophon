//! Ruby symbol extractor.

use once_cell::sync::Lazy;
use regex::Regex;

use super::SymbolExtractor;
use crate::types::{Symbol, SymbolKind};

pub struct RubyExtractor;

/// `def name` / `def self.name` — names can contain `?` and `!` but we
/// only capture the base identifier.
static RB_DEF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        def
        \s+
        (?:self\s*\.\s*)?
        (?P<name>[A-Za-z_][A-Za-z0-9_]*[?!=]?)
        ",
    )
    .unwrap()
});

/// `class Foo` / `class Foo < Bar`
static RB_CLASS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        class
        \s+
        (?P<name>[A-Z][A-Za-z0-9_:]*)
        ",
    )
    .unwrap()
});

/// `module Name`
static RB_MODULE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        module
        \s+
        (?P<name>[A-Z][A-Za-z0-9_:]*)
        ",
    )
    .unwrap()
});

impl SymbolExtractor for RubyExtractor {
    fn language(&self) -> &'static str {
        "ruby"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rb", "rake", "gemspec"]
    }

    fn extract(&self, source: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        // Track module/class nesting depth via indentation. A `def`
        // whose indent is > 0 and inside a class/module is a method.
        let mut container_indents: Vec<usize> = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            // Strip `#` line comments. Be careful: `#{...}` is string
            // interpolation in Ruby, but on a declaration line we
            // don't expect interpolation.
            let line = strip_ruby_comment(raw);
            let line_no = (idx + 1) as u32;
            let trimmed = line.trim().to_string();
            let indent_len = line.len() - line.trim_start().len();

            while let Some(&top) = container_indents.last() {
                if indent_len <= top && !trimmed.is_empty() && trimmed != "end" {
                    container_indents.pop();
                } else {
                    break;
                }
            }

            if let Some(caps) = RB_CLASS_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Class,
                    line_no,
                    trimmed,
                ));
                container_indents.push(indent_len);
                continue;
            }

            if let Some(caps) = RB_MODULE_RE.captures(line) {
                out.push(Symbol::new(
                    &caps["name"],
                    SymbolKind::Module,
                    line_no,
                    trimmed,
                ));
                container_indents.push(indent_len);
                continue;
            }

            if let Some(caps) = RB_DEF_RE.captures(line) {
                let name = caps["name"].to_string();
                let kind = if !container_indents.is_empty() && indent_len > 0 {
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

fn strip_ruby_comment(line: &str) -> &str {
    // Conservative: cut at the first `#` that isn't inside `#{...}`.
    // On declaration lines, `#` starts a line comment.
    if let Some(idx) = line.find('#') {
        // If the next char is `{`, this is probably string interpolation,
        // not a comment — keep the line.
        let rest = &line[idx..];
        if rest.starts_with("#{") {
            return line;
        }
        return &line[..idx];
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> Vec<Symbol> {
        RubyExtractor.extract(src)
    }

    #[test]
    fn captures_class_and_methods() {
        let src = "class User\n  def initialize(name)\n    @name = name\n  end\n\n  def greet?\n    \"hi\"\n  end\nend\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "User" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "initialize" && s.kind == SymbolKind::Method));
        assert!(syms
            .iter()
            .any(|s| s.name == "greet?" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_module() {
        let src = "module Authentication\n  def self.sign_in(user)\n  end\nend\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Authentication" && s.kind == SymbolKind::Module));
        assert!(syms
            .iter()
            .any(|s| s.name == "sign_in" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn captures_class_inheritance() {
        let src = "class Admin < User\n  def role\n    :admin\n  end\nend\n";
        let syms = extract(src);
        assert!(syms
            .iter()
            .any(|s| s.name == "Admin" && s.kind == SymbolKind::Class));
        assert!(syms
            .iter()
            .any(|s| s.name == "role" && s.kind == SymbolKind::Method));
    }

    #[test]
    fn ignores_commented_def() {
        let src = "# def ghost\n# end\ndef real\n  42\nend\n";
        let syms = extract(src);
        assert!(syms.iter().any(|s| s.name == "real"));
        assert!(!syms.iter().any(|s| s.name == "ghost"));
    }
}
