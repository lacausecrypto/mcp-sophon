//! Core data types shared across the crate.

use serde::{Deserialize, Serialize};

/// What kind of language-level declaration a [`Symbol`] represents.
///
/// Kept deliberately coarse — we don't try to tell a `trait` from a
/// Python `Protocol` or a TypeScript `interface`, we just flag them
/// all as `Interface`. That's enough for an LLM to know "there's a
/// contract named `Foo` in this file".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    TypeAlias,
    Const,
    Module,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Method => "method",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::TypeAlias => "type",
            Self::Const => "const",
            Self::Module => "module",
        }
    }
}

/// A single top-level declaration found in a source file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    /// 1-indexed line number in the source file.
    pub line: u32,
    /// The raw line (trimmed) — what we hand back in the digest.
    /// Extractors are expected to provide the signature line, not
    /// the body, so the caller sees something like
    /// `pub fn foo(x: i32) -> Result<()>`.
    pub signature: String,
}

impl Symbol {
    pub fn new(
        name: impl Into<String>,
        kind: SymbolKind,
        line: u32,
        signature: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            line,
            signature: signature.into(),
        }
    }
}
