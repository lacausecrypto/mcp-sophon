//! Centralised view of every `SOPHON_*` environment variable Sophon
//! reads at runtime, plus the TOML-backed config sections.
//!
//! This module is the **single source of truth** for which env vars
//! exist, what they do, what types they take, and how they're
//! validated. Individual crates (memory-manager, semantic-retriever,
//! …) still read their own env vars lazily where the knob matters —
//! this module doesn't try to rewire those deep call sites. Instead
//! it:
//!
//! 1. Documents every flag in one list (`RuntimeFlag::ALL`).
//! 2. Parses every flag at startup so invalid values are flagged
//!    **before** a user spends 5 minutes on a failing MCP call.
//! 3. Exposes a [`RuntimeFlags`] snapshot the `sophon doctor`
//!    command prints verbatim.
//!
//! New `SOPHON_*` env vars MUST be added to [`RuntimeFlag::ALL`] so
//! they show up in `sophon doctor` and participate in validation.

use std::path::PathBuf;

/// Scope tag — what the flag configures. Used by `sophon doctor`
/// to group the output sensibly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagScope {
    Paths,
    Transport,
    Retrieval,
    Llm,
    /// Conversation memory & summary maintenance (rolling summary,
    /// thresholds). Distinct from `Llm` because rolling state is
    /// useful even without an LLM configured (heuristic fallback).
    Memory,
    /// Experimental / recall-chasing flags kept for backwards compat
    /// but flagged as deprecated by the v0.5.0 pure-compression
    /// re-scope (see [project_sophon_positioning]).
    DeprecatedRecall,
    Debug,
}

impl FlagScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            FlagScope::Paths => "paths",
            FlagScope::Transport => "transport",
            FlagScope::Retrieval => "retrieval",
            FlagScope::Llm => "llm",
            FlagScope::Memory => "memory",
            FlagScope::DeprecatedRecall => "deprecated-recall",
            FlagScope::Debug => "debug",
        }
    }
}

/// Value shape the flag accepts. The parser accepts any
/// case-insensitive variant of `"1"`, `"true"`, `"on"` for `Bool`
/// flags; `"0"`, `"false"`, `"off"` are treated as disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagKind {
    Bool,
    Path,
    /// Free-form string (command lines, embedder names, …).
    String,
    /// Unsigned integer, optionally bounded.
    UInt {
        min: u64,
        max: u64,
    },
}

/// One `SOPHON_*` env var: name + semantics + validation hint.
#[derive(Debug, Clone)]
pub struct RuntimeFlag {
    pub name: &'static str,
    pub scope: FlagScope,
    pub kind: FlagKind,
    pub description: &'static str,
}

impl RuntimeFlag {
    /// Every `SOPHON_*` env var Sophon reads. Keep alphabetised per
    /// scope so `sophon doctor` output is stable.
    pub const ALL: &'static [RuntimeFlag] = &[
        // -------- paths --------
        RuntimeFlag {
            name: "SOPHON_CONFIG",
            scope: FlagScope::Paths,
            kind: FlagKind::Path,
            description: "Path to a sophon.toml override; defaults to ./sophon.toml if present.",
        },
        RuntimeFlag {
            name: "SOPHON_MEMORY_PATH",
            scope: FlagScope::Paths,
            kind: FlagKind::Path,
            description: "JSONL file for persistent conversation memory (update_memory).",
        },
        RuntimeFlag {
            name: "SOPHON_RETRIEVER_PATH",
            scope: FlagScope::Paths,
            kind: FlagKind::Path,
            description: "Directory for the semantic retriever's chunk store.",
        },
        RuntimeFlag {
            name: "SOPHON_GRAPH_MEMORY_PATH",
            scope: FlagScope::Paths,
            kind: FlagKind::Path,
            description: "JSON snapshot path for the graph-memory store.",
        },
        // -------- retrieval --------
        RuntimeFlag {
            name: "SOPHON_EMBEDDER",
            scope: FlagScope::Retrieval,
            kind: FlagKind::String,
            description: "Embedder backend: `hash` (default, deterministic) or `bge` (needs --features bge).",
        },
        RuntimeFlag {
            name: "SOPHON_HYBRID",
            scope: FlagScope::Retrieval,
            kind: FlagKind::Bool,
            description: "Fuse BM25 sparse-lexical and vector rankings via RRF. Zero latency cost on HashEmbedder.",
        },
        RuntimeFlag {
            name: "SOPHON_MULTIHOP",
            scope: FlagScope::Retrieval,
            kind: FlagKind::Bool,
            description: "Second-pass retrieval on entities extracted from the first-pass results.",
        },
        RuntimeFlag {
            name: "SOPHON_CHUNK_TARGET",
            scope: FlagScope::Retrieval,
            kind: FlagKind::UInt { min: 32, max: 4000 },
            description: "Target chunk size in tokens (default 128). Bigger chunks keep fact clusters together.",
        },
        RuntimeFlag {
            name: "SOPHON_CHUNK_MAX",
            scope: FlagScope::Retrieval,
            kind: FlagKind::UInt { min: 64, max: 8000 },
            description: "Hard cap on chunk size. Must be ≥ SOPHON_CHUNK_TARGET.",
        },
        RuntimeFlag {
            name: "SOPHON_FRAGMENT_MAX_WINDOW",
            scope: FlagScope::Retrieval,
            kind: FlagKind::UInt { min: 32, max: 10_000 },
            description: "Overrides the fragment detector window (default: module-driven).",
        },
        // -------- LLM shell-out --------
        RuntimeFlag {
            name: "SOPHON_LLM_CMD",
            scope: FlagScope::Llm,
            kind: FlagKind::String,
            description: "Command invoked for block-summary LLM calls. Default: `claude -p --model haiku`.",
        },
        RuntimeFlag {
            name: "SOPHON_RERANKER_CMD",
            scope: FlagScope::Llm,
            kind: FlagKind::String,
            description: "Command invoked for LLM reranking (off-thesis — see deprecated list).",
        },
        RuntimeFlag {
            name: "SOPHON_NO_LLM_SUMMARY",
            scope: FlagScope::Llm,
            kind: FlagKind::Bool,
            description: "Explicit opt-out from block-based Haiku summary; falls back to heuristic.",
        },
        // -------- rolling summary (v0.5.2 phase-2B) --------
        RuntimeFlag {
            name: "SOPHON_ROLLING_SUMMARY",
            scope: FlagScope::Memory,
            kind: FlagKind::Bool,
            description: "Maintain a rolling summary at update_memory time so compress_history serves it without re-summarising the full history per query (5-8 s LLM spike → ~1 ms).",
        },
        RuntimeFlag {
            name: "SOPHON_ROLLING_THRESHOLD",
            scope: FlagScope::Memory,
            kind: FlagKind::UInt { min: 10, max: 100_000 },
            description: "Refresh the rolling summary once the un-summarised tail reaches this many messages. Defaults to 50; sanity floor 10.",
        },
        // -------- debug --------
        RuntimeFlag {
            name: "SOPHON_DEBUG_LLM",
            scope: FlagScope::Debug,
            kind: FlagKind::Bool,
            description: "Capture child-process stderr for LLM failures; tracing warnings become richer.",
        },
        // -------- DEPRECATED recall-chasing (v0.4.0 experiments) --------
        // Kept for backwards compat until v0.6.0; sophon doctor flags
        // them so users aren't surprised when they stop working.
        RuntimeFlag {
            name: "SOPHON_HYDE",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Hypothetical-answer query rewrite + RRF fusion. Off-thesis for v0.5.0.",
        },
        RuntimeFlag {
            name: "SOPHON_FACT_CARDS",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Query-time fact-card extraction via extra Haiku call. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_ENTITY_GRAPH",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Heuristic NER + bipartite graph + 1-hop bridge. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_ADAPTIVE",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Haiku classifier adapts top_k / budget per query. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_LLM_RERANK",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "LLM re-scores top-(3×k) retrieval candidates. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_TAIL_SUMMARY",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Haiku summarises chunks beyond top-K. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_REACT",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Iterative retrieval with LLM decider. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_REACT_MAX_ROUNDS",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::UInt { min: 1, max: 10 },
            description: "Cap on ReAct iterations. Ignored when SOPHON_REACT is off.",
        },
        RuntimeFlag {
            name: "SOPHON_GRAPH_MEMORY",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Ingest-time triple extraction + pure-Rust graph queries. Off-thesis.",
        },
        RuntimeFlag {
            name: "SOPHON_MULTIHOP_LLM",
            scope: FlagScope::DeprecatedRecall,
            kind: FlagKind::Bool,
            description: "Route the entity-extraction multi-hop step through an LLM call. Off-thesis.",
        },
    ];
}

/// A parsed, validated snapshot of every runtime flag + the resolved
/// TOML-backed config paths. Produced once at startup.
#[derive(Debug, Clone, Default)]
pub struct RuntimeFlags {
    pub set_flags: Vec<(String, String)>,
    pub warnings: Vec<String>,
    pub deprecated_in_use: Vec<String>,
}

impl RuntimeFlags {
    /// Walk every [`RuntimeFlag::ALL`] entry, pull from the process
    /// env, validate, and collect:
    ///
    /// * `set_flags` — pairs of `(name, sanitised value)` for every
    ///   env var actually set in the current process (the doctor
    ///   output for the user).
    /// * `warnings` — human-readable diagnostics for values that
    ///   failed validation (unparseable integer, out-of-range,
    ///   non-existent path, bogus bool).
    /// * `deprecated_in_use` — names of `FlagScope::DeprecatedRecall`
    ///   flags the caller has actually enabled. These are *not*
    ///   errors — they still work — but `sophon doctor` surfaces
    ///   them so users know they're off-thesis.
    pub fn from_env() -> Self {
        let mut out = RuntimeFlags::default();
        for flag in RuntimeFlag::ALL {
            let Ok(raw) = std::env::var(flag.name) else {
                continue;
            };
            let sanitised = sanitise(flag, &raw);
            out.set_flags.push((flag.name.to_string(), sanitised));
            if let Some(warn) = validate(flag, &raw) {
                out.warnings.push(warn);
            }
            if flag.scope == FlagScope::DeprecatedRecall && is_truthy(&raw) {
                out.deprecated_in_use.push(flag.name.to_string());
            }
        }
        out
    }

    /// Emit warnings via `tracing::warn!` so they land in the stderr
    /// subscriber with the rest of Sophon's startup diagnostics.
    /// Called from `main.rs` early in `sophon serve` startup.
    pub fn log_warnings(&self) {
        for w in &self.warnings {
            tracing::warn!("{w}");
        }
        for name in &self.deprecated_in_use {
            tracing::warn!(
                flag = %name,
                "deprecated recall-chasing flag in use; will be removed in a future version. \
                See v0.5.0 positioning note: Sophon focuses on pure compression, not LOCOMO recall."
            );
        }
    }
}

/// "on" / "off" / "42" / "/path/to/store.jsonl" — we never log the
/// raw value for path-like / command-like flags (privacy), just
/// whether it was set. Bool flags are surfaced as `"on"` / `"off"`
/// so the doctor output is skim-able.
fn sanitise(flag: &RuntimeFlag, raw: &str) -> String {
    match flag.kind {
        FlagKind::Bool => {
            if is_truthy(raw) {
                "on".to_string()
            } else {
                "off".to_string()
            }
        }
        FlagKind::Path | FlagKind::String => "<set>".to_string(),
        FlagKind::UInt { .. } => raw.trim().to_string(),
    }
}

fn validate(flag: &RuntimeFlag, raw: &str) -> Option<String> {
    match flag.kind {
        FlagKind::Bool => {
            // Accept anything, but warn for values that look like a
            // typo (neither "0/false/off" nor "1/true/on").
            if !is_truthy(raw) && !is_falsy(raw) {
                Some(format!(
                    "{}: unrecognised bool value {raw:?}; interpreted as disabled",
                    flag.name
                ))
            } else {
                None
            }
        }
        FlagKind::Path => {
            let path = PathBuf::from(raw);
            let expanded = expand_tilde(&path);
            // A non-existent path is not fatal — the caller may want
            // Sophon to create it on first write — but we warn if the
            // parent directory doesn't exist either, since that's a
            // common misconfiguration (typo'd directory).
            if let Some(parent) = expanded.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    return Some(format!(
                        "{}: parent directory does not exist: {}",
                        flag.name,
                        parent.display()
                    ));
                }
            }
            None
        }
        FlagKind::String => None,
        FlagKind::UInt { min, max } => match raw.trim().parse::<u64>() {
            Ok(n) if n < min || n > max => Some(format!(
                "{}: value {n} out of range [{min}..={max}]",
                flag.name
            )),
            Ok(_) => None,
            Err(_) => Some(format!(
                "{}: expected unsigned integer, got {raw:?}",
                flag.name
            )),
        },
    }
}

fn is_truthy(s: &str) -> bool {
    let s = s.trim();
    s == "1" || s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("on")
}

fn is_falsy(s: &str) -> bool {
    let s = s.trim();
    s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")
}

fn expand_tilde(path: &std::path::Path) -> PathBuf {
    if let Some(rest) = path.to_str().and_then(|s| s.strip_prefix("~/")) {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_is_uppercase_snake_sophon_prefixed() {
        for f in RuntimeFlag::ALL {
            assert!(
                f.name.starts_with("SOPHON_"),
                "{} must start with SOPHON_",
                f.name
            );
            for c in f.name.chars() {
                assert!(
                    c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit(),
                    "{} contains disallowed char {c}",
                    f.name
                );
            }
        }
    }

    #[test]
    fn flag_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for f in RuntimeFlag::ALL {
            assert!(seen.insert(f.name), "duplicate flag {}", f.name);
        }
    }

    #[test]
    fn truthy_and_falsy_recognise_common_values() {
        for t in ["1", "true", "True", "TRUE", "on", "ON"] {
            assert!(is_truthy(t), "expected truthy: {t}");
        }
        for f in ["0", "false", "off", "FALSE"] {
            assert!(is_falsy(f), "expected falsy: {f}");
        }
        assert!(!is_truthy("yes"));
        assert!(!is_falsy("no"));
    }

    #[test]
    fn validate_bool_warns_on_typo() {
        let flag = &RuntimeFlag::ALL[0];
        // Find any bool flag to exercise this path.
        let bool_flag = RuntimeFlag::ALL
            .iter()
            .find(|f| matches!(f.kind, FlagKind::Bool))
            .unwrap_or(flag);
        assert!(validate(bool_flag, "yeah").is_some());
        assert!(validate(bool_flag, "1").is_none());
        assert!(validate(bool_flag, "0").is_none());
    }

    #[test]
    fn validate_uint_enforces_range() {
        let uint_flag = RuntimeFlag::ALL
            .iter()
            .find(|f| matches!(f.kind, FlagKind::UInt { .. }))
            .expect("at least one UInt flag");
        let FlagKind::UInt { min, max } = uint_flag.kind else {
            panic!()
        };
        assert!(validate(uint_flag, "not-a-number").is_some());
        assert!(validate(uint_flag, &(min.saturating_sub(1)).to_string()).is_some());
        assert!(validate(uint_flag, &max.saturating_add(1).to_string()).is_some());
        let in_range = (min + max) / 2;
        assert!(validate(uint_flag, &in_range.to_string()).is_none());
    }

    #[test]
    fn from_env_collects_only_set_flags() {
        // Use a name that nobody else in the test binary has set.
        const PROBE: &str = "SOPHON_CONFIG";
        std::env::remove_var(PROBE);
        let before = RuntimeFlags::from_env();
        assert!(!before.set_flags.iter().any(|(n, _)| n == PROBE));

        std::env::set_var(PROBE, "/tmp/sophon-doctor-test.toml");
        let after = RuntimeFlags::from_env();
        assert!(
            after.set_flags.iter().any(|(n, _)| n == PROBE),
            "set flag missing from snapshot"
        );
        std::env::remove_var(PROBE);
    }

    #[test]
    fn deprecated_recall_flags_are_tracked() {
        const PROBE: &str = "SOPHON_HYDE";
        std::env::set_var(PROBE, "1");
        let snap = RuntimeFlags::from_env();
        assert!(
            snap.deprecated_in_use.iter().any(|n| n == PROBE),
            "deprecated flag should appear in deprecated_in_use"
        );
        std::env::remove_var(PROBE);
    }
}
