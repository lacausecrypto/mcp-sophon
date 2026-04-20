//! End-to-end integration tests for the codebase-navigator pipeline.
//!
//! These exercise the full scanner → extractor → ranker → digest chain
//! that `navigate_codebase` exposes via MCP. Unit tests in each module
//! cover branch-level logic; this file verifies cross-module contracts
//! that can only break when multiple pieces are combined:
//!
//! - Polyglot repos: every first-class regex extractor (Rust, Python,
//!   JS, Go, Java, Ruby) fires side-by-side.
//! - Query biasing: PageRank restart vector lifts the right file to
//!   rank 0 when a query mentions one of its symbols.
//! - Digest budget: `DigestConfig::max_tokens` is honoured even on a
//!   large symbol set.
//! - Plugin loader: a user-supplied TOML plugin extends extraction to
//!   a language Sophon does not ship with, without a Rust rebuild.
//! - Empty tree: the pipeline doesn't panic on an empty directory.
//!
//! The default build (no `tree-sitter` feature) is what ships on npm,
//! so these tests exercise the regex backend. A matching `tree-sitter`
//! run is covered by the feature-gated unit tests.

use std::fs;
use std::path::Path;

use codebase_navigator::{
    scan_directory, DigestConfig, ExtractorRegistry, Navigator, NavigatorConfig,
};
use tempfile::tempdir;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn walkdir_nav() -> Navigator {
    // prefer_git=false so the tests work on any CI runner, even
    // environments where `git` isn't on the PATH.
    Navigator::new(NavigatorConfig {
        prefer_git: false,
        ..Default::default()
    })
}

#[test]
fn polyglot_repo_extracts_symbols_across_every_regex_language() {
    let dir = tempdir().unwrap();
    write(
        &dir.path().join("src/lib.rs"),
        "pub fn rust_entrypoint() {}\npub struct RustData {}\n",
    );
    write(
        &dir.path().join("app.py"),
        "def python_entrypoint():\n    pass\n\nclass PythonHandler:\n    pass\n",
    );
    write(
        &dir.path().join("script.js"),
        "function jsEntrypoint() {}\nclass JsHandler {}\n",
    );
    write(
        &dir.path().join("cmd/main.go"),
        "package main\n\nfunc GoEntrypoint() {}\n\ntype GoHandler struct{}\n",
    );
    write(
        &dir.path().join("Service.java"),
        "public class JavaService {\n    public void javaEntrypoint() {}\n}\n",
    );
    write(
        &dir.path().join("helpers.rb"),
        "def ruby_entrypoint\nend\n\nclass RubyHandler\nend\n",
    );

    let mut nav = walkdir_nav();
    let digest = nav
        .scan_and_digest(dir.path(), None, &DigestConfig::default())
        .expect("scan_and_digest should succeed");

    assert_eq!(digest.total_files_scanned, 6, "all six files should scan");

    // Flatten every captured symbol name across every file.
    let all_symbols: Vec<String> = nav
        .records()
        .iter()
        .flat_map(|r| r.symbols.iter().map(|s| s.name.clone()))
        .collect();

    // One signature per language must survive the extraction. If any
    // one of these regex extractors silently breaks, navigate_codebase
    // would serve a digest that's missing a language.
    for expected in [
        "rust_entrypoint",
        "python_entrypoint",
        "jsEntrypoint",
        "GoEntrypoint",
        "javaEntrypoint",
        "ruby_entrypoint",
    ] {
        assert!(
            all_symbols.iter().any(|s| s == expected),
            "missing symbol {expected} in extracted symbols: {all_symbols:?}"
        );
    }
}

#[test]
fn query_promotes_matching_file_to_rank_0_through_digest() {
    // Three unrelated files — only `auth.rs` defines anything a query
    // for "login" can hit via name / path. Personalised PageRank
    // restart vector should pull that file to rank 0 of the digest.
    let dir = tempdir().unwrap();
    write(
        &dir.path().join("auth.rs"),
        "pub fn login() {}\npub fn logout() {}\n",
    );
    write(
        &dir.path().join("db.rs"),
        "pub fn connect() {}\npub fn close() {}\n",
    );
    write(
        &dir.path().join("notes.rs"),
        "pub fn draft() {}\npub fn archive() {}\n",
    );

    let mut nav = walkdir_nav();
    nav.scan(dir.path()).expect("scan");

    let digest_no_query = nav.digest(None, &DigestConfig::default());
    let digest_with_query = nav.digest(Some("how does login work?"), &DigestConfig::default());

    // Ranking without a query is stable but not necessarily auth.rs
    // first (depends on lexicographic / graph tie-breaks). Ranking
    // *with* a query about login MUST put auth.rs first.
    assert!(
        digest_with_query.files[0].path.ends_with("auth.rs"),
        "query-biased rank 0 should be auth.rs, got {:?}",
        digest_with_query.files[0].path
    );
    // Sanity: the query didn't somehow drop files from the digest.
    assert_eq!(digest_no_query.files.len(), digest_with_query.files.len());
}

#[test]
fn digest_respects_max_tokens_budget_under_symbol_flood() {
    // Seed a repo with many symbols so the default budget actually
    // bites. Use a very small `max_tokens` to force the cap.
    let dir = tempdir().unwrap();
    for i in 0..30 {
        write(
            &dir.path().join(format!("mod_{i}.rs")),
            &format!("pub fn function_{i}() {{}}\npub fn helper_{i}() {{}}\n"),
        );
    }

    let mut nav = walkdir_nav();
    nav.scan(dir.path()).expect("scan");

    let cfg = DigestConfig {
        max_tokens: 120,
        max_symbols_per_file: 2,
        min_rank: 0.0,
    };
    let digest = nav.digest(None, &cfg);

    // `max_tokens` is documented as a SOFT ceiling — the final file
    // added before the budget breach can push the running total a
    // symbol line or two over the cap. The invariant we guard is:
    //
    //   - the `truncated` flag is raised (budget was respected)
    //   - overshoot is bounded to a small margin, not catastrophic
    //
    // A regression that silently removes the cap enforcement would
    // push `total_tokens` toward the unbudgeted size of ~30 files ×
    // 2 symbols ≈ several hundred tokens.
    assert!(
        digest.truncated,
        "expected truncated=true when max_tokens cap limits output"
    );
    let hard_cap = cfg.max_tokens + 32; // small per-file-header margin
    assert!(
        digest.total_tokens <= hard_cap,
        "digest total_tokens = {} exceeds soft-cap margin of {} (cfg.max_tokens={})",
        digest.total_tokens,
        hard_cap,
        cfg.max_tokens
    );
    // And the digest should still produce SOME output — not the
    // degenerate empty response.
    assert!(digest.total_tokens > 0);
    assert!(!digest.files.is_empty());
    // Under the symbol flood, we should NOT be serialising every
    // one of the 30 seeded files.
    assert!(
        digest.files.len() < 30,
        "budget failed to trim file list: {} files emitted",
        digest.files.len()
    );
}

#[test]
fn plugin_loader_registers_a_toml_defined_language() {
    // Drop a minimal Haskell-ish TOML plugin into a plugins dir and
    // load it. This mirrors the Contributing guide's extensibility
    // story: a new language gets symbol extraction without a Rust
    // rebuild.
    let plugin_dir = tempdir().unwrap();
    let plugin_path = plugin_dir.path().join("tinyscript.toml");
    fs::write(
        &plugin_path,
        r#"
[extractor]
name = "Tinyscript"
extensions = ["tscr"]

[[patterns]]
kind = "function"
pattern = '(?m)^fun\s+(\w+)'
name_group = 1
"#,
    )
    .unwrap();

    let mut registry = ExtractorRegistry::new_regex();
    let loaded = registry.load_plugins(plugin_dir.path());
    assert!(
        loaded.iter().any(|n| n == "Tinyscript"),
        "loaded plugins should include Tinyscript, got {loaded:?}"
    );

    // Functional probe: scan a source tree containing a .tscr file
    // with the plugin-enriched registry. If the plugin is correctly
    // wired, the `fun greet` signature is captured as a symbol.
    let src_dir = tempdir().unwrap();
    write(
        &src_dir.path().join("hello.tscr"),
        "fun greet() { emit 'hi' }\nfun farewell() { emit 'bye' }\n",
    );
    let records = scan_directory(src_dir.path(), &registry, 100, 1_000_000, false)
        .expect("scan_directory should succeed");
    assert_eq!(
        records.len(),
        1,
        "plugin-enriched registry should recognise .tscr files"
    );
    let names: Vec<&str> = records[0].symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"greet") && names.contains(&"farewell"),
        "Tinyscript plugin failed to extract function names, got {names:?}"
    );
}

#[test]
fn empty_directory_produces_empty_digest_without_panic() {
    // Edge case that used to panic on an earlier version where
    // rank_files assumed at least one record. Fixed in v0.2.x; this
    // test keeps the regression gate.
    let dir = tempdir().unwrap();
    let mut nav = walkdir_nav();
    let digest = nav
        .scan_and_digest(dir.path(), None, &DigestConfig::default())
        .expect("empty scan should succeed");
    assert_eq!(digest.total_files_scanned, 0);
    assert_eq!(digest.total_symbols_found, 0);
    assert!(digest.files.is_empty());
}

#[test]
fn tree_sitter_fallback_invariant_regex_still_extracts_rust() {
    // Contract: with the default feature set (no `tree-sitter`), the
    // regex backend is what's actually running and MUST still extract
    // top-level Rust items. If someone changes the default to enable
    // tree-sitter without reviewing the binary-size implications,
    // this test continues to pass (the tree-sitter wrapper also falls
    // back to regex on parse failure via `FallbackExtractor`), but
    // the guarantee remains: navigate_codebase produces Rust symbols
    // on a default build.
    let dir = tempdir().unwrap();
    write(
        &dir.path().join("crate.rs"),
        "pub fn alpha() {}\npub struct Beta;\npub trait Gamma {}\n",
    );

    let mut nav = walkdir_nav();
    nav.scan(dir.path()).expect("scan");

    let symbols: Vec<&str> = nav
        .records()
        .iter()
        .flat_map(|r| r.symbols.iter())
        .map(|s| s.name.as_str())
        .collect();
    assert!(symbols.contains(&"alpha"), "fn alpha missing: {symbols:?}");
    assert!(
        symbols.contains(&"Beta"),
        "struct Beta missing: {symbols:?}"
    );
    assert!(
        symbols.contains(&"Gamma"),
        "trait Gamma missing: {symbols:?}"
    );
}
