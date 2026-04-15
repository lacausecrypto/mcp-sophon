# Contributing to Sophon

Thanks for considering a contribution. Sophon is a small project with a clear
scope — there's room for focused improvements and honest bug reports.

## Ground rules

- **Honesty over marketing.** If a change regresses a benchmark, say so in
  the PR description. The existing numbers in [BENCHMARK.md](./BENCHMARK.md)
  are reproducible and should stay that way.
- **Deterministic over fancy.** Sophon's entire positioning is
  no-ML-at-inference-time. Please don't add a runtime dependency on a model
  weight, an embedding service, or a vector DB without a strong reason and
  an opt-in flag.
- **Small PRs.** One logical change per PR beats a 2 000-line rewrite.
- **No comments that explain the obvious.** The codebase follows the rule
  of writing comments only when the *why* is non-trivial.

## Dev setup

Requirements:

- Rust 1.75+ (`rustup install stable`)
- Node 20+ (only if you touch the `npm/` wrapper)
- Python 3.9+ (only if you re-run the benchmark scripts)

```bash
git clone https://github.com/lacausecrypto/mcp-sophon
cd mcp-sophon/sophon
cargo build --release -p mcp-integration
cargo test --workspace
```

The release binary ends up at `sophon/target/release/sophon`. You can run
it as an MCP server with `./target/release/sophon serve` and drive it with
JSON-RPC on stdin.

## Tests

Every crate has unit tests and integration tests. Run the whole suite:

```bash
cargo test --workspace --no-fail-fast
```

Expected: 20+ tests, all green. Clippy is run in CI but currently
tolerated-with-warnings; new code should still be clippy-clean.

Format before committing:

```bash
cargo fmt --all
```

## Running the benchmarks locally

The headline benchmark scripts are checked in under
[`benchmarks/`](./benchmarks/) — see
[`benchmarks/README.md`](./benchmarks/README.md) for the full usage
guide. They read paths from environment variables so nothing is
hard-coded to a specific tree.

Quick start from the repo root:

```bash
cargo build --release -p mcp-integration
export SOPHON_BIN=./sophon/target/release/sophon
export SOPHON_REPO_ROOT=$(pwd)

python3 benchmarks/llmlingua_compare.py   # § 7.8.d (Sophon vs LLMLingua-2)
python3 benchmarks/locomo_retrieval.py    # § 3.7 (Sophon 4-condition LOCOMO)
python3 benchmarks/locomo_mem0lite.py     # § 7.8.e (mem0-lite comparison)
python3 benchmarks/repos_scan.py          # § 7.2 / § 7.8.b (real-repo scan)
python3 benchmarks/repos_recall.py        # § 7.3 (recall@K)
```

Sanity-check the numbers you get against BENCHMARK.md before opening a PR
that touches compression logic.

## PR checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo fmt --all --check` passes
- [ ] No new runtime dependencies on ML models / vector DBs without an
      opt-in flag
- [ ] BENCHMARK.md updated if your change affects measured numbers
- [ ] README.md updated if you add/remove/rename a tool
- [ ] Commit message describes the *why*, not just the *what*

## Reporting bugs

Open an issue with:

- What you ran (command line, JSON-RPC payload, CLI version)
- What you expected
- What you got (stderr, JSON response, token counts)
- Platform (OS + arch + Rust version)

For bugs that affect measured benchmark numbers, bonus points for
including a minimal reproducer that can be dropped into the existing
bench scripts.

## Areas where help is especially welcome

These are documented in [BENCHMARK.md § Known limitations](./BENCHMARK.md#known-limitations-and-caveats):

1. **Topic-router analyzer** — "Python function" queries don't currently
   activate `python_guidelines`. Fix is in
   [`crates/prompt-compressor/src/analyzer.rs`](./sophon/crates/prompt-compressor/src/analyzer.rs).
2. **Plain-text compression fallback** — current behavior is "truncate to
   budget". A Selective-Context-style perplexity scorer would be better
   while staying deterministic.
3. **Persistent memory store** — `update_memory` is in-process only. A
   file-backed variant (`~/.sophon/memory/<session>.jsonl`) would close
   the gap with Mem0/Letta without requiring a vector DB.
4. **Python and TypeScript bindings** — for consumers who don't want to
   speak MCP directly.
5. **OCR for `optimize_image` / `optimize_pdf`** — via `tesseract-rs`
   bindings, opt-in at compile time.

## License

By contributing you agree that your contributions will be licensed under
the [MIT License](./LICENSE).
