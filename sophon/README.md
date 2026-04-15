# Sophon — Rust workspace

> Dev-facing README. User docs live at [`../README.md`](../README.md);
> every benchmark number and caveat lives at
> [`../BENCHMARK.md`](../BENCHMARK.md).

**Honest token economics for MCP agents.** One Rust binary. Zero ML.
Reproducible benchmarks. See the three pillars and the head-to-head
results against LLMLingua-2 and mem0-lite in the main README.

Sophon is a modular Rust workspace implementing deterministic
context management across the LLM interaction lifecycle:

- **Prompt compression** — query-driven section routing on structured prompts
- **Conversation memory** — heuristic summary + keyword index + recent window
- **Semantic retrieval** — deterministic `HashEmbedder` + linear k-NN + JSONL store
- **Session memory** — in-process, optional JSONL persistence
- **File delta streaming** — hash + version, `Unchanged`/`Delta`/`Full`
- **Fragment caching** — repeated-block detection and substitution
- **Output compression** — 14 command-aware filters for stdout/stderr
- **Codebase navigation** — tree-sitter/regex extractors + PageRank-lite + digest
- **CLI hooks** — transparent command rewriting for Claude Code and others

Multimodal handling (image / PDF / table / audio) is **not** part of
Sophon. If you need that, run Docling / Marker / Unstructured /
LlamaParse upstream and feed the extracted text in.

## Workspace layout

```text
sophon/
├── Cargo.toml
├── sophon.toml          runtime defaults
├── crates/
│   ├── sophon-core/          shared types, cl100k_base counting, hashing
│   ├── prompt-compressor/    compress_prompt
│   ├── memory-manager/       compress_history, update_memory, persistence
│   ├── delta-streamer/       read_file_delta, write_file_delta
│   ├── fragment-cache/       encode_fragments, decode_fragments
│   ├── semantic-retriever/   chunker + HashEmbedder + linear k-NN
│   ├── output-compressor/    command-aware stdout/stderr compression
│   ├── cli-hooks/            transparent command rewriter + installer
│   ├── codebase-navigator/   tree-sitter/regex extractors + PageRank + digest
│   └── mcp-integration/      MCP stdio server, CLI, tool schemas
└── tests/
    └── fixtures/
```

## Build & test

```bash
# default build — regex extractors, no C deps, ~17 MB binary
cargo build --release -p mcp-integration
cargo test --workspace

# opt into 11-language AST extraction, ~25 MB binary
cargo build --release -p mcp-integration --features codebase-navigator/tree-sitter
cargo test --features codebase-navigator/tree-sitter
```

The release binary lands at `target/release/sophon`. Full workspace
is 209 tests green on both builds.

## Benches (Criterion)

```bash
cargo bench -p prompt-compressor
cargo bench -p memory-manager
cargo bench -p delta-streamer
```

End-to-end benchmarks (LOCOMO, head-to-head vs LLMLingua-2 / mem0,
real-repo scan) live outside the Rust workspace — every section of
[`../BENCHMARK.md`](../BENCHMARK.md) explicitly names the runner
script and the output file it produced so the numbers can be
reproduced from scratch.

## MCP tools

`mcp-integration` exposes the following tools via `sophon serve` (MCP stdio,
JSON-RPC `initialize` / `notifications/initialized` / `tools/list` /
`tools/call` / `ping`):

- **`compress_prompt`** — `prompt`, `query`, optional `max_tokens`. Topic-matched
  sections are preserved during budget trimming so that query-relevant
  content survives even under a tight budget.
- **`compress_history`** — `messages[]`, optional `max_tokens`, `recent_window`,
  `include_index` (default `false`). Short histories (whose raw tokens
  already fit in the budget) are passed through unchanged instead of being
  re-wrapped with a summary.
- **`update_memory`** — stateful session memory. Append + snapshot without
  re-sending the full history. Set `SOPHON_MEMORY_PATH=~/.sophon/memory/session.jsonl`
  to persist appends to disk across server restarts.
- **`read_file_delta`** — `path`, optional `known_version` / `known_hash`.
  Returns `Unchanged` when the client already has the right hash, `Delta`
  when only a diff has to travel, `Full` on cold read.
- **`write_file_delta`** — atomic staged write.
- **`encode_fragments` / `decode_fragments`** — deduplicate repeated
  multi-paragraph blocks inside a document. Detector window is adaptive
  (up to `paragraphs/2`, hard-capped at 64) and can be overridden with
  `SOPHON_FRAGMENT_MAX_WINDOW`.
- **`count_tokens`** — ground-truth `cl100k_base` token count.
- **`get_token_stats`** — cumulative per-module stats. `compression_ratio` is
  `compressed / original`, clamped to `[0, 1]`; lower is better.

## Notes

- All compression is deterministic and CPU-only. No model weights, no
  embeddings at inference time, no GPU.
- `cl100k_base` via `tiktoken-rs` is the one source of truth for token counts.
- Modules degrade gracefully when inputs are missing rather than panicking.
