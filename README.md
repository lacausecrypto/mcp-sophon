# Sophon

> **Honest token economics for MCP agents.** One Rust binary. Zero ML. Reproducible benchmarks.

[![npm version](https://img.shields.io/npm/v/mcp-sophon.svg?color=blue)](https://www.npmjs.com/package/mcp-sophon)
[![npm downloads](https://img.shields.io/npm/dm/mcp-sophon.svg)](https://www.npmjs.com/package/mcp-sophon)
[![GitHub release](https://img.shields.io/github/v/release/lacausecrypto/mcp-sophon?sort=semver)](https://github.com/lacausecrypto/mcp-sophon/releases)
[![CI](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml/badge.svg)](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-194%20Rust%20%2B%204%20Python-brightgreen.svg)](./BENCHMARK.md)

Sophon is a deterministic context layer for agents speaking the Model
Context Protocol. It compresses prompts, conversation memory, code
digests, file deltas, and shell output — without an embedding model,
without an LLM in the compression path, without a GPU, and without
API keys. **7.2 MB** default Rust binary (**25 MB** with the optional
11-language tree-sitter AST backend), MCP-native, `cl100k_base`-accurate.

Every number below links to the reproducible benchmark script that
produced it. Every caveat is in [BENCHMARK.md](./BENCHMARK.md) — the
source of truth for everything Sophon claims.

---

## The three pillars

### 1. Measured economies, not promised ones

- **67.0 %** of session tokens saved across a real 4-call workflow
  ([BENCHMARK.md § 1.7](./BENCHMARK.md#17-session-level-aggregate-get_token_stats))
- **64.5 % ± 0.5 %** of cross-model tokens saved — **stable across
  Claude Haiku / Sonnet / Opus and Codex low / medium / high**
  ([BENCHMARK.md § 2.1](./BENCHMARK.md#21-token-economics-3-tasks-combined-per-variant))
- **47.6 %** saved on repeated boilerplate via fragment caching, even on
  the first call ([BENCHMARK.md § 1.6](./BENCHMARK.md#16-encode_fragments-on-repeated-boilerplate))
- **99.55 %** wire savings via `read_file_delta` when the client
  already has the right hash ([BENCHMARK.md § 1.5](./BENCHMARK.md#15-read_file_delta-unchanged-resume))
- **~6 ms** compression overhead per call — in-process, no GPU, no LLM

### 2. On-par with the best libraries, at a fraction of the cost

- **vs LLMLingua-2 (Microsoft, EMNLP 2024)** on the same 4 structured
  prompts: Sophon saves **77.3 % in 63 ms**, LLMLingua-2 saves 68.4 %
  in 2 176 ms — **~10 points more saved and ~35× faster**
  ([BENCHMARK.md § 7.8.d](./BENCHMARK.md#78d-head-to-head-sophon-compress_prompt-vs-llmlingua-2))
- **vs mem0-lite on LOCOMO** (same 15 items, same judge, same rubric):
  Sophon **60.0 %** accuracy ≈ mem0-lite **60.0 %** — **tie on
  accuracy, Sophon runs sub-second vs 8.7 minutes and zero LLM calls
  vs ~330**
  ([BENCHMARK.md § 7.8.e](./BENCHMARK.md#78e-mem0-lite-on-locomo--same-item-comparison))
- **BGE-small semantic embedder** (v0.2, `--features bge`): **+6.7 pts**
  over HashEmbedder on LOCOMO, **+66.7 pts on single-hop** queries where
  semantic understanding matters most — zero LLM calls, sub-second
  ([BENCHMARK.md § 7.9](./BENCHMARK.md#79-bge-small-embedder-vs-hashembedder-on-locomo-v02-upgrade))
- **LOCOMO semantic retriever gain**: enabling the opt-in retriever
  adds **+13 accuracy points** over compression alone on open-ended
  questions at N=60
  ([BENCHMARK.md § 3.7](./BENCHMARK.md#37-locomo-open-ended-retrieval-n60))

Every head-to-head above runs on the same machine, same tokenizer
(`tiktoken cl100k_base`), same LLM judge, and same inputs. The
benchmark scripts and raw JSON referenced throughout are bundled
with [BENCHMARK.md](./BENCHMARK.md) — each `§` explicitly lists the
runner script and the output file it produced.

### 3. Honest about what it isn't

**Sophon is not state-of-the-art on any single axis.** We publish our
losses as loudly as our wins:

- **Plain-text semantic compression** — LLMLingua-2 is the honest
  choice when you need to preserve every bit of meaning in an
  unstructured document. Sophon's `compress_prompt` shines on
  *structured* prompts with a query, not on arbitrary prose.
  ([BENCHMARK.md § 7.8.d](./BENCHMARK.md#78d-head-to-head-sophon-compress_prompt-vs-llmlingua-2))
- **Semantic retrieval** — v0.2 adds a real BGE-small embedder
  (`--features bge`, 384-dim, ONNX) that gains +6.7 pts over the
  deterministic `HashEmbedder` on LOCOMO. That's real but modest —
  dedicated vector DBs (Qdrant, LanceDB) with HNSW indexing will
  outperform Sophon's linear-scan k-NN on large corpora (>50k chunks).
  ([BENCHMARK.md § 7.9](./BENCHMARK.md#79-bge-small-embedder-vs-hashembedder-on-locomo-v02-upgrade))
- **Code navigation maturity** — [Aider's repomap](https://aider.chat/docs/repomap.html)
  pioneered the tree-sitter + PageRank approach Sophon uses and
  remains more mature, covering more languages in production
  integration. Sophon's `navigate_codebase` is a faithful Rust
  re-implementation, not a reinvention.
  ([BENCHMARK.md § 7.6](./BENCHMARK.md#76-tree-sitter-vs-regex-backend-on-the-same-5-repos))
- **Public corrections** — [BENCHMARK.md § 3.7](./BENCHMARK.md#37-locomo-open-ended-retrieval-n60)
  documents the jump from an optimistic N=30 headline (+23 pts) to
  the honest N=60 number (+13 pts) after we caught the sample bias
  ourselves. The git log shows the correction was pushed publicly.

If any benchmark here looks too clean, open an issue — we've already
caught and published one regression on ourselves.

---

## What's in the binary

Single Rust binary (**7.2 MB** default, **25 MB** with tree-sitter,
**34 MB** with BGE embedder), MCP stdio server, JSON-RPC 2024-11-05,
eleven tools:

| Tool | What it compresses / why |
|---|---|
| `compress_prompt` | Structured system prompt → query-relevant sections only. Topic-routed section picker, not a learned compressor. |
| `compress_history` | Long conversation → summary + keyword index + verbatim recent window. Optional `query` param activates the semantic retriever (see `SOPHON_RETRIEVER_PATH`) for the LOCOMO retrieval gain in § 3.7. |
| `compress_output` | stdout/stderr of git / cargo test / pytest / vitest / jest / go test / ls / grep / find / docker ps / docker logs → errors + modified files + test failures only. 14 command-aware filters + generic fallback, deterministic. |
| `navigate_codebase` | Repository → token-budgeted map via symbol extraction + reference graph + personalised PageRank. Git-aware (honours `.gitignore`), incremental (mtime-diff cache), parallelised via rayon. **11 AST-backed languages** with `--features tree-sitter` (Rust, Python, JS, TS, TSX, Go, Ruby, Java, C/C++, PHP, Kotlin, Swift) + regex fallback for every file. |
| `update_memory` | Stateful session memory. Append + snapshot without re-sending full history. Opt-in JSONL persistence via `SOPHON_MEMORY_PATH`. |
| `read_file_delta` | ETag-like resume. Returns `Unchanged`/`Delta`/`Full` based on the client's known hash — no bytes on the wire when nothing changed. |
| `write_file_delta` | Atomic staged write with hash-versioned state. |
| `encode_fragments` / `decode_fragments` | Repeated multi-paragraph block detection + substitution. Adaptive sliding window. |
| `count_tokens` | Ground-truth `cl100k_base` token count. |
| `get_token_stats` | Per-module cumulative savings across the session. |

**Everything is deterministic.** No embedding model, no LLM in the
compression path, no vector DB to provision, no API key. Same input →
same output bit-for-bit. Run it in CI, run it air-gapped, run it in a
scratch Docker image.

**Not in scope, on purpose**: multimodal ingestion (OCR, PDF layout,
image description, audio). If you need clean PDF/image text, run
Docling / Marker / Unstructured / LlamaParse upstream and feed the
extracted text into Sophon.

---

## When to use it — and when not

**Reach for Sophon when:**

- You're building an MCP-based agent and want to cut the tokens that
  go out the door without adding a Python runtime, a vector DB, or a
  second LLM call.
- You need CI-reproducible compression behaviour — no model weights
  that silently change, no embedding drift, no non-determinism.
- Your system prompt is structured (XML/markdown sections) and you
  want query-aware section routing.
- You're hitting provider rate limits or cost caps from re-sending
  the same history or files every turn.
- You care about binary size, boot time, and zero external
  dependencies.

**Reach for something else when:**

- You need semantically optimal compression on unstructured text at
  any cost — use [LLMLingua-2](https://github.com/microsoft/LLMLingua).
- You need persistent cross-session memory with LLM-driven fact
  extraction — use [mem0](https://github.com/mem0ai/mem0),
  [Letta](https://github.com/letta-ai/letta), or
  [Zep](https://github.com/getzep/zep).
- You need real OCR / layout analysis on PDFs — use
  [Docling](https://github.com/docling-project/docling), Marker, or
  Unstructured.
- You want provider-side cached billing rather than client-side
  compression — use [Anthropic prompt caching](https://docs.anthropic.com/claude/docs/prompt-caching)
  or OpenAI prompt caching.

Sophon and those tools are **orthogonal**. A real stack will often
run Sophon *in front of* provider caching, not instead of it.

---

## Install

### Via npm (wraps the native binary)

```bash
npm install -g mcp-sophon
sophon --version
```

The postinstall script downloads the right prebuilt binary for your
platform from the GitHub Releases page. Supported: macOS arm64/x64,
Linux arm64/x64, Windows x64.

### Prebuilt binary

Grab the archive for your platform from the
[Releases](https://github.com/lacausecrypto/mcp-sophon/releases) page and
put `sophon` on your `PATH`.

### Build from source

```bash
git clone https://github.com/lacausecrypto/mcp-sophon
cd mcp-sophon/sophon
cargo build --release -p mcp-integration
# default build at target/release/sophon (~7.2 MB, regex extractors only)

# opt into 11-language AST extraction (~25 MB):
cargo build --release -p mcp-integration --features codebase-navigator/tree-sitter

# opt into BGE-small semantic embedder (~34 MB):
cargo build --release -p mcp-integration --features bge
# activate at runtime: SOPHON_EMBEDDER=bge SOPHON_RETRIEVER_PATH=~/.sophon/retriever

# all features (~42 MB):
cargo build --release -p mcp-integration --features "codebase-navigator/tree-sitter,bge"
```

Requires Rust 1.75+.

---

## Quick start

### As an MCP server

```json
{
  "mcpServers": {
    "sophon": {
      "command": "sophon",
      "args": ["serve"]
    }
  }
}
```

### CLI

```bash
sophon compress-prompt --prompt ./system.txt --query "how do I handle errors in Rust" --max-tokens 500
sophon compress-history --input ./history.json
sophon stats --period session
sophon serve                                    # MCP stdio server

# Output compression + CLI hooks
sophon exec -- git status                       # run + compress output
sophon exec -- cargo test                       # failures only, ~90 % smaller
sophon compress-output --cmd "git diff" --input diff.txt

# Transparent hook installation for Claude Code
sophon hook install --agent claude --global
sophon hook status                              # show the 20 rewrite rules
sophon hook uninstall --agent claude --global
```

### Programmatic (one-shot JSON-RPC)

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"compress_prompt","arguments":{"prompt":"<rust>use Result and the ? operator</rust><web>fetch()</web>","query":"rust errors","max_tokens":500}}}' \
  | sophon serve
```

---

## Workspace layout

```
.
├── README.md           ← you are here
├── BENCHMARK.md        ← every number, every caveat (the source of truth)
├── LICENSE             ← MIT
├── server.json         ← MCP registry manifest
├── .github/workflows/  ← CI + release automation
├── npm/                ← npm wrapper package
└── sophon/             ← Rust workspace
    ├── Cargo.toml
    ├── sophon.toml     ← default runtime config
    └── crates/
        ├── sophon-core/          shared types, token/hash helpers
        ├── prompt-compressor/    compress_prompt
        ├── memory-manager/       compress_history, update_memory, persistence
        ├── delta-streamer/       read_file_delta, write_file_delta
        ├── fragment-cache/       encode_fragments, decode_fragments
        ├── semantic-retriever/   chunker + HashEmbedder/BGE-small + linear k-NN
        ├── sophon-storage/       SQLite persistence (WAL, embeddings cache)
        ├── output-compressor/    command-aware stdout/stderr compression
        ├── cli-hooks/            transparent command rewriter + agent installer
        ├── codebase-navigator/   tree-sitter/regex extractors + PageRank + digest
        └── mcp-integration/      stdio server, tool schemas, CLI
```

---

## Configuration

Runtime defaults live in [`sophon/sophon.toml`](./sophon/sophon.toml).
Environment variables:

- `SOPHON_MEMORY_PATH` — JSONL file for persistent session memory
  (e.g. `~/.sophon/memory/default.jsonl`).
- `SOPHON_RETRIEVER_PATH` — directory for the semantic retriever store
  (e.g. `~/.sophon/retriever`). When set, `compress_history` accepts
  an optional `query` parameter and returns top-k retrieved chunks
  alongside the compressed summary (+13-pt LOCOMO gain, § 3.7).
- `SOPHON_EMBEDDER` — `hash` (default) or `bge` (requires `--features
  bge` build). BGE-small adds +6.7 pts on LOCOMO over HashEmbedder
  ([§ 7.9](./BENCHMARK.md#79-bge-small-embedder-vs-hashembedder-on-locomo-v02-upgrade)).
- `SOPHON_FRAGMENT_MAX_WINDOW` — override the fragment detector
  window size.
- `SOPHON_CONFIG` — path to a `sophon.toml` config file.

Per-call overrides are available on every MCP tool argument set
(`max_tokens`, `recent_window`, `include_index`, …).

---

## Honest limitations

Every limitation is documented and measured in
[BENCHMARK.md § 6.6](./BENCHMARK.md#66-honest-limitations). At a glance:

- **No multimodal ingestion.** Images, PDFs, audio, and raw table
  parsing are out of scope. Run Docling / Marker / Unstructured /
  LlamaParse upstream and feed clean text in.
- **`compress_history` is heuristic.** No LLM in the summarization
  path — it's a join + keyword index + recent window. Good from ~20
  messages up; short histories are passed through unchanged so the
  payload never *grows*.
- **Embedder options**: `HashEmbedder` (default) is deterministic but
  keyword-only. `BGE-small` (`--features bge`) adds real semantic
  understanding (+6.7 pts on LOCOMO) but needs a 33 MB ONNX model
  download on first use and adds 27 MB to the binary. Neither matches
  dedicated vector DBs with HNSW on large corpora (>50k chunks).
- **11 AST languages, not every language.** Enabling tree-sitter is
  opt-in (`--features codebase-navigator/tree-sitter`) to keep the
  default build free of C compilation. Languages outside the 11
  supported still fall back to the regex extractor.
- **No cross-session memory by default.** Set `SOPHON_MEMORY_PATH` to
  opt into JSONL persistence; otherwise memory is in-process only.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). PRs especially welcome for:

- TypeScript bindings (Python bindings ship in `sophon-py/`)
- TOML-based extractor plugins for new languages (see
  `crates/codebase-navigator/plugins/haskell.toml` for the format)
- More grammars for `navigate_codebase`
- Running the real `mem0` library on LOCOMO to replace the
  `mem0-lite` surrogate in § 7.8.e

Run the full test suite with:

```bash
cd sophon && cargo test --workspace                                  # 194 tests
cd sophon && cargo test --features codebase-navigator/tree-sitter    # +15 AST tests
cd sophon && cargo test -p semantic-retriever --features bge -- --ignored  # 5 BGE tests (needs model)
cd sophon-py && .venv/bin/pytest tests/                              # 4 Python tests
```

---

## License

MIT — see [LICENSE](./LICENSE). Free to use, fork, modify, and ship.
