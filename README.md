# Sophon

> **Deterministic context compression for MCP agents.** One Rust binary. Zero ML at query time. Reproducible benchmarks, real-data measurements.

[![npm version](https://img.shields.io/npm/v/mcp-sophon.svg?color=blue)](https://www.npmjs.com/package/mcp-sophon)
[![npm total downloads](https://img.shields.io/npm/dt/mcp-sophon.svg)](https://www.npmjs.com/package/mcp-sophon)
[![GitHub release](https://img.shields.io/github/v/release/lacausecrypto/mcp-sophon?sort=semver)](https://github.com/lacausecrypto/mcp-sophon/releases)
[![CI](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml/badge.svg)](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-2025--06--18-purple.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-405%20Rust%20%2B%204%20Python-brightgreen.svg)](./BENCHMARK.md)

Sophon is a deterministic context layer for agents speaking the Model Context Protocol. It compresses prompts, conversation memory, code digests, file deltas, and shell output — without an embedding model at query time, without a GPU, and without API keys.

**Single 5.2 MB Rust binary.** MCP-native. `cl100k_base`-accurate. Default build pulls no Python, no ML weights, no network.

---

## What it does, in 30 seconds

| Tool | What it solves |
|---|---|
| `compress_prompt` | Long structured prompt → keep only sections relevant to the query |
| `compress_history` | Growing conversation → summary + facts + recent window + optional retrieval |
| `compress_output` | Shell stdout/stderr → 21 domain-aware filters (git, cargo, docker, kubectl, JSON, …) |
| `read_file_delta` / `write_file_delta` | Re-reads + edits → diffs only, never the whole file |
| `encode_fragments` | Repeated boilerplate → single token reference |
| `update_memory` | Append turn → JSONL persist + incremental rolling summary |
| `navigate_codebase` | Repo digest with tree-sitter / regex + PageRank, ranked by query |

11 MCP tools total ([full table below](#what-the-binary-ships)).

---

## Real numbers — measured on this repo's own dev cycle

We built four independent benches that each capture a different chunk of an agent's tool traffic. All four run against this repo's actual git history + working tree on the operator's machine. **Reproducible byte-for-byte** by anyone with `cargo build --release`.

| Dimension | What it measures | Saved | Bench |
|---|---|---|---|
| **history** | `compress_history` over real commits | **94.6 %** | [`real_session_capture.py`](./benchmarks/real_session_capture.py) |
| **shell** | `compress_output` on real `git`/`cargo`/`gh`/`ls` stdout | **84.4 %** | [`real_session_shell.py`](./benchmarks/real_session_shell.py) |
| **filereads** | `compress_prompt` on real Rust / Python / Markdown / TOML files | **71.7 %** | [`real_session_filereads.py`](./benchmarks/real_session_filereads.py) |
| **search** | `compress_output` on real `grep`/`find` patterns | **79.5 %** | [`real_session_search.py`](./benchmarks/real_session_search.py) |
| **🎯 Weighted blend** (35/30/20/15) | typical agent session estimate | **84.7 %** | [`real_session_holistic.py`](./benchmarks/real_session_holistic.py) |

`real_session_holistic.py` runs all four sub-benches with `--json`, parses them, and produces the weighted blend. Default weights reflect this repo's observed shape; pass `--weights "history=0.4,..."` to model your own workload.

### USD economy on Claude Opus 4.7

| | Saved per session |
|---|---|
| **Naive input pricing** ($15/MT) | **$2.03** |
| **With prompt caching** (25-turn reads at $1.50/MT) | **$3.24** |

> Pass `--model sonnet` or `--model haiku` to [`real_session_deep_dive.py`](./benchmarks/real_session_deep_dive.py) if you're re-pricing for a cheaper tier.

### Where each dimension falls short (we say it ourselves)

- **history** measures only what `git` captures (commits + diffs) — typically ~5-10 % of a real session's *tool traffic*. The 94.6 % is the **upper bound**, not the typical case.
- **shell** mixes commands that compress well (`git diff` 95 %) with commands that don't (`gh repo view --json` *adds* tokens, **−9 %**). 84.4 % is a real-world average, not a curated highlight.
- **filereads** uncovered that `compress_prompt` on raw source files compresses by budget cap, not by query routing — same file with 3 different queries → identical output. Section detection only fires on structured input (Markdown headers, XML tags). Documented inline in the bench.
- **search** depends entirely on YOUR repo's state. A repo with no TODOs gets 0 % on `grep TODO`.

The blended 84.7 % is napkin-math from a linear weighted average across four real measurements. **Not a cherry-picked synthetic.** Run the benches yourself to verify.

### Other reproducible benchmarks (synthetic, on-thesis)

| Test | Result | Bench |
|---|---|---|
| `compress_output` across 18 command families | **90.1 %** weighted aggregate | [`compress_output_per_command.py`](./benchmarks/compress_output_per_command.py) |
| 25-turn synthetic Claude Code session | **68.1 %** session tokens saved | [`session_token_economics.py`](./benchmarks/session_token_economics.py) |
| `compress_prompt` across 22 prompt shapes | **70.2 %** mean, **36 ms** mean latency | [`prompt_compression_extended.py`](./benchmarks/prompt_compression_extended.py) |
| Code retrieval on "where is X?" questions | **recall@3 = 70 %** (vs grep 10 %, FULL 20 %) | [`repo_qa.py`](./benchmarks/repo_qa.py) |
| vs LLMLingua-2 on structured prompts | **+8.9 pt accuracy at 35× lower latency** | [`llmlingua_compare.py`](./benchmarks/llmlingua_compare.py) |
| **Sophon + Anthropic prompt caching** | **+24 % tokens / +49 % $** on top of caching | [`sophon_plus_prompt_caching.py`](./benchmarks/sophon_plus_prompt_caching.py) |
| **Sophon + mem0** | Additional savings on retrieved memories | [`sophon_plus_mem0.py`](./benchmarks/sophon_plus_mem0.py) |

---

## Why Sophon — "in front of X"

Sophon is **not** a memory platform, a recall system, an OCR stack, or a replacement for provider-side caching. It's a **deterministic compressor that slots in front of** whatever memory / cache / code-nav layer you already use, and attacks the tokens those layers can't.

### In front of Anthropic / OpenAI prompt caching

Provider caching handles the **static** half of a request — system prompt, tool definitions, reused documents. It doesn't touch the dynamic half (growing conversation history, tool outputs). Sophon compresses exactly that half. The two stack cleanly.

> **+24 % tokens / +49 % $** saved on top of prompt caching on a 25-turn Claude session — because the uncached dynamic block is billed at 10× the cached rate. See [`sophon_plus_prompt_caching.py`](./benchmarks/sophon_plus_prompt_caching.py).

### In front of mem0 / Letta / Zep / Graphiti

Memory systems retrieve the right memories. Sophon shrinks what gets sent to the LLM **after** retrieval. If mem0 returns 2 kB of raw memories, `compress_prompt` keeps only the sections the query actually references.

> Honest caveat: on very short retrieved blocks (< ~200 tokens) Sophon's wrapper adds overhead and you should pass through. The bench reports this directly.

### In front of Claude Code / Cursor / Cline

Primary use case. Every repeat file read becomes a `read_file_delta`; every shell command output goes through `compress_output`; every repeated boilerplate block gets a `fragment_cache` token. Install transparently with `sophon hook install --agent claude --global`.

### In front of a RAG pipeline

`navigate_codebase` produces a PageRanked repo digest that a RAG retriever would otherwise spend expensive embedding calls to build. Tree-sitter / regex symbol extraction over 11 languages, sub-second.

### When NOT to use Sophon

- **Long-form conversational recall above 80 %** — Sophon caps at ~40 % on LOCOMO and we don't chase it. Run [mem0](https://github.com/mem0ai/mem0) / [Letta](https://github.com/letta-ai/letta) / [Zep](https://github.com/getzep/zep) for recall, then optionally pipe their output through Sophon.
- **Multi-hop reasoning on massive documents** — that's [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) or [GraphRAG](https://github.com/microsoft/graphrag).
- **OCR / PDF layout** — out of scope. Use [Docling](https://github.com/docling-project/docling) / Marker / Unstructured upstream.
- **Very small inputs (< ~200 tokens)** — Sophon's section scaffolding can cost more than it saves.

---

## Quick start

### Install via npm (recommended)

```bash
npm install -g mcp-sophon
sophon doctor          # verify install + show config
```

The postinstall script downloads the right prebuilt binary for your platform from the [GitHub Releases](https://github.com/lacausecrypto/mcp-sophon/releases) page. Supported: macOS arm64/x64, Linux arm64/x64, Windows x64.

### Build from source

```bash
git clone https://github.com/lacausecrypto/mcp-sophon
cd mcp-sophon/sophon
cargo build --release -p mcp-integration       # ~5.2 MB binary
```

Optional features:

```bash
# 11-language tree-sitter AST extraction (~25 MB):
cargo build --release -p mcp-integration --features codebase-navigator/tree-sitter

# BGE-small semantic embedder (~34 MB), activate with SOPHON_EMBEDDER=bge:
cargo build --release -p mcp-integration --features bge

# All features (~42 MB):
cargo build --release -p mcp-integration --features "codebase-navigator/tree-sitter,bge"
```

Requires Rust 1.75+.

### Wire it into an MCP client

Most clients accept this snippet (Claude Desktop, Claude Code, Cursor, Cline, Continue):

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

Run `sophon doctor` to print the right config path for your client.

### Recommended runtime setup

```bash
# Persistent memory + on-disk retriever store + BM25+Hash hybrid
export SOPHON_MEMORY_PATH=~/.sophon/memory.jsonl
export SOPHON_RETRIEVER_PATH=~/.sophon/retriever
export SOPHON_HYBRID=1

sophon serve
```

### Quick CLI

```bash
sophon exec -- cargo test                       # run + compress combined output
sophon compress-prompt --prompt ./system.txt --query "rust errors" --max-tokens 500
sophon hook install --agent claude --global     # transparent Claude Code integration
sophon stats --period session                   # token savings rollup
```

### Programmatic (one-shot JSON-RPC)

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"compress_prompt","arguments":{"prompt":"<rust>?: operator</rust><web>fetch()</web>","query":"rust errors","max_tokens":500}}}' \
  | sophon serve
```

---

## What the binary ships

**11 MCP tools, all stdio:**

| Tool | What it does |
|---|---|
| `compress_prompt` | Keep query-relevant sections of a long prompt |
| `compress_history` | Summary + facts + recent + optional retrieval over the conversation |
| `compress_output` | Strip noise from command stdout/stderr (21 domain filters + JsonStructural) |
| `navigate_codebase` | tree-sitter / regex digest of a repo, PageRanked by query |
| `update_memory` | Append messages, JSONL persist, optional rolling summary |
| `read_file_delta` | Version/hash-aware file read, unchanged → minimal payload |
| `write_file_delta` | Send edits as diffs, not full files |
| `encode_fragments` / `decode_fragments` | Detect repeated boilerplate, swap with tokens |
| `count_tokens` | `cl100k_base`-accurate token count |
| `get_token_stats` | Session-level savings rollup |

**Binary sizes by feature set:**

| Build | Size |
|---|---|
| Default (regex extractors, HashEmbedder) | **5.2 MB** |
| + tree-sitter (11 languages) | ~25 MB |
| + BGE semantic embedder | ~34 MB |
| All features | ~42 MB |

**MCP protocol:** `2025-06-18`. `notifications/cancelled` actually drops the response (since v0.5.4). Structured JSON-RPC error codes (`-32000..-32099` reserved for Sophon). Infallible dispatcher — a malformed request can't kill the stdio loop.

---

## Configuration

Run **`sophon doctor`** to see every `SOPHON_*` env var currently set with validation warnings. Full catalogue (24 flags) lives in [`runtime_flags.rs`](./sophon/crates/mcp-integration/src/runtime_flags.rs). The flags worth knowing:

| Flag | Effect | Cost |
|---|---|---|
| `SOPHON_RETRIEVER_PATH=/dir` | Activate the semantic retriever (chunk store on disk) | ~0 |
| `SOPHON_MEMORY_PATH=/file.jsonl` | Persistent conversation memory across `sophon serve` runs | ~0 |
| `SOPHON_HYBRID=1` | BM25 sparse-lexical + HashEmbedder fused via RRF | ~1 ms |
| `SOPHON_ROLLING_SUMMARY=1` | Build rolling summary at `update_memory` time, not at query time | LLM call moved to ingest |
| `SOPHON_CHUNK_TARGET=500` | Bigger chunks preserve cross-sentence context | ~0 |
| `SOPHON_EMBEDDER=bge` | Swap HashEmbedder for BGE-small (needs `--features bge`) | model load at startup |
| `SOPHON_LLM_CMD="claude -p --model haiku"` | LLM shell-out command (used by summarizer when configured) | per-call subprocess |

**Deprecated v0.4.0 recall-chasing flags** — `SOPHON_HYDE`, `SOPHON_FACT_CARDS`, `SOPHON_ENTITY_GRAPH`, `SOPHON_ADAPTIVE`, `SOPHON_LLM_RERANK`, `SOPHON_TAIL_SUMMARY`, `SOPHON_REACT`, `SOPHON_GRAPH_MEMORY`, `SOPHON_MULTIHOP_LLM` — chase LOCOMO recall, an axis we no longer optimise. Still functional but `sophon doctor` flags them. Removed in a future major.

---

## Honest limitations

The full list lives in [BENCHMARK.md § 8](./BENCHMARK.md#-8--limitations). Headlines:

- **LOCOMO conversational recall caps at ~40 %.** mem0 / HippoRAG hit 80-90 % with neural retrieval at query time — we chose determinism + sub-100 ms p99 instead. **Pipe mem0 in front of Sophon if you need that recall.**
- **HashEmbedder is keyword-bound.** "favorite food" ↔ "weakness for ginger snaps" doesn't match. Activate BGE (`SOPHON_EMBEDDER=bge`) for semantic recall — costs +25 MB binary + model load.
- **No multimodal ingestion.** Images / PDFs / audio out of scope. Run Docling / Marker / Unstructured upstream.
- **Rolling summary doesn't help on small sessions.** When the un-summarised tail fits the budget, the rolling cache is a no-op. Useful for long-running sessions with `SOPHON_LLM_CMD` set.
- **Some commands don't compress.** `gh repo view --json` *adds* tokens, `git log --oneline` saves 0.4 %. Sophon's job isn't to compress already-compact output — it's to compress redundant verbose output. The benches name the gaps explicitly.

---

## Project layout

```
.
├── README.md           ← you are here
├── BENCHMARK.md        ← full per-section benchmark detail
├── CHANGELOG.md        ← version history + deprecated numbers
├── benchmarks/         ← reproducible scripts for every number above
├── npm/                ← npm wrapper package
└── sophon/crates/      ← 11-crate Rust workspace
    ├── prompt-compressor/    compress_prompt
    ├── memory-manager/       compress_history, update_memory, rolling summary
    ├── delta-streamer/       read/write_file_delta
    ├── fragment-cache/       encode/decode_fragments
    ├── semantic-retriever/   chunker + HashEmbedder + BM25 + entity graph
    ├── output-compressor/    21 command-aware filters + JsonStructural
    ├── codebase-navigator/   tree-sitter / regex + PageRank
    ├── cli-hooks/            transparent agent installer
    └── mcp-integration/      stdio server, async dispatch, cancellation
```

---

## Contributing

PRs welcome. Run the test suite:

```bash
cd sophon && cargo test --workspace --lib --tests --exclude prompt-compressor   # 405 tests
cd sophon && cargo test --features codebase-navigator/tree-sitter               # +AST tests
cd sophon-py && .venv/bin/pytest tests/                                         # 4 Python tests
```

Every benchmark claim is reproducible — pointers to the scripts live in [BENCHMARK.md](./BENCHMARK.md). If a number doesn't reproduce on your machine, open an issue.

Particularly welcome:

- TypeScript bindings (Python bindings ship in `sophon-py/`)
- `gh` family filter (`gh run list`, `gh pr list`, `gh repo view --json`) — the bench shows this is currently a gap
- `SOPHON_EMBEDDER_CMD` shell-out plugin pattern (mirror of `SOPHON_LLM_CMD`) for Voyage / OpenAI / Cohere
- Multi-repo `real_session_holistic.py` runs against popular open-source repos

---

## License

MIT. See [LICENSE](./LICENSE).
