# Sophon

> **Honest token economics for MCP agents.** One Rust binary. Zero ML at query time. Reproducible benchmarks.

[![npm version](https://img.shields.io/npm/v/mcp-sophon.svg?color=blue)](https://www.npmjs.com/package/mcp-sophon)
[![npm downloads](https://img.shields.io/npm/dm/mcp-sophon.svg)](https://www.npmjs.com/package/mcp-sophon)
[![GitHub release](https://img.shields.io/github/v/release/lacausecrypto/mcp-sophon?sort=semver)](https://github.com/lacausecrypto/mcp-sophon/releases)
[![CI](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml/badge.svg)](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-303%20Rust%20%2B%204%20Python-brightgreen.svg)](./BENCHMARK.md)

Sophon is a deterministic context layer for agents speaking the Model
Context Protocol. It compresses prompts, conversation memory, code
digests, file deltas, and shell output — without an embedding model at
query time, without a GPU, and without API keys. **7.2 MB** default
Rust binary (**25 MB** with the optional 11-language tree-sitter AST
backend, **34 MB** with BGE embedder), MCP-native, `cl100k_base`-accurate.

Every number below links to the reproducible benchmark script that
produced it. Every caveat is in [BENCHMARK.md](./BENCHMARK.md). Version
history + deprecated numbers live in [CHANGELOG.md](./CHANGELOG.md).

---

## TL;DR — v0.4.0

Sophon is shaped around four use cases. Here's what we measured.

| Use case | Metric | Compared to |
|---|---|---|
| **Agent session token economics** | **68.1 % tokens saved** across 25-turn coding session ([§ 1](./BENCHMARK.md#-1--session-token-economics)) | Baseline: raw tokens |
| **Prompt compression** | **70.2 % mean saved**, **36 ms mean latency**, 22 prompt shapes ([§ 2](./BENCHMARK.md#-2--prompt-compression-extended)) | LLMLingua-2: +8.9 pt at 35× lower latency ([§ 6.1](./BENCHMARK.md#-61-vs-llmlingua-2)) |
| **Code retrieval (repo QA)** | **recall@3 = 70 %** on "where is X?" questions ([§ 4](./BENCHMARK.md#-4--repo-qa)) | grep: 10 % ; FULL context: 20 % |
| **Latency + reliability** | **p99 < 87 ms** on 5/7 ops, **100 % ok_rate** on 190 runs ([§ 3](./BENCHMARK.md#-3--latency--reliability)) | Sub-second guaranteed |

**LOCOMO note (conversational memory benchmark):** V032 full stack
lands at **40 %** (N=30), up from V030's 33 %. Not competitive with
mem0 (91 %, neural). We're honest about that in
[§ 5](./BENCHMARK.md#-5--locomo-honest).

### What changed in v0.4.0

Twelve new opt-in flags. Defaults unchanged from 0.3.0 — a caller
that only sets `SOPHON_RETRIEVER_PATH` gets v0.3.0 behaviour byte for
byte. Full list + measured gain: [CHANGELOG.md](./CHANGELOG.md#040--2026-04-18).

Biggest wins under the hood:
- **Rayon-parallel block summarisation** → `compress_history` with LLM summary dropped from ~40 s to ~5-8 s on 600-turn conversations
- **HyDE + FactCards + EntityGraph stack** → canary preservation on `compress_history` went from 27 % → 80 %
- **Path A graph memory** (experimental) → ingest-time LLM triple extraction, zero LLM at query time

---

## The three pillars

### 1. Measured economies, not promised ones

- **68.1 %** session tokens saved over a 25-turn coding session
  ([§ 1](./BENCHMARK.md#-1--session-token-economics))
- **70.2 %** overall savings on `compress_prompt` across 22 shapes
  ([§ 2](./BENCHMARK.md#-2--prompt-compression-extended))
- **98.0 %** savings on re-reads via `read_file_delta`
- **94.4 %** savings on targeted edits via `write_file_delta`
- **95.4 %** savings on Claude-Code-sized system prompts

### 2. Determinism + speed first

- **p99 ≤ 87 ms** on 5 of 7 ops: `count_tokens`, `compress_prompt`,
  `compress_output`, `read_file_delta`, `navigate_codebase`
- **100 % ok_rate** across 190 bench runs (zero crashes, zero
  malformed payloads)
- **Zero ML at query time** on the default build. Haiku is shell-out
  only, opt-in per feature flag.

### 3. Honest about what it isn't

- **LOCOMO conversational recall** plateaus around 40 % on V032
  full stack. mem0 / HippoRAG hit 80-90 % with neural embeddings at
  query time — we chose determinism + speed instead.
- **Adversarial questions**: V032 loses some ground (HyDE surfaces
  tangential chunks the LLM then hallucinates over). V030 default
  stays at 83 % on adversarial, V032 drops to 67 %.
- **Per-type, not global.** Our +17 pt gains on multi-hop /
  single-hop / temporal are directionally real at N=30 but CIs
  overlap — we flag that explicitly in
  [§ 5.1](./BENCHMARK.md#-51-v030-vs-v032_full-head-to-head).

---

## What's in the binary

11 MCP tools, all stdio:

| Tool | What it does |
|---|---|
| `compress_prompt` | Keep query-relevant sections of a long prompt |
| `compress_history` | Summary + facts + recent + optional retrieval over the conversation |
| `compress_output` | Strip noise from command stdout/stderr (20+ domain filters) |
| `navigate_codebase` | tree-sitter / regex digest of a repo, PageRanked by query |
| `update_memory` | Append messages to the session store (JSONL persist + graph ingest) |
| `read_file_delta` | Version/hash-aware file read, unchanged → minimal payload |
| `write_file_delta` | Send edits as diffs, not full files |
| `encode_fragments` | Detect repeated boilerplate, replace with tokens |
| `decode_fragments` | Reverse the encoding |
| `count_tokens` | `cl100k_base`-accurate token count |
| `get_token_stats` | Session-level savings rollup |

Binary sizes:
- **7.2 MB** default (regex extractors, HashEmbedder)
- **25 MB** with tree-sitter (11 languages: Rust, Python, JS, TS, TSX, Go, Ruby, Java, C/C++, PHP, Kotlin, Swift)
- **34 MB** with BGE-small semantic embedder
- **42 MB** with all features

---

## Feature flags (opt-in)

All new v0.4.0 behaviour is env-gated. Defaults unchanged from 0.3.0.

| Flag | What it adds | Measured gain | Cost |
|---|---|---|---|
| `SOPHON_HYDE=1` | Haiku writes hypothetical answers → retrieval via RRF fusion | +17 pt open-domain, +17 pt single-hop | +1 Haiku call |
| `SOPHON_FACT_CARDS=1` | Entity timeline JSON rendered into context | +17 pt temporal | +1 Haiku call |
| `SOPHON_ENTITY_GRAPH=1` | Heuristic NER + bipartite graph + 1-hop bridge | +17 pt multi-hop | ~10 ms |
| `SOPHON_ADAPTIVE=1` | Haiku classifies query → adapts top_k / budget | +5-10 pt factual | +1 Haiku call |
| `SOPHON_LLM_RERANK=1` | Haiku re-scores top-(3×k) candidates | +8 pt recall | +1 Haiku call |
| `SOPHON_TAIL_SUMMARY=1` | Haiku summarises chunks beyond top-K | +3-5 pt | +1 Haiku call |
| `SOPHON_CHUNK_TARGET=500` | Bigger chunks (default 128) | +5 pt with rerank | ~0 |
| `SOPHON_HYBRID=1` | BM25 + HashEmbedder fused via RRF | +3-5 pt rare-term | ~1 ms |
| `SOPHON_GRAPH_MEMORY=1` | Ingest-time LLM triples → pure-Rust graph query (**experimental**) | See [§ 5.4](./BENCHMARK.md#-54-historical-corrections) | 1 Haiku / ingest batch |
| `SOPHON_REACT=1` | Iterative retrieval with LLM decider (**experimental**) | Mixed — not recommended with HyDE | 2-3 Haiku calls |
| `SOPHON_NO_LLM_SUMMARY=1` | Opt-out from block-based Haiku summary | Speed (bench utility) | — |
| `SOPHON_DEBUG_LLM=1` | Log LLM call failures to stderr | — | — |

**Recommended starter configs**:
- Fast interactive (sub-second): defaults only
- Recall-heavy: `SOPHON_HYDE=1 SOPHON_FACT_CARDS=1 SOPHON_ENTITY_GRAPH=1`
- Full stack (V032): add `SOPHON_ADAPTIVE=1 SOPHON_LLM_RERANK=1 SOPHON_TAIL_SUMMARY=1 SOPHON_CHUNK_TARGET=500`

---

## When to use it — and when not

**Reach for Sophon when:**
- You're building an MCP agent and want sub-second context compression
- Token cost is a line item in your P&L
- You need reproducibility / determinism (CI, red-team audits, compliance)
- You want a single-binary deploy (~7-34 MB) with zero Python deps on the hot path

**Reach for something else when:**
- You need >80 % long-form conversational recall — run [mem0](https://github.com/mem0ai/mem0),
  [Letta](https://github.com/letta-ai/letta), or [Zep](https://github.com/getzep/zep).
- You need multi-hop reasoning on massive documents — run
  [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) or
  [GraphRAG](https://github.com/microsoft/graphrag).
- You need real OCR / layout analysis on PDFs — use [Docling](https://github.com/docling-project/docling), Marker, or Unstructured.
- You want provider-side cached billing rather than client-side
  compression — use [Anthropic prompt caching](https://docs.anthropic.com/claude/docs/prompt-caching) or OpenAI prompt caching.

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
[Releases](https://github.com/lacausecrypto/mcp-sophon/releases) page
and put `sophon` on your `PATH`.

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

### With v0.4.0 features enabled

```bash
# Recall-heavy agent session with HyDE + fact cards + entity graph
export SOPHON_RETRIEVER_PATH=~/.sophon/retriever
export SOPHON_HYDE=1
export SOPHON_FACT_CARDS=1
export SOPHON_ENTITY_GRAPH=1
sophon serve

# Ingest-time graph memory (experimental — see CHANGELOG)
export SOPHON_GRAPH_MEMORY=1
export SOPHON_GRAPH_MEMORY_PATH=~/.sophon/graph.json
sophon serve
```

---

## Workspace layout

```
.
├── README.md           ← you are here
├── BENCHMARK.md        ← current v0.4.0 numbers, per-section
├── CHANGELOG.md        ← version history + corrections + honest findings
├── LICENSE             ← MIT
├── server.json         ← MCP registry manifest
├── .github/workflows/  ← CI + release automation
├── benchmarks/         ← reproducible scripts for every number
├── npm/                ← npm wrapper package
└── sophon/             ← Rust workspace (11 crates)
    ├── Cargo.toml
    ├── sophon.toml     ← default runtime config
    └── crates/
        ├── sophon-core/          shared types, token/hash helpers
        ├── prompt-compressor/    compress_prompt
        ├── memory-manager/       compress_history, update_memory, graph memory (v0.4.0)
        ├── delta-streamer/       read_file_delta, write_file_delta
        ├── fragment-cache/       encode_fragments, decode_fragments
        ├── semantic-retriever/   chunker + HashEmbedder + BM25 + entity graph (v0.4.0)
        ├── sophon-storage/       SQLite persistence (WAL, embeddings cache)
        ├── output-compressor/    command-aware stdout/stderr compression
        ├── cli-hooks/            transparent command rewriter + agent installer
        ├── codebase-navigator/   tree-sitter/regex + PageRank + digest
        └── mcp-integration/      stdio server, tool schemas, CLI
```

---

## Configuration

Runtime defaults live in [`sophon/sophon.toml`](./sophon/sophon.toml).
See the full [feature flag table](#feature-flags-opt-in) above for
env-var-gated features. Baseline env vars:

- `SOPHON_MEMORY_PATH` — JSONL persistence for session memory
- `SOPHON_RETRIEVER_PATH` — directory for the semantic retriever store
  (enables the `query` parameter on `compress_history`)
- `SOPHON_EMBEDDER` — `hash` (default) or `bge` (needs `--features bge` build)
- `SOPHON_FRAGMENT_MAX_WINDOW` — override the fragment detector window
- `SOPHON_CONFIG` — path to a `sophon.toml` config file

Per-call overrides are available on every MCP tool argument set
(`max_tokens`, `recent_window`, `include_index`, …).

---

## Honest limitations

The full list is in [BENCHMARK.md § 8](./BENCHMARK.md#-8--limitations).
Headlines:

1. **LOCOMO caps at ~40 %.** Mem0 / HippoRAG sit at 80-90 % with
   neural retrieval — we don't match that. We chose determinism.
2. **Multi-hop is hard.** V032 brings 0 → 17 % on LOCOMO multi-hop
   stratified. FULL ceiling is 83 %. The gap is structural.
3. **V032 latency is heavy.** ~42 s p50 on long conversations when
   the full flag stack is on. Pick features a la carte.
4. **HashEmbedder is keyword-bound.** "favorite food" ↔ "weakness
   for ginger snaps" doesn't match without HyDE.
5. **No multimodal ingestion.** Images / PDFs / audio are out of
   scope — run Docling / Marker / Unstructured upstream.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). PRs especially welcome for:

- TypeScript bindings (Python bindings ship in `sophon-py/`)
- TOML-based extractor plugins for new languages (see
  `crates/codebase-navigator/plugins/haskell.toml` for the format)
- More grammars for `navigate_codebase`
- Running the real `mem0` library on LOCOMO to replace the
  `mem0-lite` surrogate in [§ 6.2](./BENCHMARK.md#-62-vs-mem0-lite)
- Multi-seed LOCOMO re-runs to tighten the V032 CI

Run the full test suite with:

```bash
cd sophon && cargo test --workspace                                  # 303 tests
cd sophon && cargo test --features codebase-navigator/tree-sitter    # +15 AST tests
cd sophon && cargo test -p semantic-retriever --features bge -- --ignored  # 5 BGE tests (needs model)
cd sophon-py && .venv/bin/pytest tests/                              # 4 Python tests
```

Every benchmark claim above is reproducible — pointers to the
scripts live in [BENCHMARK.md](./BENCHMARK.md). Open an issue if any
number doesn't reproduce on your machine.

---

## License

MIT. See [LICENSE](./LICENSE).
