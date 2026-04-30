# Sophon

> **Honest token economics for MCP agents.** One Rust binary. Zero ML at query time. Reproducible benchmarks.

[![npm version](https://img.shields.io/npm/v/mcp-sophon.svg?color=blue)](https://www.npmjs.com/package/mcp-sophon)
[![npm total downloads](https://img.shields.io/npm/dt/mcp-sophon.svg)](https://www.npmjs.com/package/mcp-sophon)
[![GitHub release](https://img.shields.io/github/v/release/lacausecrypto/mcp-sophon?sort=semver)](https://github.com/lacausecrypto/mcp-sophon/releases)
[![CI](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml/badge.svg)](https://github.com/lacausecrypto/mcp-sophon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-2025--06--18-purple.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-387%20Rust%20%2B%204%20Python-brightgreen.svg)](./BENCHMARK.md)

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

## TL;DR — v0.5.4

Sophon is a **deterministic context compressor** that slots in front
of whatever memory / cache / code-nav layer you already use — not
instead of them. v0.5.0 was the positioning re-scope (pure
compression, LOCOMO recall is mem0's territory). v0.5.1 → v0.5.4
are perf + observability passes ending with measured async
dispatch + working `notifications/cancelled`.

### 🆕 Real session — measured on this repo's own dev history

We don't pitch from synthetic benches alone. The headline number
below is blended from **four independent real-data benches** that
each measure a different chunk of an agent's tool traffic.
Reproducible byte-for-byte by anyone with `cargo build --release`.

#### Headline (blended across 4 dimensions of real tool traffic)

| Dimension | What it measures | Bench | Saved |
|---|---|---|---|
| **history** | `compress_history` over 38 real commits / 150+ messages | [`real_session_capture.py`](./benchmarks/real_session_capture.py) | **94.6 %** |
| **shell** | `compress_output` on real `cargo`/`gh`/`git`/`ls`/`find` stdout | [`real_session_shell.py`](./benchmarks/real_session_shell.py) | **84.4 %** |
| **filereads** | `compress_prompt` on real source files (Rust / Python / MD / TOML) | [`real_session_filereads.py`](./benchmarks/real_session_filereads.py) | **71.7 %** |
| **search** | `compress_output` on real `grep`/`find`/`ls`/`wc` patterns | [`real_session_search.py`](./benchmarks/real_session_search.py) | **79.5 %** |
| **🎯 Weighted blend** (default mix: 35/30/20/15) | typical agent session estimate | [`real_session_holistic.py`](./benchmarks/real_session_holistic.py) | **84.7 %** |

`real_session_holistic.py` runs all four sub-benches sequentially with `--json` and produces the side-by-side table. Default weights reflect this repo's observed shape; pass `--weights "history=0.4,shell=0.35,filereads=0.15,search=0.10"` to model your own workload.

#### Honest disclosure of where each dimension falls short

- **history** measures only what `git` captures (commit messages + diffs) — about 5-10 % of a real session's *tool traffic* even though it's the biggest single chunk by token count. The 94.6 % is the *upper bound*.
- **shell** mixes commands that compress well (`git diff` 95 %) with commands that don't (`gh repo view --json` *adds* tokens). 84.4 % is the *real-world average*, not a curated highlight.
- **filereads** finds that `compress_prompt` on raw source files compresses by budget cap, not query routing — same input + different query = identical output. Honest finding documented in the bench.
- **search** depends entirely on YOUR repo's state. A repo with no TODOs gets 0 % on `grep TODO`.

The 84.7 % blended estimate is napkin-math from a linear weighted average. It's *not* a cherry-picked synthetic. It's what Sophon's compression machinery actually does to the kind of activity this repo's own dev cycle produces.

#### USD economy on Opus 4.7 (this repo's session)

| | Value |
|---|---|
| **Naive USD saved** (Claude **Opus 4.7** input pricing, $15/MT) | **$2.03 / session** |
| **With prompt caching** (25-turn reads at $1.50/MT) | **$3.24 / session** |

> Opus pricing is ~5× Sonnet. Pass `--model sonnet` or `--model haiku` to `real_session_deep_dive.py` if you're re-pricing for a cheaper tier.

### Orthogonal-stack economics

| Stack | Additional saved by Sophon | Benchmark |
|---|---|---|
| **Sophon + Anthropic prompt caching** | **+24 % tokens / +49 % $** on a 25-turn Claude-3.5-Sonnet session | [`sophon_plus_prompt_caching.py`](./benchmarks/sophon_plus_prompt_caching.py) |
| **Sophon + mem0** | Depends on mem0 output size; the bench flags overhead on short dumps directly | [`sophon_plus_mem0.py`](./benchmarks/sophon_plus_mem0.py) |

### Single-binary efficiency (v0.5.4 release binary, macOS arm64)

| Metric | Value | vs v0.5.0 | Benchmark |
|---|---|---|---|
| **Binary on disk** | **5.2 MB** (release) | -39 % (was 8.7 MB) | `stat` |
| **Cold start → ready** | **34 ms** p50, **37 ms** p99 | unchanged | [`cold_start_and_footprint.py`](./benchmarks/cold_start_and_footprint.py) |
| **RSS after initialize** | **41 MB** (tokenizer pre-warmed) | tokens loaded at boot | idem |
| **Session scaling** (1 → 200 turns) | `update_memory` **0.1 ms** p50 / **0.4 ms** p99; `compress_history` **4 ms** p50 / **47 ms** p99 | -75× on update_memory p99 | [`session_scaling_curve.py`](./benchmarks/session_scaling_curve.py) |
| **`compress_output` synthetic coverage** | **90.1 %** weighted aggregate across 18 command families incl. JSON | +7 pt vs v0.5.2 (JsonStructural strategy) | [`compress_output_per_command.py`](./benchmarks/compress_output_per_command.py) |
| **Async tool dispatch** | `notifications/cancelled` actually drops the response | new in v0.5.4 | [`tests/cancellation_e2e.rs`](./sophon/crates/mcp-integration/tests/cancellation_e2e.rs) |

Pass `--include-python-baseline` to `cold_start_and_footprint.py` to contrast against `python -c "import mem0"` / `sentence_transformers` / `langchain` on your machine.

### Carried over from v0.4.0 (still on-thesis, unchanged)

| Use case | Metric | Compared to |
|---|---|---|
| **Agent session token economics** | **68.1 % tokens saved** across 25-turn coding session ([§ 1](./BENCHMARK.md#-1--session-token-economics)) | Baseline: raw tokens |
| **Prompt compression** | **70.2 % mean saved**, **36 ms mean latency**, 22 prompt shapes ([§ 2](./BENCHMARK.md#-2--prompt-compression-extended)) | LLMLingua-2: +8.9 pt at 35× lower latency ([§ 6.1](./BENCHMARK.md#-61-vs-llmlingua-2)) |
| **Code retrieval (repo QA)** | **recall@3 = 70 %** on "where is X?" questions ([§ 4](./BENCHMARK.md#-4--repo-qa)) | grep: 10 % ; FULL context: 20 % |
| **Latency + reliability** | **p99 < 87 ms** on 5/7 ops, **100 % ok_rate** on 190 runs ([§ 3](./BENCHMARK.md#-3--latency--reliability)) | Sub-second guaranteed |

### Protocol + DX (since v0.5.0)

- **MCP protocol `2025-06-18`** — `notifications/cancelled`
  actually interrupts the response since v0.5.4 (in-flight CPU
  work continues until v0.5.5 cooperative-interrupt landing);
  structured JSON-RPC error codes; infallible dispatcher.
- **`sophon doctor`** — read-only installation diagnostic with
  every `SOPHON_*` flag, path writability, MCP-client hints,
  rolling-summary state, deprecated-flag warnings.
- **Tests** — workspace count 303 (v0.4.0) → **405 (v0.5.4)**.
- **CI** — 4 platform matrix + nightly bench-vs-baseline +
  per-PR regression alarm on `sophon/crates/**`.

### What stopped being a goal

Long-form conversational recall above ~40 % on LOCOMO is now
explicitly out of scope. mem0 hits 91 % with neural retrieval; we
**sit in front of mem0 instead**. The v0.4.0 recall-chasing flags
(`SOPHON_HYDE`, `SOPHON_FACT_CARDS`, `SOPHON_ENTITY_GRAPH`,
`SOPHON_LLM_RERANK`, `SOPHON_ADAPTIVE`, `SOPHON_TAIL_SUMMARY`,
`SOPHON_REACT`, `SOPHON_GRAPH_MEMORY`, `SOPHON_MULTIHOP_LLM`) stay
functional but are flagged by `sophon doctor` as deprecated and
will be removed.

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

## Feature flags

Run `sophon doctor` to see every `SOPHON_*` env var currently set,
with validation warnings and a note for deprecated recall-chasing
flags. The full catalogue (24 flags, grouped by scope) lives in
[`runtime_flags.rs`](./sophon/crates/mcp-integration/src/runtime_flags.rs).

**On-thesis, still recommended:**

| Flag | What it adds | Cost |
|---|---|---|
| `SOPHON_RETRIEVER_PATH=/dir` | Activate the semantic retriever (chunk store on disk). | ~0 |
| `SOPHON_MEMORY_PATH=/file.jsonl` | Persistent conversation memory across `sophon serve` runs. | ~0 |
| `SOPHON_HYBRID=1` | BM25 sparse-lexical + HashEmbedder fused via RRF. | ~1 ms |
| `SOPHON_CHUNK_TARGET=500` | Bigger chunks preserve cross-sentence context. | ~0 |
| `SOPHON_EMBEDDER=bge` | Swap HashEmbedder for BGE-small (needs `--features bge`). | +model load at startup |
| `SOPHON_NO_LLM_SUMMARY=1` | Opt-out from Haiku summary; heuristic only. | Speed (bench utility) |
| `SOPHON_DEBUG_LLM=1` | Richer tracing warnings for LLM subprocess failures. | — |

**Deprecated (v0.4.0 recall-chasing experiments, scheduled for removal):**

`SOPHON_HYDE`, `SOPHON_FACT_CARDS`, `SOPHON_ENTITY_GRAPH`,
`SOPHON_ADAPTIVE`, `SOPHON_LLM_RERANK`, `SOPHON_TAIL_SUMMARY`,
`SOPHON_REACT`, `SOPHON_GRAPH_MEMORY`, `SOPHON_MULTIHOP_LLM` —
these chase LOCOMO recall, an axis we no longer optimise. Still
functional but `sophon doctor` flags them. See
[CHANGELOG.md § 0.5.0 Positioning re-scope](./CHANGELOG.md#050).
If you need neural recall, pipe mem0 / Letta in front of Sophon
instead (see [When to use](#when-to-use-it--sophon-in-front-of-x)
below).

---

## When to use it — Sophon in front of X

Sophon is **not** a memory platform, a recall system, an OCR stack,
or a replacement for provider-side caching. It's a deterministic
context compressor that slots **in front of** whatever memory /
cache / code-nav layer you already use, and attacks the tokens those
layers can't.

The v0.5.0 positioning is explicit: Sophon stops chasing LOCOMO
recall (mem0's territory) and doubles down on pure compression —
tokens saved %, latency p99, binary size, canary preservation, MCP
compliance. See [CHANGELOG.md](./CHANGELOG.md#050) for the re-scope
note.

### Sophon in front of Anthropic / OpenAI prompt caching

Provider caching handles the **static** half of a request — system
prompt, tool definitions, reused documents. It doesn't touch the
dynamic half (growing conversation history, tool outputs). Sophon
compresses exactly that half. The two stack cleanly.

> Reproducible measurement:
> [`benchmarks/sophon_plus_prompt_caching.py`](./benchmarks/sophon_plus_prompt_caching.py)
> simulates a 25-turn agent session with a 6600-token cacheable
> static block and claude-3.5-sonnet pricing. Sophon saves an
> **additional 23.8 % tokens / ~49 % $** on top of prompt caching —
> because the uncached dynamic block is billed at 10× the cached
> rate, so every dynamic-token Sophon removes is worth ~10 cached
> tokens in dollars.

### Sophon in front of mem0 / Letta / Zep / Graphiti

mem0 and friends retrieve the right memories. Sophon shrinks what
gets sent to the LLM **after** retrieval. If mem0 returns 2 kB of
raw memories, `compress_prompt` keeps only the sections the query
actually references.

> Reproducible measurement:
> [`benchmarks/sophon_plus_mem0.py`](./benchmarks/sophon_plus_mem0.py)
> runs against a surrogate mem0 retriever by default
> (no API keys needed) or the real `mem0ai` package with
> `--real-mem0`. It reports Sophon's **additional** savings + the
> proper-noun / date / number preservation rate. Honest caveat
> built-in: on very short mem0 outputs (< ~200 tokens) Sophon adds
> overhead from its own wrapper — only pipe larger dumps through it.

### Sophon in front of Claude Code / Cursor / Cline

This is the primary use case. Every repeat file read becomes a
`read_file_delta`; every shell command output goes through
`compress_output`; every repeated boilerplate block gets swapped for
a `fragment_cache` token. A 25-turn session drops from ~15 k
tokens/turn to ~9 k tokens/turn.

> Reproducible measurement:
> [`benchmarks/session_token_economics.py`](./benchmarks/session_token_economics.py)
> — **68.1 %** session tokens saved
> ([§ 1](./BENCHMARK.md#-1--session-token-economics)).
> Install with `sophon hook install --agent claude --global`.

### Sophon in front of a RAG pipeline

`navigate_codebase` produces a PageRanked repo digest that a RAG
retriever would otherwise spend expensive embedding calls to build.
Sophon emits it deterministically, with tree-sitter / regex symbol
extraction over 11 languages, in under a second.

---

### When NOT to pipe Sophon in front of something

- **Long-form conversational recall above 80 %** — Sophon caps at
  ~40 % LOCOMO and we don't chase it. Run
  [mem0](https://github.com/mem0ai/mem0) /
  [Letta](https://github.com/letta-ai/letta) /
  [Zep](https://github.com/getzep/zep) for recall, then optionally
  pipe their output through Sophon (see above).
- **Multi-hop reasoning on massive documents** — that's
  [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) or
  [GraphRAG](https://github.com/microsoft/graphrag)'s job.
- **OCR / PDF layout analysis** — out of scope. Use
  [Docling](https://github.com/docling-project/docling), Marker, or
  Unstructured upstream of Sophon.
- **Very small inputs (< ~200 tokens)** — Sophon's XML-tagged
  section scaffolding can cost more than it saves. Pass through raw.

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

### Typical v0.5.0 setup

```bash
# Default: zero-ML compression with BM25+Hash hybrid retrieval on
# (on-thesis, deterministic, sub-ms overhead).
export SOPHON_RETRIEVER_PATH=~/.sophon/retriever
export SOPHON_HYBRID=1
export SOPHON_MEMORY_PATH=~/.sophon/memory.jsonl
sophon serve

# Diagnose your install before wiring it into an MCP client
sophon doctor
```

The v0.4.0 recall-chasing flags (`SOPHON_HYDE`,
`SOPHON_FACT_CARDS`, `SOPHON_ENTITY_GRAPH`, `SOPHON_GRAPH_MEMORY`,
…) still parse but `sophon doctor` flags them as deprecated — see
[CHANGELOG § 0.5.0](./CHANGELOG.md#050--unreleased).

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
