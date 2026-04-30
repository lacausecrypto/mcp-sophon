# Changelog

All user-visible changes to Sophon live here. Benchmark numbers use the
same `claude -p cl100k_base` stack as the published tables unless stated
otherwise. Historical corrections are kept inline — we don't rewrite
numbers that turned out to be sampling luck, we flag them.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
with additional **Measured** / **Honest findings** sections for bench
results and failed experiments.

## [0.5.3] — 2026-04-30

Tier 1 follow-up to v0.5.2. Three structural wins ship together:
generic JSON compressor, tool-call dedup foundation, CI nightly
regression alarm.

### Feat — `CompressionStrategy::JsonStructural`

`compress_output` gains a content-aware structural compressor:
input that parses as JSON gets array-truncated and string-clipped
**recursively** before line-level passes run. Single-byte sniff
on `input.trim_start()` short-circuits non-JSON so the strategy
is essentially free for plain text.

Wired into 3 filters where JSON output is common:
* `generic_filter` (catches `aws --output json`, `gh api`, etc.)
* `curl_json_filter`
* `kubectl_get_filter`

Bug fix: when `JsonStructural` ran successfully, the legacy
`budget_cap` middle-truncate would *destroy* the structure
(reducing 5 carefully-preserved pods to `{` plus an omission
marker because preserve_head was tuned for tabular output). Fixed
by skipping `budget_cap` when `json_structural` is in
`strategies_applied` — JsonStructural's array-length cap is
already a hard upper bound.

Bench `compress_output_per_command.py` extended with 3
JSON-heavy samples (`kubectl get pods -o json` 50-pod blob,
`aws s3api list-objects-v2` 80 objects, `gh api ...issues`
40 issues). Aggregate moves:

                          v0.5.2     v0.5.3
  weighted aggregate      83.1 %     **90.1 %**  (+7 pt)
  mean per-command        56.0 %     **62.4 %**  (+6.4 pt)
  median per-command      51.8 %     **65.8 %**  (+14 pt)

Per-command on the new samples:
  kubectl get pods -o json     12892 → 1332   89.7 %
  aws s3api list-objects        8612 →  238   97.2 %
  gh api …/issues               8522 →  309   96.4 %

### Feat — tool-call dedup foundation (chunker level)

Two new `ChunkType` variants:

```rust
ChunkType::ToolUse        // tool invocation
ChunkType::ToolResult     // result of a tool call
```

JSONL store stays backwards-compatible (serde
`rename_all = "snake_case"` + new variants opt-in only when
emitted).

Detector functions in `chunker.rs` recognise:
* Anthropic JSON `{"type":"tool_use","name":"X","input":{…}}`
* Flattened transcript form `Tool call: X({"k":"v"})`
* `{"type":"tool_result","tool_use_id":"…","content":"…"}`
  (string or text-block array)

Critical detail: argument JSON gets normalised via a `BTreeMap`
detour so `{a:1,b:2}` and `{b:2,a:1}` produce byte-identical
canonical strings. Two transcripts that emit the same call with
different key orders no longer split a single semantic call
across multiple chunks. Tested via
`detect_tool_use_normalises_arg_order`.

Five identical `read_file(path=X)` calls in one session now
collapse to one chunk via the existing `chunk_id` (sha-of-content)
mechanism.

Honest scope: this is the **foundation**. The actual size of the
dedup win depends on real Claude Code session shape (how often
tools are called with identical args). A measured number is
deferred to v0.5.4 once we capture real session traces with
`tee_enabled`. What's mechanically guaranteed today: identical
invocations produce one chunk, distinct args produce distinct
chunks, arg-order variations don't split.

### CI — bench nightly + baseline regression alarm

New `benchmarks/run_all_with_baseline.py` driver runs the full
bench suite, extracts key metrics, and compares against
`benchmarks/baseline.json` (now committed in git).

Per-metric tolerance + direction (regressions = drift in the bad
direction past tolerance):

  binary_size_bytes               5 %  min  (smaller = better)
  cold_start_ms_p50              25 %  min
  cold_start_ms_p99              30 %  min
  rss_idle_mb                    25 %  min
  aggregate_saved_pct             2 %  max  (higher = better)
  mean_saved_pct                  5 %  max
  compress_history_p50_ms        30 %  min
  compress_history_p99_ms        50 %  min
  update_memory_p99_ms           50 %  min

Compression ratios are deterministic per binary (tight 2 %
tolerance); latency carries CI scheduler jitter so it gets
generous bands to avoid flake-fail. Improvements > 5 % are
flagged as "wins worth refreshing the baseline" but don't fail
the run.

`.github/workflows/bench-nightly.yml` triggers:
* `schedule: '0 3 * * *'` — daily catch on main
* `pull_request` on `sophon/crates/**`, `benchmarks/**`,
  `.github/workflows/bench-nightly.yml` — reviewers see drift
  inline on perf-touching PRs
* `workflow_dispatch` with `update_baseline=true` toggle that
  commits the refreshed baseline back to the branch

Concurrency group cancels stale runs on the same branch so
fast iteration cycles don't queue up bench wall-clock.

### Tests

* output-compressor: 49 → **55** unit tests (+6 JsonStructural)
* semantic-retriever: 59 → **68** unit tests (+9 tool-call)
* Workspace total: 392 → **401**

## [0.5.2] — 2026-04-30

Phase-2 of the post-v0.5.0 perf push. Two structural wins —
release-binary size cut and ingest-time rolling summary —
landed; tool-call dedup and streaming `compress_output` were
intentionally deferred to v0.5.3 because they need real Claude-
Code-session data to bench honestly, and we'd rather defer than
ship un-measured optimisations.

### Perf — release binary size

Stock cargo release profile (`opt-level=3`, `lto=false`,
`codegen-units=16`, no strip, `panic="unwind"`) replaced with an
empirically-tuned set:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

Result on macOS arm64:
* **Binary size: 8.6 MB → 5.2 MB (-39 %)**
* Cold start p50: **34 ms** — unchanged vs v0.5.1
* `compress_output` aggregate: **83.1 %** — unchanged
* Workspace tests: 392/392 green under `cargo test --release`

`opt-level = "z"` and `"s"` were both tried first and rejected:
each regressed the tiktoken-rs BPE merge loop measurably (cold
start went to 81 ms and 43 ms respectively). LTO + strip +
panic-abort do enough size work on their own without touching
the inner loops.

`panic = "abort"` is safe in this workspace — verified via grep:
no `catch_unwind` calls, MCP dispatcher already treats panics
as unrecoverable. Aborting on panic saves ~700 KB of unwind
tables and avoids the half-unwound-stack hazard mid-request.

### Feat — `SOPHON_ROLLING_SUMMARY` rolling summary at ingest

When `SOPHON_ROLLING_SUMMARY=1`, `MemoryManager::append`
incrementally maintains a `RollingSummary` covering the older
slice of conversation. `compress_history` then serves the
cached state instead of re-summarising the full history per
query. With `SOPHON_LLM_CMD` configured this moves the 5-8 s
LLM spike off the query path entirely (one Haiku call per ~50
new messages instead of per query).

New env vars:
* `SOPHON_ROLLING_SUMMARY=1` — feature toggle (default off, so
  callers without the env var see byte-identical behaviour to
  v0.5.1).
* `SOPHON_ROLLING_THRESHOLD=N` — refresh cadence (default 50;
  sanity floor 10).

Persistence: a JSON sidecar at `<memory_path>.sophon-summary.json`
keeps the rolling state across `sophon serve` restarts. Sidecars
ahead of the live history (e.g. JSONL truncated externally) are
discarded on reopen.

LLM activation matches the existing `compress_history` policy:
implicit when `SOPHON_LLM_CMD` is set, explicit via
`config.use_llm_summarization`, opt-out via
`SOPHON_NO_LLM_SUMMARY=1`. On LLM failure, ingest falls back to
the heuristic summariser so a transient subprocess error never
blocks `update_memory`.

`sophon doctor` now surfaces both flags under a new
`FlagScope::Memory` group.

### Bench — `rolling_summary_effect.py`

New benchmark harness measures the v0.5.2 feature's impact
honestly:

```
run                                  compress p50  compress p99   update p99
baseline (no rolling)                       4.13ms       50.61ms      0.16ms
rolling enabled (heuristic)                 4.19ms       48.35ms      0.22ms
```

The heuristic-mode delta is essentially zero — the heuristic
summariser was already fast (sub-ms on 200 messages), so caching
its output doesn't move the needle. **The pitch lives in the
LLM-mode case** (5-8 s spike → ~5 ms cached read), measurable by
passing `--with-llm` if `claude -p --model haiku` is on `$PATH`.

### Tests + counts

* `summarizer::rolling_tests` — 5 new unit tests (threshold
  gating, recent-floor, idempotence, stale state, no-rolling
  parity).
* `tests/integration_test.rs` — 5 new integration tests
  (default-off contract, threshold firing, snapshot path,
  sidecar persistence + reload, reset cleanup). Env-touching
  tests serialised via a `Mutex` guard to avoid races under
  cargo's parallel runner.
* Workspace test count: **387 → 392**.

### Deferred to v0.5.3

* **Tool-call dedup at chunker level** (phase 2C) — needs real
  Claude Code session traces to design the dedup key correctly
  and to bench against. Synthetic data would let us guess at
  the win without measuring it.
* **Streaming `compress_output`** (phase 2D) — MCP 2025-06-18
  doesn't support streamed responses naturally; would need a
  CLI-level `sophon exec --stream` mode plus a measured
  workload (`tail -f` / `cargo build --watch`) to defend the
  feature. Both pieces deferred until phase-3 async work
  lands.

## [0.5.1] — 2026-04-20

Phase-1 follow-up to v0.5.0. Three targeted perf wins + two
compression-ratio wins, all aligned with the v0.5.0 orthogonal
positioning (no new LLM dependencies, no behaviour change for
clients that don't opt in).

### Perf — tracing, warm-up, embedding reuse

- **`tracing::instrument` on the 5 hot-path entry points**:
  `memory_manager::compress_history`,
  `prompt_compressor::compress_prompt`,
  `output_compressor::OutputCompressor::compress` (plus a
  `debug!` with filter + strategies + ratio),
  `semantic_retriever::Retriever::retrieve`, and
  `Retriever::index_messages`. Zero runtime cost with
  `RUST_LOG=off`; scriptable timelines with
  `RUST_LOG=sophon=debug`. Added `tracing` dep to
  `prompt-compressor` and `semantic-retriever` (the two crates
  that didn't have it yet).

- **Eager tokenizer warm-up at `sophon` startup**. One
  `count_tokens("warmup")` call right after `init_tracing()`
  loads tiktoken-rs BPE merge tables upfront instead of lazily
  on the first real call. Measured on
  `benchmarks/session_scaling_curve.py` at 200 turns:
    * `update_memory` p99: **30.5 ms → 0.4 ms** (~75×)
    * RSS idle:           **12.5 MB → 41.6 MB** (+29 MB now paid
      at boot instead of spiking the first request)
    * RSS after 100 calls: 42.0 MB → 41.8 MB (steady-state
      delta collapses from +30 MB to +0.2 MB)

- **JSONL store embedding reuse**. `Retriever::open` used to
  re-embed every stored chunk on load; at 100k chunks that's
  multi-second on the wrong side of the MCP handshake. The
  `Chunk::embedding: Option<Vec<f32>>` field was already
  populated by `index_messages` and serialised by the JSONL
  store — we just weren't consuming the persisted vector on
  reopen. Now we reuse it when the stored dimension matches
  the current embedder, and re-embed only as a fallback (empty
  embedding field or dimension mismatch after
  `SOPHON_EMBEDDER=hash→bge` swap). Guarded by a new
  integration test that asserts top-chunk score is
  bit-identical across close/reopen.

### Compression — two new filter wins

Driven directly by the v0.5.0
`benchmarks/compress_output_per_command.py` anomalies at 0.5 %
(curl -v) and 11.8 % (git diff).

- **New `curl_verbose` filter**. `curl -v` / `--verbose` /
  `--trace*` output is ~60 % TLS handshake + cert-chain noise.
  The existing `curl_json` filter kept `^\*` lines as signal
  (correct for `-i` headers, wrong for `-v`). Dedicated filter
  drops `^\*` noise while keeping `> GET ...`, `< HTTP/...`,
  response headers, body, and the outcome marker. Registered
  ahead of `curl_json`; plain `curl URL` still routes to
  `curl_json`.
    * **`curl -v` compression: 0.5 % → 60.3 % saved (+59.8 pt)**

- **`git_diff_filter` dedup strategy**. Mechanical refactors
  (rename applied across N call sites) produce diffs with
  identical `-old` / `+new` hunk pairs. The pipeline now runs
  `CompressionStrategy::Deduplicate` with
  `similarity_threshold=1.0` (exact match only; no fuzzy
  merge) between line-filter and truncate. Collapses runs of
  identical consecutive lines to `<line>  // (× N)`.
    * **`git diff HEAD~3` compression: 11.8 % → 51.8 % saved (+40.0 pt)**

### Aggregate impact

Rerun of `benchmarks/compress_output_per_command.py` across the
same 15 canned samples:

                         before (v0.5.0)   after (v0.5.1)
  weighted aggregate     81.6 %            **83.1 %**
  mean per-command       49.4 %            **56.0 %**
  median per-command     48.4 %            **51.8 %**

Rerun of `benchmarks/session_scaling_curve.py` at 200 turns:

                              before (v0.5.0)   after (v0.5.1)
  update_memory p99             30.5 ms          **0.4 ms**
  compress_history p99          50.4 ms          49.2 ms
  compress_history p50           4.2 ms          4.0 ms

### Internal — tests + CI

- 1 new integration test:
  `e2e_cached_embeddings_survive_reopen_and_match_freshly_computed`
  guards the embedding reuse round-trip.
- 4 new unit tests around the `curl_verbose` filter + its
  dispatcher precedence over `curl_json`.
- Registry filter count: 20 → 21.
- Workspace test count: 387 → 392.

## [0.5.0] — 2026-04-20

### Positioning re-scope: orthogonal compression only

Sophon stops chasing LOCOMO conversational recall and re-anchors on
pure context compression. Product thesis going forward:

> Sophon is a deterministic compressor that slots **in front of**
> whatever memory / cache / code-nav layer you already use — not a
> replacement for mem0, Letta, Zep, or Anthropic prompt caching.

Metrics we optimise:
- tokens saved %, latency p50/p99, binary size, canary preservation,
  MCP compliance.

Metrics we no longer chase:
- LOCOMO absolute accuracy, head-to-head neural recall benchmarks
  against mem0 / HippoRAG.

The v0.4.0 recall-chasing flags (`SOPHON_HYDE`, `SOPHON_FACT_CARDS`,
`SOPHON_ENTITY_GRAPH`, `SOPHON_ADAPTIVE`, `SOPHON_LLM_RERANK`,
`SOPHON_TAIL_SUMMARY`, `SOPHON_REACT`, `SOPHON_GRAPH_MEMORY`,
`SOPHON_MULTIHOP_LLM`) remain functional but are now **deprecated**
and flagged by `sophon doctor`. They will be removed in a future
release.

### Added — MCP protocol 2025-06-18

- `initialize` advertises `protocolVersion: "2025-06-18"`, with
  `capabilities.tools.listChanged` and `capabilities.logging` set.
- `notifications/cancelled` is acknowledged and logged at DEBUG
  (tool dispatch stays synchronous on the stdio loop; an async
  rework is the next step).
- JSON-RPC dispatch is now infallible — every error path produces a
  spec-shaped error response instead of killing the stdio loop.
- New stable Sophon server-error codes in the reserved
  `-32000..-32099` range: `SOPHON_TOOL_NOT_FOUND (-32001)`,
  `SOPHON_TOOL_ARGUMENTS_INVALID (-32002)`,
  `SOPHON_TOOL_EXECUTION_FAILED (-32003)`,
  `SOPHON_RETRIEVER_UNAVAILABLE (-32004)`,
  `SOPHON_IO_ERROR (-32005)`,
  `SOPHON_CONFIG_ERROR (-32006)`.
- Tool-level errors keep MCP's `isError: true` result shape but now
  include `structuredContent.error.code` so clients can branch on
  the code without regex-matching English.

### Added — observability (tracing)

- `tracing 0.1` + `tracing-subscriber 0.3` workspace dependencies.
- `sophon serve` installs a stderr subscriber with
  `RUST_LOG`-controlled filter; defaults to `info`.
- 18 production `eprintln!` calls replaced with `tracing::warn! /
  info! / debug!` (memory-manager/llm_client, server.rs, main.rs,
  output-compressor, codebase-navigator/extractors).

### Added — `sophon doctor` subcommand

Read-only installation diagnostic. Prints: binary metadata, resolved
config source + parse status, every `SOPHON_*` env var in use
(sanitised), path existence + writability checks for the persistence
locations, LLM-command PATH probe, MCP client-config hints per
agent. Flags deprecated recall-chasing env vars so users see the
v0.5.0 trajectory.

### Added — centralised `SOPHON_*` flag documentation

`mcp-integration/src/runtime_flags.rs` enumerates every env var
Sophon reads (24 flags, grouped by scope). Each entry carries name,
kind (Bool / Path / String / UInt{min,max}), scope, and one-line
description. Called at `sophon serve` startup: invalid values emit
`tracing::warn` before the MCP handshake. New env vars MUST land
here so they show up in `sophon doctor` and validation.

### Added — orthogonal-stack benchmarks

- `benchmarks/sophon_plus_prompt_caching.py` — measures additional
  tokens / dollars Sophon saves on top of Anthropic prompt caching.
  A 25-turn synthetic session at claude-3.5-sonnet pricing shows
  **~24 % additional tokens / ~49 % additional $** saved. Includes a
  live empirical probe against `compress_history` and
  `compress_output` so the numbers aren't purely theoretical.
- `benchmarks/sophon_plus_mem0.py` — Sophon's additional savings on
  top of mem0 retrieval output. Defaults to a surrogate keyword-
  gated retriever so the bench runs without API keys; pass
  `--real-mem0` after `pip install mem0ai`. Honest caveat: on short
  mem0 outputs (< ~200 tokens) Sophon adds overhead; the bench
  reports this directly.

### Added — single-binary efficiency benchmarks

Three new scripts that quantify the "one Rust binary, zero ML"
positioning claim with numbers a Python-based stack can't match.
No API keys, no network.

- `benchmarks/cold_start_and_footprint.py` — spawn-to-ready time,
  resident-set size at idle and after 100 tokenizer calls, release
  binary size on disk. Pass `--include-python-baseline` to contrast
  with `python -c "import mem0"`, `sentence_transformers`,
  `langchain`. On macOS arm64 the v0.5.0 release binary measured:
  **8.7 MB** on disk, **10.6 ms p50 / 25 ms p99** cold start,
  **12.5 MB** RSS after initialize, +30 MB after a 100-call steady
  state.
- `benchmarks/compress_output_per_command.py` — per-command coverage
  of `compress_output` across 15 realistic shell samples (git,
  cargo, docker, pytest, npm install, kubectl get, curl -v, tail,
  grep, ls, make, …). Reports weighted aggregate + mean + median
  saved %, plus the filter and strategies Sophon routed each
  sample through. Current run: **81.6 % weighted aggregate**, best
  performers `tail` / `docker logs` / `ls` (90-99 %), worst
  `git log` / `curl json` (already terse). Guides users on which
  workflows benefit most from piping through Sophon.
- `benchmarks/session_scaling_curve.py` — drives one long-lived
  `sophon serve` from turn 1 → N and samples `update_memory_ms`,
  `compress_history_ms`, token_count, RSS at each checkpoint. At
  200 turns: update-memory stays **0.1 ms p50** (flat),
  compress-history **4.2 ms p50 / 50 ms p99**, RSS growth
  **+30 MB** (linear-ish in the JSONL store, bounded by token
  budget). Catches any future regression that would make Sophon
  O(turns²) or unbound RSS — pure sanity gate.

### Changed — archive deprecated benchmarks

`benchmarks/_deprecated/` holds four LOCOMO-adjacent scripts that
targeted neural-recall parity with mem0 / HippoRAG: the new
positioning makes them misleading. README in the archive directory
documents the rationale.

### Internal — tests + cleanup

- 5 new `mcp-integration` integration tests guard the 2025-06-18
  handshake, `notifications/cancelled`, structured error codes, and
  `sophon doctor` output.
- 6 new `semantic-retriever` end-to-end integration tests covering
  the chunker → embedder → index → store → retrieve contract.
- 6 new `codebase-navigator` end-to-end tests covering the polyglot
  regex extraction path, PageRank query biasing, digest budget
  enforcement, and TOML plugin loader.
- Workspace test count: 303 (v0.4.0) → 405+.

## [0.4.0] — 2026-04-18

### TL;DR (archived from the v0.4.0 README)

Sophon was shaped around four use cases. Measured numbers:

| Use case | Metric | Compared to |
|---|---|---|
| **Agent session token economics** | **68.1 % tokens saved** across 25-turn coding session ([§ 1](./BENCHMARK.md#-1--session-token-economics)) | Baseline: raw tokens |
| **Prompt compression** | **70.2 % mean saved**, **36 ms mean latency**, 22 prompt shapes ([§ 2](./BENCHMARK.md#-2--prompt-compression-extended)) | LLMLingua-2: +8.9 pt at 35× lower latency ([§ 6.1](./BENCHMARK.md#-61-vs-llmlingua-2)) |
| **Code retrieval (repo QA)** | **recall@3 = 70 %** on "where is X?" questions ([§ 4](./BENCHMARK.md#-4--repo-qa)) | grep: 10 % ; FULL context: 20 % |
| **Latency + reliability** | **p99 < 87 ms** on 5/7 ops, **100 % ok_rate** on 190 runs ([§ 3](./BENCHMARK.md#-3--latency--reliability)) | Sub-second guaranteed |

**LOCOMO note (conversational memory benchmark):** V032 full stack
lands at **40 %** (N=30), up from V030's 33 %. Not competitive with
mem0 (91 %, neural). Honest about that in
[§ 5](./BENCHMARK.md#-5--locomo-honest). **v0.5.0 made this a
deliberate non-goal** — see [§ 0.5.0 positioning re-scope](#050--unreleased).

### What changed in v0.4.0

Twelve new opt-in flags. Defaults unchanged from 0.3.0 — a caller
that only sets `SOPHON_RETRIEVER_PATH` gets v0.3.0 behaviour byte
for byte.

Biggest wins under the hood:
- **Rayon-parallel block summarisation** → `compress_history` with LLM summary dropped from ~40 s to ~5-8 s on 600-turn conversations
- **HyDE + FactCards + EntityGraph stack** → canary preservation on `compress_history` went from 27 % → 80 %
- **Path A graph memory** (experimental) → ingest-time LLM triple extraction, zero LLM at query time

### Added — optional feature flags (defaults unchanged from 0.3.0)

All new behaviour is gated on env vars. Existing `compress_history`
callers see byte-identical output when no flag is set.

| Flag | What it does | Gain measured |
|---|---|---|
| `SOPHON_HYDE=1` | Haiku writes 2-3 hypothetical answers, each used as a retrieval query, results fused via RRF | +17 pt open-domain, +17 pt single-hop on LOCOMO N=30 |
| `SOPHON_FACT_CARDS=1` | Haiku extracts `{entity: [events]}` JSON timeline, rendered into context | +17 pt temporal on LOCOMO N=30 |
| `SOPHON_ENTITY_GRAPH=1` | Heuristic NER + bipartite entity↔chunk index, 1-hop bridge | +17 pt multi-hop on LOCOMO N=30 (Rec #2) |
| `SOPHON_ADAPTIVE=1` | Haiku classifies query as `factual_recall` vs `general`, bumps top_k + budget for recall-heavy | +5-10 pt on factual |
| `SOPHON_LLM_RERANK=1` | One Haiku call rescores top-(3×k) candidates, re-sorts before token budget | +8 pt recall (measured in isolation) |
| `SOPHON_TAIL_SUMMARY=1` | Haiku summarises chunks K+1..K+10 into one paragraph appended to context | +3-5 pt |
| `SOPHON_CHUNK_TARGET=500` | Chunker target size (default 128) — bigger chunks preserve cross-sentence context | +5 pt when paired with rerank |
| `SOPHON_HYBRID=1` | BM25 sparse-lexical ranking fused with HashEmbedder via RRF | +3-5 pt on rare-term queries |
| `SOPHON_MULTIHOP_LLM=1` | Haiku decomposes query into 2-3 sub-questions | 0 pt net (see Honest findings) |
| `SOPHON_REACT=1` | Iterative retrieval loop, Haiku decides follow-up queries | **experimental** — net loss when combined with HyDE |
| `SOPHON_GRAPH_MEMORY=1` | Ingest-time LLM triple extraction → pure-Rust graph query at retrieval time | **experimental** — see Path A notes |
| `SOPHON_GRAPH_MEMORY_PATH=/path` | JSON-file persistence for the graph store | — |
| `SOPHON_NO_LLM_SUMMARY=1` | Opt-out from block-based Haiku summary (bench utility) | — |
| `SOPHON_DEBUG_LLM=1` | Log every LLM call failure/empty/parse-error to stderr | — |

### Added — new modules (Rust, workspace test count 215 → 303)

- `semantic-retriever/src/bm25.rs` — Okapi BM25 with insert + search (7 tests)
- `semantic-retriever/src/fusion.rs` — Reciprocal Rank Fusion (6 tests)
- `semantic-retriever/src/entity_graph.rs` — bipartite entity↔chunk store with IDF-weighted scoring and bounded 1-hop bridge (10 tests)
- `memory-manager/src/llm_client.rs` — shared Haiku shell-out helper with failure diagnostics
- `memory-manager/src/llm_reranker.rs` — batched rerank call (9 tests)
- `memory-manager/src/multihop.rs` — heuristic multi-hop detector (8 tests)
- `memory-manager/src/query_decomposer.rs` — LLM sub-query decomposition (7 tests)
- `memory-manager/src/query_rewriter.rs` — HyDE rewrites (6 tests)
- `memory-manager/src/question_classifier.rs` — adaptive query classifier (9 tests)
- `memory-manager/src/react.rs` — ReAct decision parser (10 tests)
- `memory-manager/src/tail_summary.rs` — tail-chunk compressor (8 tests)
- `memory-manager/src/fact_cards.rs` — entity timeline extractor (8 tests)
- `memory-manager/src/graph/` — full Path A implementation, 57 tests
  - `types.rs` — EntityId, Predicate, Fact, FactObject, FactId (16 tests)
  - `store.rs` — HashMap-backed graph with JSON persistence (11 tests)
  - `extract.rs` — one-call triple extraction (11 tests)
  - `ingest.rs` — rayon-parallel batched ingest (6 tests)
  - `query.rs` — query → entities → facts ranking (13 tests)

### Added — infrastructure

- **Rayon-parallel block summaries** in `summarizer.rs` — ~5× wall-clock speedup on 600-turn conversations (40 s → 5-8 s)
- **Rayon-parallel graph ingest** — single `update_memory` fans N batches across CPU cores
- **`benchmarks/llm_cli.py`** — universal CLI wrapper normalising `claude` and `codex` stdin/stdout for bench harnesses

### Added — benchmark scripts

| Script | What it measures |
|---|---|
| `benchmarks/session_token_economics.py` | 25-turn agent coding session, per-op + aggregate token savings |
| `benchmarks/prompt_compression_extended.py` | `compress_prompt` on 22 prompt shapes, savings + latency distribution |
| `benchmarks/latency_reliability.py` | p50/p95/p99/max + ok_rate + preserved_rate on 7 ops |
| `benchmarks/repo_qa.py` | `navigate_codebase` vs grep vs FULL on 20 "where is X?" queries |
| `benchmarks/locomo_v032_ab.py` | V030 / V031 / V032_FULL / FULL on LOCOMO stratified |
| `benchmarks/locomo_pathA_ab.py` | Graph memory (Path A) vs V030 / V032 / FULL |
| `benchmarks/locomo_cross_model.py` | Stack A (Sonnet) vs Stack B (codex) with configurable answerer |
| `benchmarks/audit_rate_limit.py` | `SOPHON_DEBUG_LLM=1` fail counter across 10 parallel items |
| `benchmarks/audit_adversarial.py` | Adversarial-only audit to validate regression claims |
| `benchmarks/audit_v030_baseline.py` | V030 reproducibility check vs the 40 % public claim |

### Measured — v0.4.0 headline numbers

| Bench | Result |
|---|---|
| Session token economics (N=25) | **68.1 % tokens saved** on a realistic coding session |
| Prompt compression extended (N=22) | **70.2 % mean saved**, 36 ms mean latency |
| Latency + reliability (N=30 light, N=10 heavy) | **p99 < 87 ms** on 5/7 ops, **100 % ok_rate** across 190 runs, 87.4 % canary preserved |
| Repo QA (N=20) | `navigate_codebase` **recall@3 = 70 %** (vs grep 10 %, FULL 20 %) with 9.5 K tokens |
| LOCOMO V030 baseline (N=30) | 33.3 % accuracy — at the low end of the published 25-40 % band |
| LOCOMO V032_FULL (N=30) | **40.0 % accuracy**, +6.7 pt over V030 baseline |
| LOCOMO V032 per-type | +17 pt single-hop, +17 pt multi-hop, +17 pt temporal, +33 pt open-domain, tied adversarial |

### Cross-model validation

Ran `locomo_cross_model.py` with Stack A = Sonnet answerer, Stack B =
`gpt-5.3-codex` (Codex CLI via `benchmarks/llm_cli.py --provider codex`).
On 6 items completed before we stopped:

- V030 @ Sonnet : 50 % ▪ V030 @ codex : 17 %
- V032 @ Sonnet : 67 % ▪ V032 @ codex : 50 %

Sophon gains **amplify** on the weaker baseline — codex is more
conservative by default, V032 closes most of the gap. Confirms the
v0.3.0 claim that Sophon's architecture is provider-agnostic.

### Honest findings (things that did not work)

1. **Path A (graph memory)** — predicted 75-85 % LOCOMO, measured 20 % alone and 40 % combined with V030 retrieval. Root cause: query returns ~8 facts (too narrow). Graph extraction works (~270 facts / conversation) but the query-side scoring is too strict and misses reasoning-heavy questions. Kept as `SOPHON_GRAPH_MEMORY=1` opt-in, flagged **experimental**.
2. **ReAct iterative retrieval** — when stacked on HyDE / FactCards the accumulated rankings dilute each other via RRF. Best config at N=3 smoke: HyDE alone > HyDE + ReAct. Kept as opt-in flag but not on by default.
3. **Multi-hop LLM decomposer** (`SOPHON_MULTIHOP_LLM=1`) — 0 pt on N=15. The decomposer paraphrases the question rather than splitting it into genuinely independent sub-queries, and HashEmbedder keyword retrieval can't follow the distinction.
4. **Adversarial -33 pt regression at N=6** (reported early) — confirmed as sampling noise by a focused N=5 audit run (V030 4/5 = V032 4/5). No fix needed; we were about to build an over-engineered coverage guard.
5. **Reproducing the 40 % LOCOMO public claim** — our stratified N=20/30 V030 reruns land at 25-33 %, inside the Wilson 95 % CI for the published N=80 40 % point but noticeably below the point estimate. We flag this and run V032 comparisons against our own V030 baseline, not against the published number.

### Breaking changes

None. Every new behaviour is opt-in via env var. A caller that only
sets `SOPHON_RETRIEVER_PATH` gets v0.3.0 behaviour byte-for-byte.

### Known limitations (carried from 0.3.0)

- LOCOMO multi-hop stays hard — V032 improves 0 → 17 % but mem0 / HippoRAG (neural embeddings) sit at 80-90 %. Sophon's zero-ML-at-query-time constraint caps us below that without relaxing the architecture.
- Default HashEmbedder remains keyword-bound. Query/answer vocabulary mismatch (e.g. "favorite food" ↔ "weakness for ginger snaps") requires HyDE to bridge.
- `compress_history` without retrieval still drops mid-conversation details — the canary test shows 27 % preservation at V030 default, 80 % with V032 stack.

---

## [0.3.0] — 2026-04-10

### Added

- LOCOMO multi-scale bench validating COMP_LLM at 40 % accuracy stable across N=30 / N=60 / N=80 (§ 7.12 of v0.3.0 BENCHMARK)
- `SOPHON_RETRIEVER_PATH` activates the semantic retriever — adds the `query` / `retrieval_top_k` args to `compress_history` and unlocks the +13-pt LOCOMO open-ended gain (§ 3.7 of v0.3.0 BENCHMARK)
- BGE-small-en-v1.5 embedder behind the `bge` cargo feature (§ 7.9 of v0.3.0 BENCHMARK)

### Measured (preserved for context)

- Session aggregate savings : 67.0 % on the 4-call workflow
- Cross-model savings : 64.5 % ± 0.5 % across 6 variants × 3 tasks
- LOCOMO COMP_LLM : 40.0 % (N=80)
- LOCOMO RETR_HASH : 32.5 % (N=80)
- LOCOMO FULL ceiling : 71.2 % (N=80)
- Multi-hop accuracy : 0 % across all Sophon conditions
- vs LLMLingua-2 on structured prompts : +8.9 pt at 35× lower latency
- vs mem0-lite on LOCOMO (N=15) : tied at 60 % accuracy, sub-second vs 8.7 min, zero vs ~330 LLM calls
- Compression overhead per call : ~6 ms

### Historical corrections (kept as methodological record)

- **N=15 → N=40 LOCOMO** — RETR_HASH dropped from 60 % to 42.5 % when the sample scaled up. Original N=15 was optimistic.
- **N=30 → N=60 LOCOMO open-ended** — SOPHON_RETR dropped from 46.7 % to 36.7 %. The +13-pt retrieval gain vs compression held, the absolute number did not.
- **N=15 all-conditions** — early v0.2.1 table showed RETR_BGE beating RETR_HASH by +6.7 pt; at N=40 the gap inverted (Hash +10 pt), at N=80 they were tied. The BGE win was sampling luck on easy items.

---

## [0.2.x] — historical

Sophon 0.2.x introduced the LLM-summariser path, the fragment cache,
and the delta streamer. Block-based summarisation replaced 4000-char
truncation in 0.2.2, which fixed the N=40 COMP_LLM gap identified in
the prior release. See the archived BENCHMARK tables in git history
for per-version numbers.

---

## [0.1.x] — initial release

Stateless token counting, heuristic summariser, codebase navigator,
output compressor with command-aware filters. Rust binary ~10 MB.
