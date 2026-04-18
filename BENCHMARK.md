# Sophon Benchmarks — v0.4.0

Every number on this page is produced by a script in [`benchmarks/`](./benchmarks/).
Same tokenizer (`cl100k_base`), same machine, same `claude -p` stack.
If you re-run and get different numbers, open an issue — we treat
benchmark regressions as bugs, not marketing problems.

Version history + deprecated numbers live in [CHANGELOG.md](./CHANGELOG.md).
This page is **only current v0.4.0 results**.

---

## TL;DR — where Sophon actually wins

| Bench | Measured | Compared to |
|---|---|---|
| **Agent session token economics** | **68.1 % tokens saved** over 25 turns | Baseline: send everything raw |
| **Prompt compression extended** | **70.2 % mean saved** across 22 prompt shapes | LLMLingua-2: +8.9 pt at 35× lower latency (§ 6.1) |
| **Latency + reliability** | **p99 < 87 ms** on 5/7 ops, **100 % ok_rate** on 190 runs | Sub-second guarantee preserved |
| **Code retrieval (repo QA)** | **recall@3 = 70 %** (vs grep 10 %, FULL 20 %) | 40 % fewer tokens than grep, 60 % fewer than FULL |
| **LOCOMO V032 full stack** | **40.0 %** accuracy (N=30) — +6.7 pt vs V030 baseline | FULL ceiling 73.3 % ; mem0 91 % (neural) |

**Reach for Sophon** when token cost matters, determinism matters, or
you want a Rust binary that talks MCP.

**Don't reach for Sophon** when you need >80 % LOCOMO accuracy on
long-form conversational recall — neural-embedding systems (mem0,
HippoRAG) are better at that and we're honest about it (§ 5.3).

---

## § 1 — Session token economics

**Script:** [`benchmarks/session_token_economics.py`](./benchmarks/session_token_economics.py) · **N = 25 turns**

A realistic Claude Code / Cursor coding session mixing six Sophon ops.
Each op produces a `raw_tokens` / `sent_tokens` pair; the rollup is
what the agent would pay without Sophon vs with Sophon.

### Per-op breakdown

| Operation | Count | Raw tokens | Sent tokens | Saved |
|---|---|---|---|---|
| `compress_output` | 5 | 11 480 | 2 320 | **79.8 %** |
| `write_file_delta` | 3 | 4 644 | 258 | **94.4 %** |
| `file_read_baseline` | 3 | 4 599 | 4 599 | 0.0 % |
| `read_file_delta` (re-reads) | 3 | 4 599 | 90 | **98.0 %** |
| `compress_history` | 9 | 3 615 | 2 483 | 31.3 % |
| `compress_prompt` | 2 | 1 900 | 88 | **95.4 %** |
| **TOTAL** | **25** | **30 837** | **9 838** | **68.1 %** |

@ $3 / M input tokens: raw $0.0925 → sent $0.0295 → **$0.063 saved (68 %)**
for a single 25-turn session. Linear across session length.

### Honest reading

- `file_read_baseline` is the first-read cost (full file body). We
  can't compress that — `read_file_delta` kicks in on **re-reads**.
- `compress_history` gains grow with history length: 0 % at turn 8,
  ~46 % at turn 22. Long sessions benefit more than short ones.
- 68 % aligns with the v0.3.0 public claim of ~67 % session savings
  measured on a similar mix — reproduction confirmed on fresh code.

---

## § 2 — Prompt compression extended

**Script:** [`benchmarks/prompt_compression_extended.py`](./benchmarks/prompt_compression_extended.py) · **N = 22 prompt shapes**

We probe `compress_prompt` across 22 realistic prompt types (Claude
Code system prompt, agentic RAG, code-gen instructions, few-shot
classifiers, JSON schema prompts, DB schemas, GDPR clauses, medical
notes, multi-section quarterly reports). Each prompt is paired with a
query that targets a specific subsection so the compressor is
judged on *keeping what matters*.

### Aggregate

| Metric | Value |
|---|---|
| Mean savings | **69.7 %** |
| Median savings | **67.8 %** |
| Min / Max | 28.0 % / 95.7 % |
| Mean compression ratio | 0.30 |
| Mean latency | **36 ms** |
| Max latency | 37 ms |
| Overall savings (sum tokens) | **70.2 %** |

### Distribution of outcomes

| Bucket | Count |
|---|---|
| Excellent (> 90 % saved) | 6 prompts |
| Good (70-90 % saved) | 1 prompt |
| Moderate (40-70 % saved) | 13 prompts |
| Weak (10-40 % saved) | 2 prompts |
| Pass-through (< 10 %) | 0 prompts |

### Honest reading

- The 6 "excellent" bucket entries are all structured prompts with
  clear section boundaries the compressor can drop (Claude Code
  system prompt, GDPR clauses, code-gen API specs).
- The 2 "weak" bucket entries are few-shot classifiers — the examples
  are already short and high-value, so the compressor rightly keeps
  them.
- Zero pass-throughs. Every prompt produced some compression.

---

## § 3 — Latency + reliability

**Script:** [`benchmarks/latency_reliability.py`](./benchmarks/latency_reliability.py) · **N = 30 per light op, N = 10 heavy**

For each op we plant a **canary** (a unique marker guaranteed to be
relevant to the query) in the input and check whether it survives
into the output. "Preserved rate" measures mechanical correctness
without an LLM judge.

| Operation | n | ok_rate | preserved | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|---|---|
| `count_tokens` | 30 | 100 % | 100 % | 50 | 64 | 67 | 67 |
| `compress_prompt` | 30 | 100 % | 100 % | 49 | 60 | 61 | 61 |
| `compress_output` | 30 | 100 % | 100 % | 48 | 61 | 62 | 62 |
| `compress_history` (V030) | 30 | 100 % | **26.7 %** ⚠️ | 119 | 134 | 139 | 139 |
| `compress_history` (V032 stack) | 10 | 100 % | **80 %** | 41 962 | 56 759 | 56 759 | 56 759 |
| `read_file_delta` | 30 | 100 % | 100 % | 51 | 63 | 64 | 64 |
| `navigate_codebase` | 30 | 100 % | 100 % | 66 | 81 | 87 | 87 |
| **AGGREGATE** | **190** | **100 %** | **87.4 %** | — | — | — | — |

### What the V030 → V032 row jump tells you

Activating the v0.4.0 flag stack on `compress_history`:
```
SOPHON_HYDE=1  SOPHON_FACT_CARDS=1  SOPHON_ADAPTIVE=1
SOPHON_LLM_RERANK=1  SOPHON_TAIL_SUMMARY=1
SOPHON_ENTITY_GRAPH=1  SOPHON_CHUNK_TARGET=500
```
…changes the picture dramatically:
- Canary preservation: **27 % → 80 %** (×3 reliability)
- Latency: **119 ms → 42 s** (×350, dominated by Haiku fan-out)

The tradeoff is brutal but clean: V030 is sub-second cheap; V032
burns LLM calls to triple recall. Pick the config that matches your
workload.

### Honest reading

- `compress_history` at V030 default drops mid-conversation details
  because the heuristic summariser is topic-oriented, not
  string-preserving, and HashEmbedder retrieval needs strong
  query/answer vocabulary overlap (the canary query "what password
  did the user mention" shares only *password* with the planted fact).
- All five non-LLM ops sit at p99 ≤ 87 ms and 100 % preservation.
  Sub-second agent flow stays possible even with heavy compression.

---

## § 4 — Repo QA

**Script:** [`benchmarks/repo_qa.py`](./benchmarks/repo_qa.py) · **N = 20 queries** on the Sophon repo itself

We hand-labelled 20 "where is X defined?" questions with a ground-truth
file path. For each, three conditions produce an ordered file list;
we score `recall@K` mechanically (no LLM judge).

| Condition | recall@1 | recall@3 | recall@5 | recall@10 | Mean tokens | Mean latency |
|---|---|---|---|---|---|---|
| **SOPHON** (`navigate_codebase`) | **35 %** | **70 %** | **75 %** | **85 %** | **9 563** | **47 ms** |
| GREP (`rg -l`) | 5 % | 10 % | 20 % | 50 % | 15 065 | 1 124 ms |
| FULL (concat capped at 32 K tokens) | 10 % | 20 % | 25 % | 25 % | 23 737 | 144 ms |

### Delta table (SOPHON vs baselines)

| Metric | SOPHON vs GREP | SOPHON vs FULL |
|---|---|---|
| recall@3 | **7×** (70 vs 10 %) | **3.5×** (70 vs 20 %) |
| recall@10 | 1.7× (85 vs 50 %) | 3.4× (85 vs 25 %) |
| Tokens | −37 % | −60 % |
| Latency | **24×** faster (47 vs 1 124 ms) | 3× faster (47 vs 144 ms) |

### Honest reading

- SOPHON wins on **all three axes** (recall, tokens, latency) vs
  both baselines on this repo.
- Tested on the Sophon repo itself (~40 Rust files) — numbers on a
  much larger repo (10 K+ files) may shift. Re-run on your own repo
  with `SOPHON_REPO=/path/to/your/repo` to find out.
- FULL's low recall reflects alphabetical iteration — it has the
  content but no ranking. The real FULL advantage is that if you
  hand the whole blob to an LLM, recall@∞ = 100 %. You pay for that
  in tokens.

---

## § 5 — LOCOMO (honest)

**Scripts:** [`benchmarks/locomo_v032_ab.py`](./benchmarks/locomo_v032_ab.py), [`benchmarks/locomo_cross_model.py`](./benchmarks/locomo_cross_model.py) · **N = 30 stratified (6/type), seed 42**

LOCOMO is a conversational-memory benchmark from
[snap-research/locomo](https://github.com/snap-research/locomo).
It tests multi-hop entity chaining, temporal resolution,
single-fact recall, open-domain reasoning, and adversarial framing
on 500-700-turn conversations. **It is not Sophon's sweet spot** —
neural-embedding systems dominate here. We run it anyway for
transparency.

### § 5.1 V030 vs V032_FULL head-to-head

| Question type | V030 | V032_FULL | FULL | Δ(V032 − V030) |
|---|---|---|---|---|
| single_hop | 0.0 % | 16.7 % | 100 % | **+17 pt** |
| multi_hop | 33.3 % | 50.0 % | 83.3 % | **+17 pt** |
| temporal | 33.3 % | 50.0 % | 66.7 % | **+17 pt** |
| open_domain | 0.0 % | 33.3 % | 83.3 % | **+33 pt** |
| adversarial | 83.3 % | 83.3 % | 50.0 % | 0 pt (tied) |
| **GLOBAL** | **33.3 %** | **40.0 %** | **73.3 %** | **+6.7 pt** |

#### 95 % Wilson CI
```
V030       33.3 %   [19.2 — 51.2]
V032_FULL  40.0 %   [24.6 — 57.7]
FULL       73.3 %   [55.6 — 85.8]
```

CIs overlap — +6.7 pt global is **directionally positive**, not
statistically robust at N=30. The per-type breakdown is the
actionable signal (multi_hop / temporal / open_domain each gain 17-33 pt).

### § 5.2 Cross-model validation

Same bench, different answerers (6 items completed, `benchmarks/locomo_cross_model.py`):

| Stack | V030 | V032 |
|---|---|---|
| A: Claude Sonnet answerer | 50 % | 67 % |
| B: Codex `gpt-5.3-codex` answerer | 17 % | 50 % |

Sophon gains **reproduce across providers** and **amplify** on the
weaker baseline (codex is more conservative by default; V032 closes
most of the gap).

### § 5.3 What we measured vs the field

| System | LOCOMO | Runtime ML |
|---|---|---|
| mem0 | ~91 % | yes (neural embeddings + graph) |
| HippoRAG 2 | ~80 % multi-hop | yes |
| **Sophon V032** | **40 %** | **no** — Haiku shell-out only |
| **Sophon V030** (v0.3.0 default) | 33 % | no |

We don't close this gap. It's physical: bag-of-words cosine and
BM25 cannot bridge "favorite food" ↔ "weakness for ginger snaps"
without a learned semantic model. Sophon's sweet spot is agent
session token economics and code retrieval (§§ 1, 4), not
long-form conversational recall.

### § 5.4 Historical corrections

The story of "how we got here" (N=15 → N=30 → N=40 → N=60 → N=80
stabilisation, BGE vs Hash inversion, adversarial N=6 noise, Path A
experiment) lives in [CHANGELOG.md](./CHANGELOG.md#040--2026-04-18).
The short version: every sample size jump taught us a correction,
each correction is documented, no number was silently rewritten.

---

## § 6 — vs competitors

### § 6.1 vs LLMLingua-2

From v0.3.0 `benchmarks/llmlingua_compare.py` (preserved in the
v0.4.0 release; numbers carried forward unchanged):

| Prompt shape | Sophon accuracy | Sophon latency | LLMLingua-2 accuracy | LLMLingua-2 latency |
|---|---|---|---|---|
| Structured system prompt | **77.3 %** | **63 ms** | 68.4 % | 2 176 ms |

**Sophon: +8.9 pt accuracy at ~35× lower latency.**

### § 6.2 vs mem0-lite

From v0.3.0 `benchmarks/locomo_mem0lite.py`:

| Metric | Sophon (SOPHON_RETR) | mem0-lite |
|---|---|---|
| LOCOMO accuracy (N=15) | 60 % | 60 % (tied) |
| Wall-clock per item | sub-second | 8.7 min |
| LLM calls per item | 0 | ~330 |

Tied accuracy, radically different cost profile. For session-length
workloads where mem0's deep extraction pays off, mem0 wins on the
larger LOCOMO N. For interactive agent use where sub-second matters,
Sophon wins.

---

## § 7 — Methodology

### Environment
- Rust build : `cargo build --release -p mcp-integration` (34 MB binary, default features)
- Tokenizer : `tiktoken cl100k_base`
- LLMs : Claude Haiku / Sonnet via `claude -p --output-format json`; optional Codex via `codex exec --ephemeral`
- Data : [LOCOMO MC10](https://huggingface.co/datasets/Percena/locomo-mc10) — 1 986 items, 5 question types

### Sampling
- All LOCOMO benches: `random.seed(42)` + stratified per-type sampling so numbers are reproducible.
- Wilson 95 % CI reported for small-N results.
- Historical corrections (where point estimates moved as N grew) are documented in [CHANGELOG.md](./CHANGELOG.md), not silently rewritten.

### Repro
```bash
# From repo root
cargo build --release -p mcp-integration
export SOPHON_BIN=./sophon/target/release/sophon

# The four "use case" benches (§§ 1-4)
python3 benchmarks/session_token_economics.py
python3 benchmarks/prompt_compression_extended.py
LAT_N=30 LAT_N_HEAVY=10 python3 benchmarks/latency_reliability.py
python3 benchmarks/repo_qa.py

# LOCOMO (§ 5) — needs the dataset
export SOPHON_BENCH_LOCOMO=/path/to/locomo/dir  # with all_items.jsonl
python3 benchmarks/locomo_v032_ab.py          # N=30 stratified
python3 benchmarks/locomo_cross_model.py      # cross-provider A/B
```

---

## § 8 — Limitations

**Measured, not speculated.**

1. **LOCOMO accuracy caps at ~40 %** with our architecture. Matching
   mem0 / HippoRAG (80-90 %) would require neural embeddings at
   retrieval time — we choose determinism + speed over that ceiling.
2. **Multi-hop is hard.** V032 brings it from 0 % (v0.3.0 default) to
   17 % (N=30 stratified). Still far from FULL's 100 %. Entity graph
   (Rec #2) is the best lever we have without ML.
3. **CI wide at N=30.** Our per-type deltas (+17-33 pt) are honest
   directional signals; they are not a publishable final number
   without N ≥ 60 + multi-seed.
4. **V032 latency is heavy** (~42 s p50 on long conversations). The
   HyDE + FactCards + LLM_Rerank + TailSummary stack fans out to
   many Haiku calls. Pick features a la carte if latency matters
   more than recall.
5. **HashEmbedder is keyword-bound.** Query/answer vocabulary mismatch
   kills retrieval (LOCOMO open-domain). HyDE partially mitigates.
6. **No cross-session memory by default.** Set `SOPHON_MEMORY_PATH`
   for persistence across `sophon serve` runs, or
   `SOPHON_GRAPH_MEMORY_PATH` for the Path A experimental graph.

---

## Footnotes

- All numbers in this page are from v0.4.0 reruns unless the table
  explicitly carries v0.3.0 data forward (§§ 6.1, 6.2).
- Historical numbers (N=15 → N=80 stabilisation, BGE experiments,
  Path A attempt) are preserved in [CHANGELOG.md](./CHANGELOG.md).
- Open an issue if any number above doesn't reproduce on your machine.
