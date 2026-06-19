# Sophon — correctness bench (key-fact recall)

Answerer **haiku** · blind judge **sonnet** · compress ratio 1/3 · **N=51** real-repo tasks.

Each arm answers from a context of comparable token budget; the judge scores what fraction of the pre-frozen key facts survive into the answer.

**Baselines.** `oracle` = full context (ceiling). `truncate` = naive cap at the *same token count Sophon emitted*: HEAD-truncation for `prompt`/`output` (keep the start), TAIL/recent-window for `history` (keep recent turns — the realistic sliding window compress_history claims to beat by summarising the dropped tail). History questions deliberately target OLD turns: that is compress_history's reason to exist, so it is the charitable test of its core value.

## Headline

| Arm | Mean recall | Mean token savings |
|---|---|---|
| oracle | 68.7% | 0.0% |
| sophon | 43.7% | 62.7% |
| truncate | 23.7% | 63.1% |

**Sophon − truncate (equal budget): +20.0 pts** (95% CI [+10.9, +29.5], n=51)  
**Sophon − oracle (loss vs full context): -25.0 pts** (95% CI [-33.6, -16.5], n=51)

## Per-op breakdown

| op | n | recall oracle | recall sophon | recall truncate | sophon−trunc | sophon savings |
|---|---|---|---|---|---|---|
| history | 20 | 69.4% | 32.7% | 0.0% | +32.7 | 69.2% |
| output | 7 | 54.0% | 49.1% | 51.2% | -2.0 | 22.7% |
| prompt | 24 | 72.4% | 51.3% | 35.5% | +15.7 | 68.9% |

## Where Sophon loses to dumb truncation (8/51)

| task | op | sophon | truncate |
|---|---|---|---|
| prompt-011 (src:sophon/crates/cli-hooks/src/installer.rs) | prompt | 17% | 33% |
| prompt-012 (src:sophon/crates/cli-hooks/src/rewriter.rs) | prompt | 14% | 29% |
| prompt-015 (src:sophon/crates/codebase-navigator/src/extractors/go.rs) | prompt | 17% | 33% |
| prompt-016 (src:sophon/crates/delta-streamer/src/differ.rs) | prompt | 50% | 100% |
| prompt-019 (src:sophon/crates/fragment-cache/src/detector.rs) | prompt | 33% | 83% |
| prompt-020 (src:sophon/crates/fragment-cache/src/encoder.rs) | prompt | 33% | 50% |
| prompt-022 (src:sophon/crates/mcp-integration/src/jsonrpc.rs) | prompt | 20% | 40% |
| output-051 (cmd:manifest heads) | output | 57% | 71% |

## What this means

- **The 63% token saving is real but not free**: compressing to 1/3 budget costs **25 pts of key-fact recall** vs the full context (69% → 44%). This is the cost the token-only headline hid.
- **At equal budget, Sophon beats naive truncation by +20.0 pts** (95% CI [+10.9, +29.5] — excludes 0, so the edge is statistically real on this task mix). The edge is large and uneven: it wins on `prompt`, `history` and still loses on `output`.
- **`compress_history` summariser** (`memory-manager/src/summarizer.rs`): the old default kept only the first sentence of each dropped message, so buried old facts vanished (recall ~0.8%). The current default is a deterministic **query-aware extractive** summariser: it ranks the dropped older lines by relevance to the question (then information density), so the line that answers the query lands in the summary → history recall **32.7%**. See `COMPARISON.md`. (Pass the question via the `query` arg to activate it; with no query it falls back to density-only ranking.)
- **Honest pitch**: bounded, *measured* token savings AND a statistically real key-fact-recall edge over naive truncation at equal budget — driven by `prompt`, `history`. The remaining soft spot is `output`.
