# Sophon — correctness bench (key-fact recall)

Answerer **haiku** · blind judge **sonnet** · compress ratio 1/3 · **N=51** real-repo tasks.

Each arm answers from a context of comparable token budget; the judge scores what fraction of the pre-frozen key facts survive into the answer.

**Baselines.** `oracle` = full context (ceiling). `truncate` = naive cap at the *same token count Sophon emitted*: HEAD-truncation for `prompt`/`output` (keep the start), TAIL/recent-window for `history` (keep recent turns — the realistic sliding window compress_history claims to beat by summarising the dropped tail). History questions deliberately target OLD turns: that is compress_history's reason to exist, so it is the charitable test of its core value.

## Headline

| Arm | Mean recall | Mean token savings |
|---|---|---|
| oracle | 68.7% | 0.0% |
| sophon | 20.9% | 67.2% |
| truncate | 21.5% | 67.5% |

**Sophon − truncate (equal budget): -0.6 pts** (95% CI [-7.5, +5.8], n=51)  
**Sophon − oracle (loss vs full context): -47.8 pts** (95% CI [-57.0, -38.4], n=51)

## Per-op breakdown

| op | n | recall oracle | recall sophon | recall truncate | sophon−trunc | sophon savings |
|---|---|---|---|---|---|---|
| history | 20 | 69.4% | 0.8% | 0.8% | +0.0 | 76.6% |
| output | 7 | 54.0% | 34.9% | 51.2% | -16.3 | 22.3% |
| prompt | 24 | 72.4% | 33.7% | 30.2% | +3.5 | 72.3% |

## Where Sophon loses to dumb truncation (6/51)

| task | op | sophon | truncate |
|---|---|---|---|
| prompt-006 (doc:CHANGELOG.md#1) | prompt | 0% | 62% |
| prompt-008 (doc:CHANGELOG.md#3) | prompt | 0% | 57% |
| prompt-011 (src:sophon/crates/cli-hooks/src/installer.rs) | prompt | 17% | 33% |
| prompt-014 (src:sophon/crates/codebase-navigator/src/extractors/c_cpp.rs) | prompt | 0% | 20% |
| output-050 (cmd:public fns) | output | 0% | 100% |
| output-051 (cmd:manifest heads) | output | 57% | 71% |

## What this means

- **The 67% token saving is real but not free**: compressing to 1/3 budget costs **48 pts of key-fact recall** vs the full context (69% → 21%). This is the cost the token-only headline hid.
- **At equal budget, Sophon ≈ naive truncation** (-0.6 pts, CI spans 0). The compression is not meaningfully "smarter" than a dumb cap on average — it wins modestly on `prompt` (section selection), loses on `output`, and ties on `history`.
- **`compress_history` summariser** (`memory-manager/src/summarizer.rs`): the old default kept only the first sentence of each dropped message, so buried old facts vanished (recall ~0.8%). The current default is a deterministic **extractive** summariser that keeps the highest-signal lines verbatim → history recall **15.4%** (19× the old heuristic, 2× the opt-in LLM path). See `COMPARISON.md`. Next lever is making it query-aware (it summarises blind at ingest).
- **Honest pitch**: sell bounded, *measured* token savings (and the structured included/excluded-section output), not "we keep more than truncation" — at a fixed budget that claim does not hold here.
