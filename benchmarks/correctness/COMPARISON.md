# compress_history summariser — A/B (key-fact recall)

Same 51 real-repo tasks, same answerer (haiku), same blind judge (sonnet), ratio 1/3,
N=51. The fix targets `compress_history`: the old default summary kept only the first
sentence of each dropped message, so specific old facts vanished. We compare three
summarisers on the metric that matters — does a buried old fact survive into the answer?

| `compress_history` summariser | history recall | overall sophon recall | Sophon−truncate (overall) | Sophon−oracle (overall) | deterministic? |
|---|---|---|---|---|---|
| first-sentence (old default) | **0.8%** | 20.9% | −0.6 pts | −47.8 | yes |
| LLM abstractive (haiku, ~100 words/block) | 4.6% | 22.4% | +1.2 | −46.3 | no (LLM call) |
| **extractive, fact-preserving (NEW default)** | **19.1%** | **28.1%** | **+4.2** | **−40.6** | **yes** |

## Read

- **The extractive default lifts old-fact recall 24× over the old heuristic (0.8% → 19.1%)** and
  ~4× over the LLM path — while staying deterministic, network-free, zero-added-latency.
  True to "compression pure, zéro ML au query-time".
- **Abstractive summarisation is *worse than* extractive here (4.6% < 19.1%)**: the LLM
  "100-word paragraph" prompt paraphrases away the exact numbers / identifiers / paths
  (5.2 MB, 405 tests, `README.md`) the questions need. Keeping the high-signal lines
  *verbatim* beats rewriting them.
- **Overall, the fix flips Sophon from behind dumb truncation (−0.6) to ahead (+4.2)** at equal
  budget, and cuts the loss-vs-full-context from −47.8 to −40.6 pts.
- **Absolute recall is still bounded (19.1%)** because compressing ~6.5k tokens to ~2.2k and then
  asking about one specific old turn is hard, and `compress_history` is *not query-aware* (it
  summarises blind at ingest). That is the next architectural lever, not a summariser tweak.
- The CI on Sophon−truncate still brushes zero (n=51), so the honest claim is "at least as good
  as truncation at equal budget, clearly better on history and prompt".

## Reproduce

```bash
cd benchmarks/correctness
python3 run_correctness.py                            # extractive (new default)  → REPORT.md
python3 run_correctness.py --summary-mode llm         # LLM abstractive            → REPORT.llm.md
python3 run_correctness.py --summary-mode legacy      # old first-sentence         → REPORT.legacy.md
```

Successful LLM calls are cached on disk (`results/cache/`); re-runs only pay for cells whose
context changed or that failed before (failures are retried, not cached).
