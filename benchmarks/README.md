# Sophon benchmarks

These scripts produce the numbers published in
[`../BENCHMARK.md`](../BENCHMARK.md). Everything here is reproducible
from the same machine, same tokenizer (`tiktoken cl100k_base`), same
inputs.

## Scripts

| Script | What it produces | BENCHMARK.md section |
|---|---|---|
| `llmlingua_compare.py` | Sophon `compress_prompt` vs LLMLingua-2 head-to-head | [§ 7.8.d](../BENCHMARK.md#78d-head-to-head-sophon-compress_prompt-vs-llmlingua-2) |
| `locomo_retrieval.py` | Sophon 4-condition LOCOMO run (NONE / SOPHON_COMP / SOPHON_RETR / FULL) | [§ 3.7](../BENCHMARK.md#37-locomo-open-ended-retrieval-n60) |
| `locomo_mem0lite.py` | mem0-lite LOCOMO run using `claude -p` as the LLM | [§ 7.8.e](../BENCHMARK.md#78e-mem0-lite-on-locomo--same-item-comparison) |
| `repos_scan.py` | Scan 5 SHA-pinned GitHub repos | [§ 7.2 / § 7.8.b](../BENCHMARK.md#72-scanning-real-code) |
| `repos_recall.py` | Recall@5 / @10 on real commits from those repos | [§ 7.3](../BENCHMARK.md#73-recall-on-real-commits) |
| `llm_cli.py` | Universal CLI wrapper — normalises `claude` and `codex` output shapes into plain stdout. Used by the cross-model harness below. | — |
| `locomo_cross_model.py` | 4-way LOCOMO run (V030 / V032_FULL × Stack A / Stack B) with parameterisable answerer and judge. Tests whether Sophon gains are provider-agnostic. | — |

## Cross-model benches

Sophon's internal LLM calls (block summaries, HyDE, fact cards, rerank, tail
summary, classifier) shell out to whatever `SOPHON_LLM_CMD` points at, and
the bench harness's answerer + judge are now parameterisable too. This
lets you validate Sophon as provider-agnostic and confirm gains reproduce
outside the default Claude-only setup.

```bash
# Default: Stack A = Sonnet answerer+judge, Stack B = codex answerer.
# Internal Sophon LLM stays on Haiku so the RETRIEVAL stays fixed and
# only the answering layer changes.
python3 benchmarks/locomo_cross_model.py

# All-Claude control (both stacks = Sonnet, but different models).
CROSS_ANSWERER_A="python3 benchmarks/llm_cli.py --provider claude --model sonnet" \
CROSS_ANSWERER_B="python3 benchmarks/llm_cli.py --provider claude --model opus" \
python3 benchmarks/locomo_cross_model.py

# Focused: adversarial items only, N=10 per type × 1 type
CROSS_N=10 CROSS_TYPES=adversarial python3 benchmarks/locomo_cross_model.py
```

Env vars:

| Var | Default | Role |
|---|---|---|
| `SOPHON_LLM_CMD` | `claude -p --model haiku --output-format json` | Internal Sophon LLM (summary / HyDE / FC / rerank / tail / classifier) |
| `CROSS_ANSWERER_A` | `python3 benchmarks/llm_cli.py --provider claude --model sonnet` | Stack A answer model |
| `CROSS_ANSWERER_B` | `python3 benchmarks/llm_cli.py --provider codex` | Stack B answer model |
| `CROSS_JUDGE` | `python3 benchmarks/llm_cli.py --provider claude --model sonnet` | Shared judge (constant so the comparator is fixed) |
| `CROSS_N` | `4` | Samples per question type |
| `CROSS_TYPES` | all 5 LOCOMO types | Comma-separated filter |

The harness reports per-type accuracy for each `V030 × Stack` and
`V032_FULL × Stack` combination, plus two deltas — "same Sophon,
different answerer" and "same answerer, different Sophon" — so
it's obvious which axis drives any accuracy change.

## Environment variables

Every script reads its paths from environment variables so nothing is
hard-coded to a specific working tree.

| Variable | Default | What it points at |
|---|---|---|
| `SOPHON_BIN` | `sophon` | The `sophon` binary (defaults to `$PATH` lookup) |
| `SOPHON_REPO_ROOT` | `.` | Repository root (for `llmlingua_compare.py`) |
| `SOPHON_BENCH_DIR` | `./benchmarks/data` | Bench fixtures root |
| `SOPHON_BENCH_REPOS` | `./benchmarks/data/repos` | 5 cloned GitHub repos (SHAs pinned, see `sha_pins.json`) |
| `SOPHON_BENCH_LOCOMO` | `./benchmarks/data/locomo` | LOCOMO `all_items.jsonl` + per-item run caches |
| `LLMLINGUA_OUT` | `llmlingua_results.json` | Output file for the LLMLingua comparison |

## Prerequisites

- **Rust binary** — `cargo build --release -p mcp-integration`
  (optionally with `--features codebase-navigator/tree-sitter` for
  § 7.6 and § 7.8.a).
- **`claude` CLI** — used by every LOCOMO script as the answer and
  judge model. Tested with Sonnet 4.6.
- **Python 3.9+** — `pip install tiktoken llmlingua` for
  `llmlingua_compare.py`. The LOCOMO scripts are pure-stdlib.
- **5 GitHub repos cloned at pinned SHAs** — for `repos_scan.py` and
  `repos_recall.py`. See `../BENCHMARK.md § 7.1` for the SHA list.
- **LOCOMO dataset** — download `all_items.jsonl` from the
  [LOCOMO GitHub](https://github.com/snap-research/locomo) or
  [Hugging Face mirror](https://huggingface.co/datasets/Percena/locomo-mc10)
  into `$SOPHON_BENCH_LOCOMO/`.

## Quick start

```bash
# from the repo root:
cargo build --release -p mcp-integration
export SOPHON_BIN=./sophon/target/release/sophon
export SOPHON_REPO_ROOT=$(pwd)

# LLMLingua head-to-head (~2 minutes)
mkdir -p benchmarks/data
cp sophon/tests/fixtures/claude_system_prompt.txt benchmarks/data/system_prompt_large.txt  # or use your own
python3 benchmarks/llmlingua_compare.py

# LOCOMO retrieval (~1 hour at N=60)
mkdir -p benchmarks/data/locomo
# ...place all_items.jsonl here...
python3 benchmarks/locomo_retrieval.py

# mem0-lite LOCOMO (~10 min at N=15)
python3 benchmarks/locomo_mem0lite.py
```

## A note on data files

The scripts expect input fixtures to live under `benchmarks/data/`.
That directory is **not** checked in (too large, too volatile). You
provide it yourself — the scripts error out with a readable message
if a fixture is missing. See `../BENCHMARK.md § 7.1` for the exact
list of fixtures each section depends on.

## Honesty clause

Every number in `../BENCHMARK.md` is produced by these scripts. If
you run them on your machine and get different numbers, open an
issue — the repo treats benchmark regressions as bugs, not
marketing problems.
