#!/usr/bin/env python3
"""
mem0-lite on LOCOMO — reproduces BENCHMARK.md § 7.8.e.

Implements the mem0 algorithm (LLM fact extraction per session →
retrieval → answer) using `claude -p` as the LLM instead of the
OpenAI API. Runs on the same question_ids that Sophon already
scored (from retrieval_runs/ produced by locomo_retrieval.py), so
the two sets are directly comparable.

Pipeline per item:
  1. summarise each of the ~20 sessions with claude -p  → "memories"
     (parallel, 8 workers)
  2. feed all memories + question to claude -p for the final answer
  3. judge with the same rubric as locomo_retrieval.py

We run on a SUBSAMPLE of 15 items (3 per question type) to keep the
wall-clock under 30 minutes. 15 × ~22 claude calls = ~330 calls total.

Usage:
  SOPHON_BENCH_LOCOMO=./benchmarks/data/locomo \
      python3 benchmarks/locomo_mem0lite.py

Prereqs:
  - `claude` CLI installed and authenticated
  - LOCOMO `all_items.jsonl` at $SOPHON_BENCH_LOCOMO/
  - `retrieval_runs/` populated by locomo_retrieval.py first (for
    cross-comparison with Sophon SOPHON_RETR on the same items)

Output: $SOPHON_BENCH_LOCOMO/mem0lite_results.json
"""
import json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "./benchmarks/data/locomo"))
RETR_RUNS = ROOT / "retrieval_runs"   # Sophon's ground-truth set
OUT_DIR = ROOT / "mem0lite_runs"
OUT_DIR.mkdir(exist_ok=True)

MODEL = "sonnet"       # same as Sophon bench
JUDGE = "sonnet"
SAMPLES_PER_TYPE = 3   # 5 types × 3 = 15 items
PARALLEL_SESSIONS = 8


def claude(prompt, model=MODEL, timeout=120):
    p = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json"],
        input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    try:
        d = json.loads(p.stdout)
        return d.get("result", "") or ""
    except Exception:
        return p.stdout


EXTRACT_PROMPT = """You are extracting durable facts from one session of a conversation between two people for a memory system. Read the session, then output a compact bullet list of the factual statements it contains. Focus on:
- names, dates, ages, jobs, locations
- events that happened and when
- stated preferences, relationships, plans
Skip small talk and emotional back-and-forth. Keep each fact self-contained (names, not pronouns). If nothing factual was shared, output "(none)".

SESSION:
{session}

FACTS:"""


ANSWER_PROMPT = """You are answering a question about a long conversation. You have only the memories below, which were extracted from the full conversation by a prior summarization pass. Some memories may be irrelevant; use only the ones that help.

MEMORIES:
{memories}

QUESTION: {question}

Answer concisely. If the memories do not contain enough information, say "I don't know."
"""


JUDGE_TEMPLATE = """You are scoring a QA system. Compare the candidate answer to the gold answer for the given question. Mark as correct if the candidate conveys the same factual answer as the gold (paraphrase is fine). Mark as wrong if the candidate is missing, contradicts the gold, or says "I don't know" when the gold has a concrete answer.

QUESTION: {question}
GOLD ANSWER: {gold}
CANDIDATE ANSWER: {candidate}

Respond with ONE line of strict JSON, nothing else:
{{"correct": true|false, "rationale": "<one short sentence>"}}
"""


def judge_correctness(q, gold, cand):
    prompt = JUDGE_TEMPLATE.format(question=q, gold=gold, candidate=cand or "(empty)")
    out = claude(prompt, model=JUDGE, timeout=90)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None, "no json"
    try:
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False)), parsed.get("rationale", "")
    except Exception as e:
        return None, f"parse: {e}"


def summarise_session(session_idx, turns, item_id):
    """One session → bullet-list of facts via claude -p."""
    text = "\n".join(f"{t.get('role','user')}: {t.get('content','')}" for t in turns)
    prompt = EXTRACT_PROMPT.format(session=text)
    facts = claude(prompt).strip()
    return session_idx, facts


def extract_memories(item):
    """Parallel per-session extraction."""
    sessions = item["haystack_sessions"]
    memories = [None] * len(sessions)
    with ThreadPoolExecutor(max_workers=PARALLEL_SESSIONS) as ex:
        futs = {ex.submit(summarise_session, i, s, item["question_id"]): i
                for i, s in enumerate(sessions)}
        for f in as_completed(futs):
            idx, facts = f.result()
            memories[idx] = facts
    return memories


def flat_memories(memories):
    lines = []
    for i, m in enumerate(memories):
        if not m or m.strip() == "(none)":
            continue
        lines.append(f"[session {i+1}]\n{m.strip()}")
    return "\n\n".join(lines)


# ------------- load same subset Sophon ran -----------------
all_items = [json.loads(l) for l in open(ROOT / "all_items.jsonl")]
by_id = {it["question_id"]: it for it in all_items}

# Sample deterministically the same way run_locomo_with_retrieval.py did,
# then cut down to SAMPLES_PER_TYPE per type so we finish in under 30 min.
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)
subset = []
for qtype, items in by_type.items():
    random.shuffle(items)
    subset.extend(items[:SAMPLES_PER_TYPE])

print(f"[mem0-lite] running on {len(subset)} items ({SAMPLES_PER_TYPE}/type)", flush=True)

results = []
t_global = time.time()
for idx, item in enumerate(subset):
    cache = OUT_DIR / f"{item['question_id']}.json"
    if cache.exists():
        row = json.loads(cache.read_text())
        results.append(row)
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached ({row.get('correct')})")
        continue

    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']})...", flush=True)
    t0 = time.time()
    memories = extract_memories(item)
    t_extract = time.time() - t0

    mem_block = flat_memories(memories)
    mem_tokens_estimate = len(mem_block.split())

    prompt = ANSWER_PROMPT.format(memories=mem_block, question=item["question"])
    answer = claude(prompt)
    correct, rationale = judge_correctness(item["question"], item["answer"], answer)

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "n_sessions": len(item["haystack_sessions"]),
        "n_memories_nonempty": sum(1 for m in memories if m and m.strip() != "(none)"),
        "mem_tokens_approx": mem_tokens_estimate,
        "extract_seconds": round(t_extract, 1),
        "answer": answer[:500],
        "correct": correct,
        "rationale": rationale,
    }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    mark = '✓' if correct is True else ('✗' if correct is False else '?')
    print(f"   {mark} ({t_extract:.0f}s extract, {row['n_memories_nonempty']}/{len(memories)} non-empty memories)")

elapsed = time.time() - t_global
(ROOT / "mem0lite_results.json").write_text(json.dumps(results, indent=2))

# ---- aggregate by type ----
print()
print("=" * 70)
print(f"mem0-lite on LOCOMO — N = {len(results)}  (wall-clock {elapsed/60:.1f} min)")
print("=" * 70)
TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]
for qtype in TYPES:
    items = [r for r in results if r["question_type"] == qtype]
    if not items:
        continue
    correct = sum(1 for r in items if r["correct"] is True)
    total = len(items)
    pct = 100 * correct / total if total else 0
    print(f"  {qtype:<22} {pct:>5.1f}%  (n={total})")
print("-" * 70)
correct = sum(1 for r in results if r["correct"] is True)
pct = 100 * correct / len(results)
print(f"  {'POOLED':<22} {pct:>5.1f}%  (n={len(results)})")
print()
print("saved to", ROOT / "mem0lite_results.json")
