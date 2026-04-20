#!/usr/bin/env python3
"""
LOCOMO oracle labeler — one-shot preprocessing.

For each LOCOMO item, asks Haiku: "which session indices contain the
evidence for this answer?" Caches results to oracle_labels.json keyed by
question_id. Once labeled, the retrieval oracle bench can run entirely
offline with zero LLM calls.

Cost: ~200 items × ~1500 input tokens each → ~300K tokens ≈ $0.10 Haiku.
Runtime: ~1-2 s per item, with a thread pool ~2-3 min for N=200.
"""
import json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
CACHE = ROOT / "oracle_labels.json"
N_LABEL = int(os.environ.get("SOPHON_ORACLE_N", "100"))
PER_TYPE = int(os.environ.get("SOPHON_ORACLE_PER_TYPE", "20"))
WORKERS = int(os.environ.get("SOPHON_ORACLE_WORKERS", "8"))

PROMPT_TEMPLATE = """You are labeling a long-conversation QA dataset. Given the question, the gold answer, and a numbered list of session summaries, identify which session indices (0-based) contain the evidence needed to answer.

Rules:
- Return EVERY session that contains evidence, not just one.
- If the answer requires bridging multiple sessions (multi-hop), return all of them.
- If NO session clearly contains the evidence, return an empty array.
- Respond with ONLY a JSON object on one line: {{"sessions": [indices]}}. No prose.

QUESTION: {question}
GOLD ANSWER: {answer}

SESSIONS:
{sessions}
"""


def format_sessions(item):
    summaries = item.get("haystack_session_summaries", [])
    datetimes = item.get("haystack_session_datetimes", [])
    lines = []
    for i, summ in enumerate(summaries):
        ts = datetimes[i] if i < len(datetimes) else ""
        lines.append(f"[{i}] {ts}\n{summ}")
    return "\n\n".join(lines)


def label_one(item):
    prompt = PROMPT_TEMPLATE.format(
        question=item["question"],
        answer=item["answer"],
        sessions=format_sessions(item),
    )
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return item["question_id"], None, "timeout"
    try:
        d = json.loads(p.stdout)
        txt = d.get("result", "") or ""
        m = re.search(r"\{[^{}]*\"sessions\"\s*:\s*\[[^\]]*\][^{}]*\}", txt, re.DOTALL)
        if not m:
            return item["question_id"], None, f"no json in: {txt[:200]}"
        parsed = json.loads(m.group(0))
        sessions = parsed.get("sessions", [])
        if not isinstance(sessions, list):
            return item["question_id"], None, "sessions not a list"
        sessions = [int(s) for s in sessions if isinstance(s, int) or (isinstance(s, str) and s.isdigit())]
        return item["question_id"], sessions, None
    except Exception as e:
        return item["question_id"], None, f"parse error: {e}"


def main():
    all_items = [json.loads(l) for l in open(ROOT / "all_items.jsonl")]
    by_type = {}
    for it in all_items:
        by_type.setdefault(it["question_type"], []).append(it)

    # Balanced sampling, deterministic seed.
    import random
    random.seed(42)
    subset = []
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype][:]
        random.shuffle(items)
        subset.extend(items[:PER_TYPE])
    print(f"[labeler] N={len(subset)}, {PER_TYPE}/type, seed=42")

    # Load existing cache
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        print(f"[labeler] cache hit: {len(cache)} existing labels")

    to_label = [it for it in subset if it["question_id"] not in cache]
    print(f"[labeler] labeling {len(to_label)} new items with {WORKERS} workers")

    import time
    t0 = time.perf_counter()
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(label_one, it): it for it in to_label}
        for fut in as_completed(futs):
            qid, sessions, err = fut.result()
            done += 1
            if err:
                failed += 1
                print(f"  [{done}/{len(to_label)}] {qid}: FAIL ({err})", flush=True)
            else:
                cache[qid] = sessions
                print(f"  [{done}/{len(to_label)}] {qid}: sessions={sessions}", flush=True)
            # Checkpoint every 20
            if done % 20 == 0:
                CACHE.write_text(json.dumps(cache, indent=2))

    CACHE.write_text(json.dumps(cache, indent=2))
    dt = time.perf_counter() - t0
    print(f"\n[labeler] done: {done - failed}/{len(to_label)} labeled ({failed} failed) in {dt:.0f}s")
    print(f"[labeler] total cached: {len(cache)} → {CACHE}")


if __name__ == "__main__":
    main()
