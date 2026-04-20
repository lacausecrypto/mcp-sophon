#!/usr/bin/env python3
"""
LOCOMO v5 vs v4 A/B — measures the impact of the newly-added flag-gated
features on top of the shipped v0.4.0 (V032_FULL) baseline.

Conditions
  v4  = V032_FULL env bundle (SOPHON_HYDE + SOPHON_FACT_CARDS + SOPHON_ADAPTIVE
                               + SOPHON_LLM_RERANK + SOPHON_TAIL_SUMMARY
                               + SOPHON_CHUNK_TARGET=500 + SOPHON_ENTITY_GRAPH)
  v5  = v4 + SOPHON_ENTITY_WEIGHTED=1  (Pick #5)
           + SOPHON_CHUNK_ENTITY_AWARE=1 (Pick #6)

Other picks:
  * Picks #2 (parallelized V032 retrieval) and #7 (ASCII NER fast path) are
    always-on in the current release binary — BOTH conditions benefit.
    Latency improvements here are vs the original sequential v0.4.0 code,
    not vs the v4 bar in this file.
  * Pick #1 (SOPHON_EXTRACTIVE) replaces the LLM block summary with a
    zero-LLM extractive path — it's a different trade-off and belongs in
    its own bench, not stacked here.
  * Pick #3 (fragment cache) is prompt-level, not compress_history-level.
  * Pick #4 (count_tokens_batch) is a new tool, no behavior change.

SOPHON_V5_PER_TYPE controls samples per question type. Default 10 → N=50.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "v5_vs_v4_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get(
    "SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon",
)
MODEL = os.environ.get("SOPHON_BENCH_MODEL", "sonnet")
JUDGE = os.environ.get("SOPHON_BENCH_JUDGE", "sonnet")
SAMPLES_PER_TYPE = int(os.environ.get("SOPHON_V5_PER_TYPE", "10"))
CONDITIONS = ["v4", "v5"]
HAIKU_CMD = "claude -p --model haiku --output-format json"


def rpc(requests, env=None, timeout=900):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    try:
        p = subprocess.run(
            [SOPHON, "serve"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {}
    out = {}
    for line in p.stdout.splitlines():
        if line.strip():
            try:
                d = json.loads(line)
                out[d.get("id")] = d
            except json.JSONDecodeError:
                continue
    return out


INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "v5_vs_v4", "version": "0"},
    },
}


def call(name, args, rid):
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }


def extract(resp):
    if not resp:
        return None
    res = resp.get("result")
    if not res:
        return None
    if "structuredContent" in res:
        return res["structuredContent"]
    try:
        return json.loads(res["content"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def count_tokens(text):
    out = rpc([INIT, call("count_tokens", {"text": text}, 1)])
    e = extract(out.get(1)) if out else None
    return e["token_count"] if e else 0


def base_v032_env(retriever_path):
    return {
        **os.environ,
        "SOPHON_RETRIEVER_PATH": str(retriever_path),
        "SOPHON_LLM_CMD": HAIKU_CMD,
        "SOPHON_HYDE": "1",
        "SOPHON_FACT_CARDS": "1",
        "SOPHON_ADAPTIVE": "1",
        "SOPHON_LLM_RERANK": "1",
        "SOPHON_TAIL_SUMMARY": "1",
        "SOPHON_CHUNK_TARGET": "500",
        "SOPHON_CHUNK_MAX": "700",
        "SOPHON_ENTITY_GRAPH": "1",
    }


def sophon_compress(messages, query, retriever_path, mode):
    args = {
        "messages": messages,
        "recent_window": 5,
        "max_tokens": 2000,
        "query": query,
        "retrieval_top_k": 5,
    }
    env = base_v032_env(retriever_path)
    if mode == "v5":
        env["SOPHON_ENTITY_WEIGHTED"] = "1"
        env["SOPHON_CHUNK_ENTITY_AWARE"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=900)
    return extract(out.get(1)) if out else None


def flatten(item):
    """Inject session datetime into each message content — same practice as
    locomo_v032_ab.py so gold answers that mention 'October 2023' resolve."""
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""] * len(sessions))
    sophon_msgs = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        session_tag = f"[Session {i+1} | {ts}]"
        for turn in sess:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = turn.get("content", "") or ""
            sophon_msgs.append({"role": role, "content": f"{session_tag} {content}"})
    return sophon_msgs


def build_block(comp):
    parts = []
    if comp.get("summary"):
        parts.append("SUMMARY: " + comp["summary"])
    if comp.get("stable_facts"):
        lines = []
        for f in comp["stable_facts"]:
            if isinstance(f, dict) and not f.get("superseded"):
                lines.append(f"- {f.get('content','')}")
        if lines:
            parts.append("FACTS:\n" + "\n".join(lines))
    fc = comp.get("fact_cards")
    if fc and fc.get("rendered"):
        parts.append("ENTITY TIMELINE:\n" + fc["rendered"].strip())
    ts = comp.get("tail_summary")
    if ts:
        parts.append("TAIL SUMMARY:\n" + ts.strip())
    if comp.get("recent_messages"):
        r = [
            f"{m.get('role','user')}: {m.get('content','')}"
            for m in comp["recent_messages"]
        ]
        parts.append("RECENT MESSAGES:\n" + "\n".join(r))
    rc = comp.get("retrieved_chunks", {})
    if rc and rc.get("chunks"):
        retrieved_lines = []
        for sc in rc["chunks"]:
            chunk = sc.get("chunk", {})
            score = sc.get("score", 0.0)
            retrieved_lines.append(f"[score={score:.3f}] {chunk.get('content','')}")
        parts.append("RETRIEVED CONTEXT:\n" + "\n".join(retrieved_lines))
    return "\n\n".join(parts) or "(empty)"


def build_prompt(context_block, q):
    return f"""You are answering a question about a long conversation.

CONTEXT:
{context_block}

QUESTION: {q}

Answer concisely. If the context does not contain enough information to answer, say "I don't know."
"""


def call_claude(prompt, model=MODEL, timeout=180):
    t0 = time.perf_counter_ns()
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", (time.perf_counter_ns() - t0) / 1e6
    latency_ms = (time.perf_counter_ns() - t0) / 1e6
    try:
        d = json.loads(p.stdout)
        return d.get("result", "") or "", latency_ms
    except Exception:
        return p.stdout, latency_ms


JUDGE_TEMPLATE = """You are scoring a QA system. Compare the candidate answer to the gold answer for the given question. Mark as correct if the candidate conveys the same factual answer as the gold (paraphrase is fine). Mark as wrong if the candidate is missing, contradicts the gold, or says "I don't know" when the gold has a concrete answer.

QUESTION: {question}

GOLD ANSWER: {gold}

CANDIDATE ANSWER: {candidate}

Respond with ONE line of strict JSON, nothing else:
{{"correct": true|false, "rationale": "<one short sentence>"}}
"""


def judge_correctness(question, gold, candidate):
    prompt = JUDGE_TEMPLATE.format(
        question=question, gold=gold, candidate=candidate or "(empty)"
    )
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", JUDGE, "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "judge timeout"
    try:
        d = json.loads(p.stdout)
        txt = d.get("result", "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None, "no json"
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False)), parsed.get("rationale", "")
    except Exception as e:
        return None, f"parse: {e}"


def run_condition(cond, messages, question):
    d = Path(tempfile.mkdtemp(prefix=f"l5v4_{cond}_"))
    t0 = time.perf_counter()
    try:
        comp = sophon_compress(messages, question, d, mode=cond)
        dt = time.perf_counter() - t0
        if not comp:
            return cond, "(compress failed)", {}, dt
        block = build_block(comp)
        rc = comp.get("retrieved_chunks") or {}
        meta = {
            "n_chunks": len(rc.get("chunks", [])),
            "question_mode": rc.get("question_mode"),
            "hyde_rewrites": rc.get("hyde_rewrites"),
            "retrieval_confidence": comp.get("retrieval_confidence"),
        }
        return cond, block, meta, dt
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    all_items = [json.loads(l) for l in open(ROOT / "all_items.jsonl")]
    by_type = {}
    for it in all_items:
        by_type.setdefault(it["question_type"], []).append(it)

    subset = []
    for qtype, items in sorted(by_type.items()):
        random.shuffle(items)
        subset.extend(items[:SAMPLES_PER_TYPE])
    print(
        f"[v5-vs-v4] N={len(subset)} items, {SAMPLES_PER_TYPE}/type, seed=42, model={MODEL}",
        flush=True,
    )

    results = []
    t_start = time.perf_counter()
    for idx, item in enumerate(subset):
        cache = RUNS / f"{item['question_id']}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text()))
            print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
            continue

        t0 = time.perf_counter()
        print(
            f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) — {item['question'][:55]}...",
            flush=True,
        )
        sophon_msgs = flatten(item)

        blocks, metas, per_cond_dt = {}, {}, {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {
                pool.submit(run_condition, c, sophon_msgs, item["question"]): c
                for c in CONDITIONS
            }
            for fut in futs:
                cond, block, meta, dt = fut.result()
                blocks[cond] = block
                metas[cond] = meta
                per_cond_dt[cond] = round(dt, 2)

        row = {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "gold": item["answer"],
            "metas": metas,
            "compress_dt": per_cond_dt,
            "conditions": {},
        }
        for cond in CONDITIONS:
            ctx = blocks[cond]
            prompt = build_prompt(ctx, item["question"])
            answer, lat = call_claude(prompt)
            correct, rationale = judge_correctness(item["question"], item["answer"], answer)
            row["conditions"][cond] = {
                "ctx_tokens": count_tokens(ctx),
                "latency_ms": round(lat),
                "answer": answer[:400],
                "correct": correct,
                "rationale": rationale,
            }
        cache.write_text(json.dumps(row, indent=2))
        results.append(row)
        dt = time.perf_counter() - t0
        summary = " ".join(
            f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}"
            for c in CONDITIONS
        )
        print(f"   {summary}  dt={dt:.0f}s", flush=True)

    (ROOT / "v5_vs_v4_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s", flush=True)

    # ---- aggregation ----
    import statistics

    def pct(k, n):
        return (100 * k / n) if n else 0

    def wilson(k, n, z=1.96):
        if n == 0:
            return (0, 0)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return (max(0, (c - h) * 100), min(100, (c + h) * 100))

    print()
    print("=" * 80)
    print(f"LOCOMO  v5 vs v4  —  N = {len(results)}")
    print("=" * 80)
    TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]
    print(f"{'Type':<22}" + "".join(f"{c:>14}" for c in CONDITIONS) + f"{'Δ':>10}{'n':>5}")
    for qtype in TYPES:
        items = [r for r in results if r["question_type"] == qtype]
        if not items:
            continue
        scores = {}
        for c in CONDITIONS:
            correct = sum(1 for r in items if r["conditions"][c]["correct"] is True)
            scores[c] = correct
        line = f"{qtype:<22}"
        for c in CONDITIONS:
            line += f"{pct(scores[c], len(items)):>13.1f}%"
        delta = pct(scores["v5"], len(items)) - pct(scores["v4"], len(items))
        line += f"{delta:>+9.1f}%{len(items):>5}"
        print(line)
    print("-" * 80)
    totals = {}
    line = f"{'GLOBAL':<22}"
    for c in CONDITIONS:
        correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
        totals[c] = (correct, len(results))
        line += f"{pct(correct, len(results)):>13.1f}%"
    delta = pct(totals["v5"][0], len(results)) - pct(totals["v4"][0], len(results))
    line += f"{delta:>+9.1f}%{len(results):>5}"
    print(line)

    print()
    print("95% Wilson CI")
    print("-" * 80)
    for c in CONDITIONS:
        k, n = totals[c]
        lo, hi = wilson(k, n)
        print(f"  {c:<6}  {pct(k,n):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]   ({k}/{n})")

    print()
    print("Context tokens  +  Sophon compress latency")
    print("-" * 80)
    for c in CONDITIONS:
        vals = [r["conditions"][c]["ctx_tokens"] for r in results]
        dts = [r.get("compress_dt", {}).get(c) for r in results]
        dts = [d for d in dts if d is not None]
        if dts:
            dts_sorted = sorted(dts)
            p50 = dts_sorted[len(dts_sorted) // 2]
            p95 = dts_sorted[min(len(dts_sorted) - 1, int(len(dts_sorted) * 0.95))]
            dt_line = f"p50={p50:.1f}s p95={p95:.1f}s mean={statistics.mean(dts):.1f}s"
        else:
            dt_line = "(no dt)"
        print(
            f"  {c:<6}  ctx_mean={statistics.mean(vals):>7.0f}  max={max(vals):>7}  {dt_line}"
        )

    # Paired delta latency (v5 - v4) across same items
    pairs = [
        (r["compress_dt"].get("v4"), r["compress_dt"].get("v5"))
        for r in results
        if r.get("compress_dt", {}).get("v4") is not None
        and r.get("compress_dt", {}).get("v5") is not None
    ]
    if pairs:
        deltas = [b - a for a, b in pairs]
        print(
            f"\n  latency Δ (v5 - v4): mean={statistics.mean(deltas):+.2f}s median={sorted(deltas)[len(deltas)//2]:+.2f}s"
        )

    print()
    print(f"saved → {ROOT/'v5_vs_v4_results.json'}")


if __name__ == "__main__":
    main()
