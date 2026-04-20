#!/usr/bin/env python3
"""
LOCOMO v0.4.0 vs v0.5.0 — Axe A (accuracy end-to-end).

Conditions
  v4  — V032_FULL env bundle MINUS SOPHON_HYBRID (reproduces the original
        v0.4.0 bench setup where hybrid BM25 was opt-in but rarely set).
        Fact cards extracted at *query time* (SOPHON_FACT_CARDS=1). This is
        what a v0.4.0 user who didn't manually enable hybrid would see.
  v5  — v0.5.0 defaults: SOPHON_HYBRID + SOPHON_ENTITY_GRAPH on by default
        (flipped in this release), plus SOPHON_FACT_CARDS_PATH for
        ingest-time extraction. The `update_memory` call populates the
        fact-cards store once; `compress_history` serves from the store
        with zero LLM calls on that path.

Hypothesis: v5 > v4 on accuracy driven by hybrid default (+18 pt hit@10 at
the retrieval oracle); ctx tokens and latency similar (hybrid is ~free,
ingest-time fact_cards pays the same Haiku cost as query-time on a
1-query-per-item bench like LOCOMO).

SOPHON_V050_PER_TYPE controls items per question type (default 10 → N=50).
Results cached to `v040_vs_v050_runs/` so interrupted runs resume cleanly.
"""
import json, os, re, shutil, subprocess, tempfile, time, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import random
random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "v040_vs_v050_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get(
    "SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon",
)
MODEL = os.environ.get("SOPHON_BENCH_MODEL", "sonnet")
JUDGE = os.environ.get("SOPHON_BENCH_JUDGE", "sonnet")
PER_TYPE = int(os.environ.get("SOPHON_V050_PER_TYPE", "10"))
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
        "clientInfo": {"name": "v040_v050", "version": "0"},
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


def env_v4(retriever_path):
    """v0.4.0-manual: V032_FULL bundle WITHOUT hybrid (explicit =0 to be
    sure no default leaks through). Fact cards at query time."""
    return {
        **os.environ,
        "SOPHON_RETRIEVER_PATH": str(retriever_path),
        "SOPHON_LLM_CMD": HAIKU_CMD,
        "SOPHON_HYBRID": "0",  # explicit off — mirror original v0.4.0 defaults
        "SOPHON_ENTITY_GRAPH": "1",
        "SOPHON_HYDE": "1",
        "SOPHON_FACT_CARDS": "1",  # query-time extraction
        "SOPHON_ADAPTIVE": "1",
        "SOPHON_LLM_RERANK": "1",
        "SOPHON_TAIL_SUMMARY": "1",
        "SOPHON_CHUNK_TARGET": "500",
        "SOPHON_CHUNK_MAX": "700",
    }


def env_v5(retriever_path, fact_cards_path):
    """v0.5.0: new defaults (hybrid + entity_graph implicit on). Fact cards
    are extracted *inside compress_history* (rayon parallel with retrieval,
    matching v0.4.0's latency profile) but ALSO persisted to a disk store
    via SOPHON_FACT_CARDS_PATH so subsequent queries on the same session
    hit the cache and skip the Haiku call.

    Note: SOPHON_FACT_CARDS=1 is kept ON (same as v4) — it's the signal to
    extract fact cards. The new SOPHON_FACT_CARDS_PATH just adds
    persistence on top."""
    return {
        **os.environ,
        "SOPHON_RETRIEVER_PATH": str(retriever_path),
        "SOPHON_LLM_CMD": HAIKU_CMD,
        # SOPHON_HYBRID + SOPHON_ENTITY_GRAPH are now default-on → no set
        "SOPHON_HYDE": "1",
        "SOPHON_FACT_CARDS": "1",  # inline extract, parallel with retrieval
        "SOPHON_ADAPTIVE": "1",
        "SOPHON_LLM_RERANK": "1",
        "SOPHON_TAIL_SUMMARY": "1",
        "SOPHON_CHUNK_TARGET": "500",
        "SOPHON_CHUNK_MAX": "700",
        "SOPHON_FACT_CARDS_PATH": str(fact_cards_path),  # persistent store
    }


def flatten(item):
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""] * len(sessions))
    sophon_msgs = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        tag = f"[Session {i+1} | {ts}]"
        for turn in sess:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = turn.get("content", "") or ""
            sophon_msgs.append({"role": role, "content": f"{tag} {content}"})
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


def run_v4(messages, question):
    """v4: single compress_history call with query-time fact_cards."""
    d = Path(tempfile.mkdtemp(prefix="lbv4_"))
    t0 = time.perf_counter()
    try:
        env = env_v4(d)
        out = rpc(
            [
                INIT,
                call(
                    "compress_history",
                    {
                        "messages": messages,
                        "recent_window": 5,
                        "max_tokens": 2000,
                        "query": question,
                        "retrieval_top_k": 5,
                    },
                    1,
                ),
            ],
            env=env,
            timeout=900,
        )
        dt = time.perf_counter() - t0
        comp = extract(out.get(1))
        meta = {}
        if comp:
            rc = comp.get("retrieved_chunks") or {}
            meta = {
                "n_chunks": len(rc.get("chunks", [])),
                "fact_cards_source": (comp.get("fact_cards") or {}).get("source", "query_time_extract"),
            }
        block = build_block(comp) if comp else "(compress failed)"
        return block, meta, dt
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_v5(messages, question):
    """v5: a single compress_history call. The refactored handler runs
    fact-card extraction *inside* compress_history in a rayon closure
    concurrent with retrieval (so latency = v4 profile), and self-populates
    the persistent store so future queries on the same session hit the
    cache. On a 1-query-per-item bench we don't see the caching benefit,
    only the parity-with-v4 latency — the point here is to confirm no
    regression from the new architecture."""
    d = Path(tempfile.mkdtemp(prefix="lbv5_"))
    store_path = d / "facts.json"
    (d / "retr").mkdir()
    t0 = time.perf_counter()
    try:
        env = env_v5(d / "retr", store_path)
        out = rpc(
            [
                INIT,
                call(
                    "compress_history",
                    {
                        "messages": messages,
                        "recent_window": 5,
                        "max_tokens": 2000,
                        "query": question,
                        "retrieval_top_k": 5,
                    },
                    1,
                ),
            ],
            env=env,
            timeout=900,
        )
        dt = time.perf_counter() - t0
        comp = extract(out.get(1))
        meta = {}
        if comp:
            rc = comp.get("retrieved_chunks") or {}
            fc = comp.get("fact_cards") or {}
            meta = {
                "n_chunks": len(rc.get("chunks", [])),
                "fact_cards_source": fc.get("source", "none"),
                "fact_cards_entity_count": fc.get("entity_count", 0),
            }
        block = build_block(comp) if comp else "(compress failed)"
        return block, meta, dt
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_condition(cond, messages, question):
    if cond == "v4":
        return cond, *run_v4(messages, question)
    else:
        return cond, *run_v5(messages, question)


def main():
    all_items = [json.loads(l) for l in open(ROOT / "all_items.jsonl")]
    by_type = {}
    for it in all_items:
        by_type.setdefault(it["question_type"], []).append(it)

    subset = []
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype][:]
        random.shuffle(items)
        subset.extend(items[:PER_TYPE])
    print(
        f"[v040-v050] N={len(subset)} items, {PER_TYPE}/type, seed=42, model={MODEL}",
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

    (ROOT / "v040_vs_v050_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s", flush=True)

    # ---- aggregation ----
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
    print("=" * 86)
    print(f"LOCOMO  v0.4.0 vs v0.5.0  —  N = {len(results)}  (Axe A — accuracy e2e)")
    print("=" * 86)
    TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]
    print(f"{'Type':<22}{'v4':>12}{'v5':>12}{'Δ':>10}{'n':>5}")
    totals = {}
    for qtype in TYPES:
        items = [r for r in results if r["question_type"] == qtype]
        if not items:
            continue
        scores = {}
        for c in CONDITIONS:
            correct = sum(1 for r in items if r["conditions"][c]["correct"] is True)
            scores[c] = correct
        delta = pct(scores["v5"], len(items)) - pct(scores["v4"], len(items))
        print(
            f"{qtype:<22}{pct(scores['v4'],len(items)):>11.1f}%{pct(scores['v5'],len(items)):>11.1f}%{delta:>+9.1f}%{len(items):>5}"
        )
    print("-" * 86)
    for c in CONDITIONS:
        correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
        totals[c] = (correct, len(results))
    delta = pct(totals["v5"][0], len(results)) - pct(totals["v4"][0], len(results))
    print(
        f"{'GLOBAL':<22}{pct(totals['v4'][0],len(results)):>11.1f}%{pct(totals['v5'][0],len(results)):>11.1f}%{delta:>+9.1f}%{len(results):>5}"
    )

    print()
    print("95% Wilson CI")
    print("-" * 86)
    for c in CONDITIONS:
        k, n = totals[c]
        lo, hi = wilson(k, n)
        print(f"  {c:<6}  {pct(k,n):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]   ({k}/{n})")

    # McNemar exact for paired significance
    from math import comb
    b = sum(1 for r in results if r["conditions"]["v4"]["correct"] and not r["conditions"]["v5"]["correct"])
    c = sum(1 for r in results if not r["conditions"]["v4"]["correct"] and r["conditions"]["v5"]["correct"])
    print(f"\nMcNemar exact (paired): b={b} (v4-only), c={c} (v5-only)")
    n_pair = b + c
    if n_pair > 0:
        k = min(b, c)
        p = 2 * sum(comb(n_pair, i) for i in range(k + 1)) / (2 ** n_pair)
        print(f"  two-sided p = {p:.3f}  → {'NOT significant' if p > 0.05 else 'SIGNIFICANT'}")

    print()
    print("Context tokens + Sophon compress latency")
    print("-" * 86)
    for c in CONDITIONS:
        vals = [r["conditions"][c]["ctx_tokens"] for r in results]
        dts = [r.get("compress_dt", {}).get(c) for r in results]
        dts = [d for d in dts if d is not None]
        dts_sorted = sorted(dts) if dts else []
        p50 = dts_sorted[len(dts_sorted) // 2] if dts_sorted else 0
        p95 = dts_sorted[min(len(dts_sorted) - 1, int(len(dts_sorted) * 0.95))] if dts_sorted else 0
        print(
            f"  {c:<6}  ctx_mean={statistics.mean(vals):>7.0f}  max={max(vals):>7}  "
            f"p50={p50:.1f}s p95={p95:.1f}s mean={statistics.mean(dts) if dts else 0:.1f}s"
        )

    # Fact cards source breakdown (v5 only)
    print()
    print("v5 fact_cards source distribution:")
    from collections import Counter
    sources = Counter(r["metas"].get("v5", {}).get("fact_cards_source") for r in results)
    for k, v in sources.most_common():
        print(f"  {k!s:<25}  {v}")

    print(f"\nsaved → {ROOT/'v040_vs_v050_results.json'}")


if __name__ == "__main__":
    main()
