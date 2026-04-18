#!/usr/bin/env python3
"""
LOCOMO Path A benchmark — tests the ingest-time-LLM / query-time-zero-LLM
graph memory architecture against the earlier V030 baseline and the V032
full feature stack.

Path A flow (per item):
  1. One `sophon serve` subprocess per item.
  2. Batch messages into groups of 30, send each as `update_memory`
     (return_snapshot=false). Each batch triggers ONE Haiku call that
     extracts (subject, predicate, object) triples.
  3. Final `compress_history` call with the question. Graph memory is
     queried in pure Rust; no LLM call at query time.
  4. The `graph_facts` field of the response is the entire retrieval
     payload used by the answerer — there is no block summary, no
     HashEmbedder/BM25 retrieval, no HyDE. Only the triples.

This isolates the Path A architecture. The intent is NOT to produce the
best possible number; it is to measure what the graph alone can do.

Conditions:
  V030       — current baseline (block LLM summary + vector retrieval)
  V032_FULL  — V031 + Rec #1/#2/#3 flags
  PATH_A     — ingest triples, query graph only
  FULL       — raw conversation ceiling

SOPHON_PATHA_PER_TYPE controls sample count per type (default 3).
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "pathA_ab_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
PER_TYPE = int(os.environ.get("SOPHON_PATHA_PER_TYPE", "3"))
INGEST_BATCH = int(os.environ.get("SOPHON_PATHA_INGEST_BATCH", "30"))

SOPHON_CONDS = ["V030", "V032_FULL", "PATH_A", "PATH_A_COMBINED"]
CONDITIONS = SOPHON_CONDS + ["FULL"]
HAIKU_CMD = "claude -p --model haiku --output-format json"


def rpc(requests, env=None, timeout=1500):
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
    "jsonrpc": "2.0", "id": 0, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "path_a", "version": "0"},
    },
}


def call(name, args, rid):
    return {
        "jsonrpc": "2.0", "id": rid, "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }


def extract(resp):
    if not resp: return None
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res:
        return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def count_tokens(text):
    out = rpc([INIT, call("count_tokens", {"text": text}, 1)])
    e = extract(out.get(1)) if out else None
    return e["token_count"] if e else 0


def flatten(item):
    """Same session-datetime prefix we use in every v0.3.1+ bench so
    the chunks carry temporal context end-to-end."""
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""] * len(sessions))
    msgs = []
    full_lines = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        tag = f"[Session {i+1} | {ts}]"
        full_lines.append(tag)
        for turn in sess:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = turn.get("content", "") or ""
            msgs.append({"role": role, "content": f"{tag} {content}"})
            full_lines.append(f"  {content}")
    return msgs, "\n".join(full_lines)


# ---------- per-condition compressors ----------
def run_v030(messages, question):
    d = Path(tempfile.mkdtemp(prefix="v030_"))
    try:
        env = {
            **os.environ,
            "SOPHON_RETRIEVER_PATH": str(d),
            "SOPHON_LLM_CMD": HAIKU_CMD,
        }
        args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
                "query": question, "retrieval_top_k": 5}
        out = rpc([INIT, call("compress_history", args, 1)], env=env)
        return extract(out.get(1))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_v032(messages, question):
    d = Path(tempfile.mkdtemp(prefix="v032_"))
    try:
        env = {
            **os.environ,
            "SOPHON_RETRIEVER_PATH": str(d),
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
        args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
                "query": question, "retrieval_top_k": 5}
        out = rpc([INIT, call("compress_history", args, 1)], env=env)
        return extract(out.get(1))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_path_a(messages, question):
    """PATH_A (graph only). Disables block-LLM summary and retrieval —
    the downstream answerer only sees graph facts. Isolates the pure
    Path A architecture."""
    env = {
        **os.environ,
        "SOPHON_GRAPH_MEMORY": "1",
        "SOPHON_NO_LLM_SUMMARY": "1",
        "SOPHON_LLM_CMD": HAIKU_CMD,
    }
    return _run_graph_flow(messages, question, env)


def run_path_a_combined(messages, question):
    """PATH_A_COMBINED (graph + retrieval + LLM summary).

    Hypothesis: the 163-token graph-only context was too narrow for
    questions not answerable by a bare triple. Adding the existing
    retrieval + summary pipeline on top should give the answerer the
    best of both: structured facts for recall-heavy queries AND
    verbatim chunks for reasoning-heavy ones.

    Uses V030-style retrieval (no HyDE / FC / rerank) to keep the
    delta attributable to graph facts alone."""
    retriever_path = Path(tempfile.mkdtemp(prefix="pac_retr_"))
    env = {
        **os.environ,
        "SOPHON_GRAPH_MEMORY": "1",
        "SOPHON_RETRIEVER_PATH": str(retriever_path),
        "SOPHON_LLM_CMD": HAIKU_CMD,
    }
    try:
        return _run_graph_flow(messages, question, env)
    finally:
        shutil.rmtree(retriever_path, ignore_errors=True)


def _run_graph_flow(messages, question, env):
    """Shared ingest → query flow. One `update_memory` (rayon parallel
    extraction), then one `compress_history` with the question. The
    env dict decides whether retrieval / summariser are also active."""
    requests = [
        INIT,
        call("update_memory", {
            "messages": messages,
            "return_snapshot": False,
        }, 1),
        call("compress_history", {
            "messages": messages,  # pass through for retriever indexing
            "query": question,
            "max_tokens": 2000,
            "recent_window": 5,
            "retrieval_top_k": 8,
        }, 2),
    ]

    out = rpc(requests, env=env, timeout=1800)
    if not out:
        return None, []

    comp = extract(out.get(2))
    ingest_resp = extract(out.get(1)) or {}
    ingest_summaries = []
    if isinstance(ingest_resp, dict) and ingest_resp.get("graph_ingest"):
        ingest_summaries = [ingest_resp["graph_ingest"]]
    return comp, ingest_summaries


# ---------- context block builders ----------
def build_block(comp):
    parts = []
    if comp.get("summary"):
        parts.append("SUMMARY: " + comp["summary"])
    if comp.get("stable_facts"):
        lines = [f"- {f.get('content','')}" for f in comp["stable_facts"]
                 if isinstance(f, dict) and not f.get("superseded")]
        if lines:
            parts.append("FACTS:\n" + "\n".join(lines))
    fc = comp.get("fact_cards")
    if fc and fc.get("rendered"):
        parts.append("ENTITY TIMELINE:\n" + fc["rendered"].strip())
    ts = comp.get("tail_summary")
    if ts:
        parts.append("TAIL SUMMARY:\n" + ts.strip())
    if comp.get("recent_messages"):
        r = [f"{m.get('role','user')}: {m.get('content','')}" for m in comp["recent_messages"]]
        parts.append("RECENT MESSAGES:\n" + "\n".join(r))
    rc = comp.get("retrieved_chunks", {})
    if rc and rc.get("chunks"):
        lines = [f"[score={sc.get('score',0):.3f}] {sc.get('chunk',{}).get('content','')}"
                 for sc in rc["chunks"]]
        parts.append("RETRIEVED CONTEXT:\n" + "\n".join(lines))
    return "\n\n".join(parts) or "(empty)"


def build_path_a_block(comp):
    """Path A (graph only): context = graph facts + (possibly heuristic)
    summary. Deliberately narrow."""
    parts = []
    gf = comp.get("graph_facts")
    if gf and gf.get("rendered"):
        parts.append("ENTITY FACTS (graph):\n" + gf["rendered"].strip())
    if comp.get("summary"):
        parts.append("SUMMARY: " + comp["summary"])
    return "\n\n".join(parts) or "(empty)"


def build_path_a_combined_block(comp):
    """Path A COMBINED: the full V030 block (summary + facts + recent +
    retrieved chunks) PLUS the graph facts on top. The answerer sees
    both the verbatim conversational context AND the structured entity
    triples — hypothesis is that the combination beats either alone."""
    parts = []
    if comp.get("summary"):
        parts.append("SUMMARY: " + comp["summary"])
    if comp.get("stable_facts"):
        lines = [f"- {f.get('content','')}" for f in comp["stable_facts"]
                 if isinstance(f, dict) and not f.get("superseded")]
        if lines:
            parts.append("FACTS:\n" + "\n".join(lines))
    gf = comp.get("graph_facts")
    if gf and gf.get("rendered"):
        parts.append("ENTITY FACTS (graph):\n" + gf["rendered"].strip())
    if comp.get("recent_messages"):
        r = [f"{m.get('role','user')}: {m.get('content','')}" for m in comp["recent_messages"]]
        parts.append("RECENT MESSAGES:\n" + "\n".join(r))
    rc = comp.get("retrieved_chunks", {})
    if rc and rc.get("chunks"):
        lines = [f"[score={sc.get('score',0):.3f}] {sc.get('chunk',{}).get('content','')}"
                 for sc in rc["chunks"]]
        parts.append("RETRIEVED CONTEXT:\n" + "\n".join(lines))
    return "\n\n".join(parts) or "(empty)"


# ---------- QA + judge ----------
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
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", (time.perf_counter_ns() - t0) / 1e6
    lat = (time.perf_counter_ns() - t0) / 1e6
    try:
        d = json.loads(p.stdout)
        return d.get("result", "") or "", lat
    except Exception:
        return p.stdout, lat


JUDGE_TEMPLATE = """You are scoring a QA system. Compare the candidate answer to the gold answer for the given question. Mark as correct if the candidate conveys the same factual answer as the gold (paraphrase is fine). Mark as wrong if the candidate is missing, contradicts the gold, or says "I don't know" when the gold has a concrete answer.

QUESTION: {question}

GOLD ANSWER: {gold}

CANDIDATE ANSWER: {candidate}

Respond with ONE line of strict JSON, nothing else:
{{"correct": true|false, "rationale": "<one short sentence>"}}
"""


def judge_correctness(question, gold, candidate):
    prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, candidate=candidate or "(empty)")
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", JUDGE, "--output-format", "json"],
            input=prompt, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    try:
        d = json.loads(p.stdout)
        txt = d.get("result", "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False))
    except Exception:
        return None


def _accumulate_graph_meta(meta, comp, ingest_summaries):
    if ingest_summaries:
        meta["ingest_calls"] = len(ingest_summaries)
        last = ingest_summaries[-1]
        if isinstance(last, dict):
            meta["fact_total"] = last.get("fact_total")
            meta["entity_total"] = last.get("entity_total")
        meta["triples_seen"] = sum(
            s.get("triples_seen", 0) for s in ingest_summaries if isinstance(s, dict)
        )
    if comp and comp.get("graph_facts"):
        meta["graph_fact_count"] = comp["graph_facts"].get("fact_count")


def run_condition(name, messages, question):
    t0 = time.perf_counter()
    meta = {}
    if name == "V030":
        comp = run_v030(messages, question)
        block = build_block(comp) if comp else "(compress failed)"
    elif name == "V032_FULL":
        comp = run_v032(messages, question)
        block = build_block(comp) if comp else "(compress failed)"
    elif name == "PATH_A":
        comp, ingest_summaries = run_path_a(messages, question)
        block = build_path_a_block(comp) if comp else "(compress failed)"
        _accumulate_graph_meta(meta, comp, ingest_summaries)
    elif name == "PATH_A_COMBINED":
        comp, ingest_summaries = run_path_a_combined(messages, question)
        block = build_path_a_combined_block(comp) if comp else "(compress failed)"
        _accumulate_graph_meta(meta, comp, ingest_summaries)
    else:
        raise ValueError(name)
    dt = time.perf_counter() - t0
    return name, block, meta, dt


# ---------- main ----------
all_items = [json.loads(l) for l in open(ROOT / "all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qt, items in sorted(by_type.items()):
    random.shuffle(items)
    subset.extend(items[:PER_TYPE])
print(f"[pathA-ab] N={len(subset)} items, {PER_TYPE}/type, seed=42, ingest_batch={INGEST_BATCH}")
print(f"[pathA-ab] conditions: {CONDITIONS}")

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
    sophon_msgs, full_text = flatten(item)

    blocks = {"FULL": full_text}
    metas = {}
    compress_dts = {}

    # Run Sophon conditions in parallel so wall-clock matches `max`
    # rather than `sum`.
    with ThreadPoolExecutor(max_workers=len(SOPHON_CONDS)) as pool:
        futs = {
            pool.submit(run_condition, name, sophon_msgs, item["question"]): name
            for name in SOPHON_CONDS
        }
        for fut in futs:
            name, block, meta, dt = fut.result()
            blocks[name] = block
            metas[name] = meta
            compress_dts[name] = round(dt)

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "metas": metas,
        "compress_dt": compress_dts,
        "conditions": {},
    }
    for cond in CONDITIONS:
        ctx = blocks[cond]
        prompt = build_prompt(ctx, item["question"])
        answer, lat = call_claude(prompt)
        correct = judge_correctness(item["question"], item["answer"], answer)
        row["conditions"][cond] = {
            "ctx_tokens": count_tokens(ctx),
            "latency_ms": round(lat),
            "answer": answer[:400],
            "correct": correct,
        }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    dt = time.perf_counter() - t0
    summary = " ".join(
        f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}"
        for c in CONDITIONS
    )
    pa_meta = metas.get("PATH_A_COMBINED") or metas.get("PATH_A") or {}
    extra = f" facts={pa_meta.get('fact_total', '?')}" if pa_meta else ""
    print(f"   {summary}{extra}  dt={dt:.0f}s", flush=True)

(ROOT / "pathA_ab_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter() - t_start:.0f}s", flush=True)

# ---------- aggregation ----------
import statistics


def pct(k, n): return (100 * k / n) if n else 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0, (c - h) * 100), min(100, (c + h) * 100))


TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]
print()
print("=" * 100)
print(f"PATH A A/B — N = {len(results)}")
print("=" * 100)
header = f"{'Type':<22}" + "".join(f"{c:>14}" for c in CONDITIONS) + f"{'n':>5}"
print(header)
for qt in TYPES:
    items = [r for r in results if r["question_type"] == qt]
    if not items:
        continue
    line = f"{qt:<22}"
    for c in CONDITIONS:
        k = sum(1 for r in items if r["conditions"][c]["correct"] is True)
        line += f"{pct(k, len(items)):>13.1f}%"
    line += f"{len(items):>5}"
    print(line)
print("-" * 100)
line = f"{'GLOBAL':<22}"
totals = {}
for c in CONDITIONS:
    k = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    totals[c] = (k, len(results))
    line += f"{pct(k, len(results)):>13.1f}%"
line += f"{len(results):>5}"
print(line)

print()
print("95% Wilson CI")
print("-" * 100)
for c in CONDITIONS:
    k, n = totals[c]
    lo, hi = wilson(k, n)
    print(f"  {c:<12}  {pct(k, n):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]    ({k}/{n})")

print()
print("Context tokens (mean) + compress latency (mean)")
print("-" * 100)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    dts = [r.get("compress_dt", {}).get(c) for r in results]
    dts = [d for d in dts if d is not None]
    dt_str = f"compress_dt={statistics.mean(dts):.1f}s" if dts else "n/a"
    print(f"  {c:<12}  ctx_mean={statistics.mean(vals):>7.0f}  max={max(vals):>7}  {dt_str}")

print()
for lab in ("PATH_A", "PATH_A_COMBINED"):
    items_with = [r for r in results if r["metas"].get(lab)]
    if not items_with:
        continue
    avg_facts = statistics.mean(
        r["metas"][lab].get("fact_total", 0) for r in items_with
    )
    avg_entities = statistics.mean(
        r["metas"][lab].get("entity_total", 0) for r in items_with
    )
    avg_seen = statistics.mean(
        r["metas"][lab].get("triples_seen", 0) for r in items_with
    )
    avg_graph_ctx = statistics.mean(
        r["metas"][lab].get("graph_fact_count", 0) or 0 for r in items_with
    )
    print(
        f"{lab:<18} graph stats: facts={avg_facts:.0f}  entities={avg_entities:.0f}"
        f"  triples_seen={avg_seen:.0f}  graph_ctx_facts={avg_graph_ctx:.1f}"
    )

print(f"\nsaved → {ROOT / 'pathA_ab_results.json'}")
