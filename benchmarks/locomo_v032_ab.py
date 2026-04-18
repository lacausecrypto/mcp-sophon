#!/usr/bin/env python3
"""
LOCOMO v0.3.2 A/B bench — 4-way comparison:

  * V030       — v0.3.0 baseline (block-based LLM summary + vector retrieval)
  * V031       — V030 + SOPHON_HYDE + SOPHON_FACT_CARDS
  * V032_FULL  — V031 + SOPHON_ADAPTIVE + SOPHON_LLM_RERANK
                        + SOPHON_TAIL_SUMMARY + SOPHON_CHUNK_TARGET=500
                        + SOPHON_ENTITY_GRAPH
  * FULL       — raw conversation ceiling

SOPHON_V032_PER_TYPE controls samples per question type. Use 1 for the
10-min sanity pass, 6 for the 60-min publication-directional pass.

Session datetimes are prepended to each message content for all conditions —
this is now the recommended "application side" best practice.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "v032_ab_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLES_PER_TYPE = int(os.environ.get("SOPHON_V032_PER_TYPE", "1"))

SOPHON_CONDS = ["V030", "V031", "V032_FULL"]
CONDITIONS = SOPHON_CONDS + ["FULL"]
HAIKU_CMD = "claude -p --model haiku --output-format json"


def rpc(requests, env=None, timeout=900):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    try:
        p = subprocess.run([SOPHON, "serve"], input=payload, capture_output=True,
                           text=True, timeout=timeout, env=env)
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


INIT = {"jsonrpc":"2.0","id":0,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
                  "clientInfo":{"name":"v032","version":"0"}}}


def call(name, args, rid):
    return {"jsonrpc":"2.0","id":rid,"method":"tools/call","params":{"name":name,"arguments":args}}


def extract(resp):
    if not resp: return None
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res: return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def count_tokens(text):
    out = rpc([INIT, call("count_tokens",{"text":text},1)])
    e = extract(out.get(1)) if out else None
    return e["token_count"] if e else 0


def sophon_compress(messages, query, retriever_path, mode):
    """`mode` is one of V030 / V031 / V032_FULL — maps to env-flag bundle."""
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    if mode in ("V031", "V032_FULL"):
        env["SOPHON_HYDE"] = "1"
        env["SOPHON_FACT_CARDS"] = "1"
    if mode == "V032_FULL":
        env["SOPHON_ADAPTIVE"] = "1"
        env["SOPHON_LLM_RERANK"] = "1"
        env["SOPHON_TAIL_SUMMARY"] = "1"
        env["SOPHON_CHUNK_TARGET"] = "500"
        env["SOPHON_CHUNK_MAX"] = "700"
        env["SOPHON_ENTITY_GRAPH"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=900)
    return extract(out.get(1)) if out else None


def flatten(item):
    """Inject session datetime into each message content. This is the
    application-side practice we standardised on: gold answers of the form
    'last week of October 2023' need the session date attached to each
    chunk to be resolvable by retrieval."""
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""]*len(sessions))
    sophon_msgs = []
    full_lines = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        session_tag = f"[Session {i+1} | {ts}]"
        full_lines.append(session_tag)
        for turn in sess:
            role = turn.get("role","user")
            if role not in ("user","assistant","system"):
                role = "user"
            content = turn.get("content","") or ""
            tagged = f"{session_tag} {content}"
            sophon_msgs.append({"role": role, "content": tagged})
            full_lines.append(f"  {content}")
    return sophon_msgs, "\n".join(full_lines)


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
        r = [f"{m.get('role','user')}: {m.get('content','')}" for m in comp["recent_messages"]]
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
        p = subprocess.run(["claude","-p","--model",model,"--output-format","json"],
                           input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", (time.perf_counter_ns() - t0) / 1e6
    latency_ms = (time.perf_counter_ns() - t0) / 1e6
    try:
        d = json.loads(p.stdout)
        return d.get("result","") or "", latency_ms
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
    prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, candidate=candidate or "(empty)")
    try:
        p = subprocess.run(["claude","-p","--model",JUDGE,"--output-format","json"],
                           input=prompt, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "judge timeout"
    try:
        d = json.loads(p.stdout)
        txt = d.get("result","").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None, "no json"
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False)), parsed.get("rationale","")
    except Exception as e:
        return None, f"parse: {e}"


def run_sophon_condition(name, messages, question):
    d = Path(tempfile.mkdtemp(prefix=f"lv032_{name}_"))
    t0 = time.perf_counter()
    try:
        comp = sophon_compress(messages, question, d, mode=name)
        dt = time.perf_counter() - t0
        if not comp:
            return name, "(compress failed)", {}, dt
        block = build_block(comp)
        rc = comp.get("retrieved_chunks") or {}
        meta = {
            "hyde_rewrites": rc.get("hyde_rewrites"),
            "question_mode": rc.get("question_mode"),
            "react_followups": rc.get("react_followups"),
            "fact_cards": bool(comp.get("fact_cards")),
            "tail_summary": bool(comp.get("tail_summary")),
            "n_chunks": len(rc.get("chunks", [])),
        }
        return name, block, meta, dt
    finally:
        shutil.rmtree(d, ignore_errors=True)


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qtype, items in sorted(by_type.items()):
    random.shuffle(items)
    subset.extend(items[:SAMPLES_PER_TYPE])
print(f"[v032-ab] N={len(subset)} items, {SAMPLES_PER_TYPE}/type, seed=42")

results = []
t_start = time.perf_counter()
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    t0 = time.perf_counter()
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) — {item['question'][:55]}...", flush=True)
    sophon_msgs, full_text = flatten(item)

    blocks = {"FULL": full_text}
    metas = {}
    per_cond_dt = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_sophon_condition, name, sophon_msgs, item["question"]): name
                for name in SOPHON_CONDS}
        for fut in futs:
            name, block, meta, dt = fut.result()
            blocks[name] = block
            metas[name] = meta
            per_cond_dt[name] = round(dt)

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
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in CONDITIONS)
    print(f"   {summary}  dt={dt:.0f}s", flush=True)

(ROOT/"v032_ab_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s", flush=True)

# ---- aggregation ----
import statistics

def pct(k, n): return (100*k/n) if n else 0

def wilson(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5) / d
    return (max(0,(c-h)*100), min(100,(c+h)*100))

print()
print("="*95)
print(f"v0.3.2 A/B — N = {len(results)}")
print("="*95)
TYPES = ["single_hop","multi_hop","temporal_reasoning","open_domain","adversarial"]
print(f"{'Type':<22}" + "".join(f"{c:>14}" for c in CONDITIONS) + f"{'n':>5}")
for qtype in TYPES:
    items = [r for r in results if r["question_type"]==qtype]
    if not items: continue
    line = f"{qtype:<22}"
    for c in CONDITIONS:
        correct = sum(1 for r in items if r["conditions"][c]["correct"] is True)
        line += f"{pct(correct,len(items)):>13.1f}%"
    line += f"{len(items):>5}"
    print(line)
print("-"*95)
line = f"{'GLOBAL':<22}"
totals = {}
for c in CONDITIONS:
    correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    totals[c] = (correct, len(results))
    line += f"{pct(correct,len(results)):>13.1f}%"
line += f"{len(results):>5}"
print(line)

print()
print("95% Wilson CI")
print("-"*95)
for c in CONDITIONS:
    k, n = totals[c]
    lo, hi = wilson(k, n)
    print(f"  {c:<12}  {pct(k,n):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]   ({k}/{n})")

print()
print("Context tokens + Sophon compress latency")
print("-"*95)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    dts = [r.get("compress_dt", {}).get(c) for r in results]
    dts = [d for d in dts if d is not None]
    dt_str = f"compress_dt mean={statistics.mean(dts):.1f}s" if dts else "(no dt)"
    print(f"  {c:<12}  ctx_mean={statistics.mean(vals):>7.0f}  max={max(vals):>7}  {dt_str}")

print()
print(f"saved → {ROOT/'v032_ab_results.json'}")
