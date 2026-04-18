#!/usr/bin/env python3
"""
LOCOMO ReAct multi-hop bench — targeted evaluation of the iterative-retrieval
fix on the question type that stayed stuck at 0 % across every prior variant.

Conditions (Sophon-side, 3 × multihop items only):
  * V030       — v0.3.0 baseline (block LLM summary + vector retrieval)
  * V031       — HyDE + FactCards (no iterative refinement)
  * V031+RE    — V031 + SOPHON_REACT=1 (up to 3 retrieval rounds)
  * FULL       — raw conversation ceiling

Multi-hop is the sharpest test of iterative refinement: single-pass retrieval
systematically misses chunks that share an entity but not the query's
vocabulary. If ReAct works, it should close part of the 0 % → 100 % gap
(FULL scores near 100 % on multi-hop at every N we've measured).
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "react_multihop_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLE_N = int(os.environ.get("SOPHON_REACT_N", "12"))

SOPHON_CONDS = ["V030", "V031", "V031+RE"]
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
                  "clientInfo":{"name":"react","version":"0"}}}


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


def sophon_compress(messages, query, retriever_path, hyde=False, fact_cards=False, react=False):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    if hyde:       env["SOPHON_HYDE"] = "1"
    if fact_cards: env["SOPHON_FACT_CARDS"] = "1"
    if react:      env["SOPHON_REACT"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=900)
    return extract(out.get(1)) if out else None


def flatten(item):
    """Flatten sessions → (sophon_msgs, full_text).

    Session datetime is prepended to each message content so it survives
    chunking and shows up in retrieved chunks. Without this, gold answers
    of the form "last week of October 2023" are unresolvable by the
    retrieval path — Sonnet only has the chunk text, not when it was
    uttered, and defaults to the current date as reference.
    """
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
    flags = {
        "V030":    dict(hyde=False, fact_cards=False, react=False),
        "V031":    dict(hyde=True,  fact_cards=True,  react=False),
        "V031+RE": dict(hyde=True,  fact_cards=True,  react=True),
    }[name]
    d = Path(tempfile.mkdtemp(prefix=f"lre_{name}_"))
    try:
        comp = sophon_compress(messages, question, d, **flags)
        if not comp:
            return name, "(compress failed)", {}
        block = build_block(comp)
        rc = comp.get("retrieved_chunks") or {}
        meta = {
            "hyde_rewrites": rc.get("hyde_rewrites"),
            "react_followups": rc.get("react_followups"),
            "n_chunks": len(rc.get("chunks", [])),
        }
        return name, block, meta
    finally:
        shutil.rmtree(d, ignore_errors=True)


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
mh_items = [it for it in all_items if it["question_type"] == "multi_hop"]
random.shuffle(mh_items)
subset = mh_items[:SAMPLE_N]
print(f"[react-multihop] N={len(subset)} items, seed=42")

results = []
t_start = time.perf_counter()
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    t0 = time.perf_counter()
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} — {item['question'][:65]}...", flush=True)
    sophon_msgs, full_text = flatten(item)

    blocks = {"FULL": full_text}
    metas = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_sophon_condition, name, sophon_msgs, item["question"]): name
                for name in SOPHON_CONDS}
        for fut in futs:
            name, block, meta = fut.result()
            blocks[name] = block
            metas[name] = meta

    row = {
        "question_id": item["question_id"],
        "question": item["question"],
        "gold": item["answer"],
        "metas": metas,
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
    follow_n = len(metas.get("V031+RE", {}).get("react_followups") or [])
    print(f"   {summary}  react_followups={follow_n}  ({dt:.0f}s)")

(ROOT/"react_multihop_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s")

# -------- aggregation ---------
import statistics

def pct(k, n): return (100*k/n) if n else 0

def wilson(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/den
    return (max(0,(c-h)*100), min(100,(c+h)*100))

print()
print("="*80)
print(f"ReAct multi-hop — N = {len(results)}")
print("="*80)
print(f"{'Cond':<10}{'acc':>8}  95% CI              n correct  follows")
for c in CONDITIONS:
    k = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    n = len(results)
    lo, hi = wilson(k, n)
    if c == "V031+RE":
        avg_follows = statistics.mean(
            (len(r["metas"].get("V031+RE", {}).get("react_followups") or [])) for r in results
        )
        print(f"{c:<10}{pct(k,n):>7.1f}%  [{lo:>5.1f} — {hi:>5.1f}]    {k}/{n}      {avg_follows:.1f}")
    else:
        print(f"{c:<10}{pct(k,n):>7.1f}%  [{lo:>5.1f} — {hi:>5.1f}]    {k}/{n}")

print()
print("Context tokens (mean / max)")
print("-"*80)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    print(f"  {c:<10}  mean={statistics.mean(vals):>7.0f}  max={max(vals):>7}")

print()
print(f"saved → {ROOT/'react_multihop_results.json'}")
