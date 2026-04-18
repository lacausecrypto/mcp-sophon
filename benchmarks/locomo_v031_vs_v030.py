#!/usr/bin/env python3
"""
LOCOMO v0.3.1 vs v0.3.0 — direct apples-to-apples bench in the SAME setup
as benchmarks/locomo_retrieval.py (BENCHMARK.md §3.7).

3 conditions only:
  * V030       — current v0.3.0 behaviour: block-based LLM summary + retrieval
  * V031       — v0.3.1 candidate: same + SOPHON_HYDE=1 + SOPHON_FACT_CARDS=1
  * FULL       — raw conversation ceiling

Same LLM command as the public bench (Haiku for Sophon LLM calls, Sonnet
for QA and judge). Same stratified sampling, same seed.

No SOPHON_NO_LLM_SUMMARY flag — the v0.3.0 summariser runs for real so
numbers are directly comparable to the published §3.7 table.

Result interpretation:
  * If V031 accuracy > V030 by > a couple of points → HyDE+FC is a net win
    and can ship as the new default.
  * Per-type breakdown shows where the gain (or loss) concentrates.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "v031_vs_v030_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLES_PER_TYPE = int(os.environ.get("SOPHON_V031_PER_TYPE", "6"))

CONDITIONS = ["V030", "V031", "FULL"]
HAIKU_CMD = "claude -p --model haiku --output-format json"


def rpc(requests, env=None, timeout=600):
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
                  "clientInfo":{"name":"v031","version":"0"}}}


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


def sophon_compress(messages, query, retriever_path, hyde=False, fact_cards=False):
    """v0.3.0 mode: LLM summary (block-based Haiku) + retrieval. SAME setup
    as the public §3.7 bench — no SOPHON_NO_LLM_SUMMARY flag."""
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    if hyde:       env["SOPHON_HYDE"] = "1"
    if fact_cards: env["SOPHON_FACT_CARDS"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=600)
    return extract(out.get(1)) if out else None


def flatten(item):
    """Flatten sessions → (sophon_msgs, full_text).

    Session datetime is prepended to each message content so it survives
    chunking and shows up in retrieved chunks. Gold answers for multi-hop
    and temporal items are often of the form "last week of October 2023",
    computed from session_datetime + a relative phrase inside the turn.
    Without the datetime in the chunk, even a perfect retrieval leaves the
    downstream LLM unable to resolve the answer.
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


def run_sophon(name, messages, question):
    flags = {
        "V030": dict(hyde=False, fact_cards=False),
        "V031": dict(hyde=True,  fact_cards=True),
    }[name]
    d = Path(tempfile.mkdtemp(prefix=f"v031_{name}_"))
    try:
        comp = sophon_compress(messages, question, d, **flags)
        if not comp:
            return name, "(compress failed)"
        return name, build_block(comp)
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
print(f"[v031-vs-v030] N={len(subset)} items, {SAMPLES_PER_TYPE}/type, seed=42")

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
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(run_sophon, name, sophon_msgs, item["question"]): name
                for name in ["V030", "V031"]}
        for fut in futs:
            name, block = fut.result()
            blocks[name] = block

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
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
    print(f"   {summary}   ({dt:.0f}s)")

(ROOT/"v031_vs_v030_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s")

# ---------- aggregation ----------
import statistics

def pct(k, n):
    return (100.0*k/n) if n else 0.0

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return (max(0, (center-half)*100), min(100, (center+half)*100))

print()
print("="*88)
print(f"v0.3.1 (HyDE+FactCards) vs v0.3.0  —  N = {len(results)}")
print("="*88)
TYPES = ["single_hop","multi_hop","temporal_reasoning","open_domain","adversarial"]
print(f"{'Type':<22}{'V030':>10}{'V031':>10}{'FULL':>10}{'Δ(V031-V030)':>16}{'n':>5}")
for qtype in TYPES:
    items = [r for r in results if r["question_type"]==qtype]
    if not items: continue
    v030_k = sum(1 for r in items if r["conditions"]["V030"]["correct"] is True)
    v031_k = sum(1 for r in items if r["conditions"]["V031"]["correct"] is True)
    full_k = sum(1 for r in items if r["conditions"]["FULL"]["correct"] is True)
    n = len(items)
    print(f"{qtype:<22}{pct(v030_k,n):>9.1f}%{pct(v031_k,n):>9.1f}%{pct(full_k,n):>9.1f}%"
          f"{(pct(v031_k,n)-pct(v030_k,n)):>15.1f}pt{n:>5}")
print("-"*88)
total = len(results)
v030_k = sum(1 for r in results if r["conditions"]["V030"]["correct"] is True)
v031_k = sum(1 for r in results if r["conditions"]["V031"]["correct"] is True)
full_k = sum(1 for r in results if r["conditions"]["FULL"]["correct"] is True)
print(f"{'GLOBAL':<22}{pct(v030_k,total):>9.1f}%{pct(v031_k,total):>9.1f}%{pct(full_k,total):>9.1f}%"
      f"{(pct(v031_k,total)-pct(v030_k,total)):>15.1f}pt{total:>5}")

print()
print("95% Wilson CI")
print("-"*88)
for name, k in [("V030", v030_k), ("V031", v031_k), ("FULL", full_k)]:
    lo, hi = wilson_ci(k, total)
    print(f"  {name:<6}  {pct(k,total):>5.1f}%  [{lo:>5.1f}% — {hi:>5.1f}%]   ({k}/{total})")

print()
print("Context tokens (mean / max)")
print("-"*88)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    print(f"  {c:<6}  mean={statistics.mean(vals):>8.0f}  max={max(vals):>7}")

print()
print(f"saved → {ROOT/'v031_vs_v030_results.json'}")
