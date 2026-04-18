#!/usr/bin/env python3
"""
Audit whether V030 (current v0.3.0 behaviour, LLM summary + vector retrieval)
reproduces the published ~40 % LOCOMO accuracy claimed in BENCHMARK.md §3.7.

If V030 drifts well below 40 % on a fresh stratified sample, the V031 delta
reported in later benches would be measuring "V031 vs a degraded V030" rather
than "V031 vs the published baseline". That would make the claim dishonest.

Sample: 4/type × 5 types = 20 items.

Two conditions only (to keep wall-clock under an hour):
  * V030 — block-based LLM summary + semantic retrieval (SOPHON_RETR in §3.7)
  * FULL — raw conversation ceiling (sanity: should land near ~75 %)

No session-timestamp injection here — the published §3.7 numbers were
produced WITHOUT that enhancement. So we mirror that setup precisely.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "v030_baseline_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLES_PER_TYPE = int(os.environ.get("V030_PER_TYPE", "4"))

CONDITIONS = ["V030", "FULL"]
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
                  "clientInfo":{"name":"v030","version":"0"}}}


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


def sophon_compress(messages, query, retriever_path):
    """V030 setup: block-based LLM summary + retrieval, NO v0.3.1 flags."""
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=600)
    return extract(out.get(1)) if out else None


def flatten(item):
    """§3.7-faithful flatten: NO session datetime injection. The public
    bench did not preprocess messages this way, so we don't either here."""
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""]*len(sessions))
    sophon_msgs = []
    full_lines = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        full_lines.append(f"[Session {i+1} | {ts}]")
        for turn in sess:
            role = turn.get("role","user")
            if role not in ("user","assistant","system"):
                role = "user"
            content = turn.get("content","") or ""
            sophon_msgs.append({"role": role, "content": content})
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
    try:
        p = subprocess.run(["claude","-p","--model",model,"--output-format","json"],
                           input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    try:
        d = json.loads(p.stdout)
        return d.get("result","") or ""
    except Exception:
        return p.stdout


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
        return None
    try:
        d = json.loads(p.stdout)
        txt = d.get("result","").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False))
    except Exception:
        return None


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qtype, items in sorted(by_type.items()):
    random.shuffle(items)
    subset.extend(items[:SAMPLES_PER_TYPE])
print(f"[v030-baseline] N={len(subset)} items, {SAMPLES_PER_TYPE}/type, seed=42")

results = []
t_start = time.perf_counter()
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    t0 = time.perf_counter()
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) — {item['question'][:50]}...", flush=True)
    sophon_msgs, full_text = flatten(item)

    d = Path(tempfile.mkdtemp(prefix="v030_"))
    try:
        comp = sophon_compress(sophon_msgs, item["question"], d)
        v030_block = build_block(comp) if comp else "(compress failed)"
    finally:
        shutil.rmtree(d, ignore_errors=True)

    row = {"question_id": item["question_id"], "question_type": item["question_type"],
           "question": item["question"], "gold": item["answer"], "conditions": {}}
    for cond, ctx in [("V030", v030_block), ("FULL", full_text)]:
        prompt = build_prompt(ctx, item["question"])
        answer = call_claude(prompt)
        correct = judge_correctness(item["question"], item["answer"], answer)
        row["conditions"][cond] = {
            "ctx_tokens": count_tokens(ctx),
            "answer": answer[:300],
            "correct": correct,
        }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    dt = time.perf_counter() - t0
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in CONDITIONS)
    print(f"   {summary}   ({dt:.0f}s)")

(ROOT/"v030_baseline_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s")

# ---- aggregation ----
import statistics

def pct(k,n): return (100*k/n) if n else 0
def wilson(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return (max(0,(c-h)*100), min(100,(c+h)*100))

print()
print("="*80)
print(f"V030 baseline audit — N = {len(results)}")
print("="*80)
TYPES = ["single_hop","multi_hop","temporal_reasoning","open_domain","adversarial"]
print(f"{'Type':<22}{'V030':>10}{'FULL':>10}{'n':>5}")
for qtype in TYPES:
    items = [r for r in results if r["question_type"]==qtype]
    if not items: continue
    v030_k = sum(1 for r in items if r["conditions"]["V030"]["correct"] is True)
    full_k = sum(1 for r in items if r["conditions"]["FULL"]["correct"] is True)
    n = len(items)
    print(f"{qtype:<22}{pct(v030_k,n):>9.1f}%{pct(full_k,n):>9.1f}%{n:>5}")
print("-"*80)
total = len(results)
v030_k = sum(1 for r in results if r["conditions"]["V030"]["correct"] is True)
full_k = sum(1 for r in results if r["conditions"]["FULL"]["correct"] is True)
print(f"{'GLOBAL':<22}{pct(v030_k,total):>9.1f}%{pct(full_k,total):>9.1f}%{total:>5}")

print()
print("95% Wilson CI")
print("-"*80)
lo, hi = wilson(v030_k, total)
print(f"  V030   {pct(v030_k,total):>5.1f}%   [{lo:>5.1f} — {hi:>5.1f}]   ({v030_k}/{total})")
lo, hi = wilson(full_k, total)
print(f"  FULL   {pct(full_k,total):>5.1f}%   [{lo:>5.1f} — {hi:>5.1f}]   ({full_k}/{total})")

print()
print("Published §3.7 reference: V030 (SOPHON_RETR COMP_LLM) = ~40 %, FULL = ~75 %")
gap_v030 = pct(v030_k, total) - 40
gap_full = pct(full_k, total) - 75
if abs(gap_v030) <= 10 and abs(gap_full) <= 10:
    print(f"✅ Baseline within ±10 pt of published claims "
          f"(V030 drift={gap_v030:+.1f}pt, FULL drift={gap_full:+.1f}pt)")
else:
    print(f"⚠️  Baseline diverges from published claims "
          f"(V030 drift={gap_v030:+.1f}pt, FULL drift={gap_full:+.1f}pt)")

print(f"\nsaved → {ROOT/'v030_baseline_results.json'}")
