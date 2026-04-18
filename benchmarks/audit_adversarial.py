#!/usr/bin/env python3
"""
Adversarial-only validation bench — tests whether the -33 pt V032 regression
observed at N=6 stratified is real or noise. Runs V030 and V032_FULL on 20
adversarial items (seed 42), no FULL / V031 to halve wall-clock.

If V032 lands at 80-90 % → the earlier figure was small-sample noise, no fix
needed before v0.3.2 release.
If V032 lands at 60-75 % → real regression, warrants implementing a guard
(but the battle-test showed the obvious designs are full of holes, so this
would require a proper LLM-answerability check, not a naive coverage gate).
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "adv_audit_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLE_N = int(os.environ.get("ADV_N", "20"))

SOPHON_CONDS = ["V030", "V032_FULL"]
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
                  "clientInfo":{"name":"adv","version":"0"}}}


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
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    if mode == "V032_FULL":
        env["SOPHON_HYDE"] = "1"
        env["SOPHON_FACT_CARDS"] = "1"
        env["SOPHON_ADAPTIVE"] = "1"
        env["SOPHON_LLM_RERANK"] = "1"
        env["SOPHON_TAIL_SUMMARY"] = "1"
        env["SOPHON_CHUNK_TARGET"] = "500"
        env["SOPHON_CHUNK_MAX"] = "700"
        env["SOPHON_ENTITY_GRAPH"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=900)
    return extract(out.get(1)) if out else None


def flatten(item):
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""]*len(sessions))
    sophon_msgs = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        session_tag = f"[Session {i+1} | {ts}]"
        for turn in sess:
            role = turn.get("role","user")
            if role not in ("user","assistant","system"):
                role = "user"
            content = turn.get("content","") or ""
            sophon_msgs.append({"role": role, "content": f"{session_tag} {content}"})
    return sophon_msgs


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
        if not m: return None
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False))
    except Exception:
        return None


def run_cond(name, messages, question):
    d = Path(tempfile.mkdtemp(prefix=f"adv_{name}_"))
    try:
        comp = sophon_compress(messages, question, d, mode=name)
        return name, build_block(comp) if comp else "(compress failed)"
    finally:
        shutil.rmtree(d, ignore_errors=True)


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
adv = [it for it in all_items if it["question_type"] == "adversarial"]
random.shuffle(adv)
subset = adv[:SAMPLE_N]
print(f"[adv-audit] N={len(subset)} adversarial items, seed=42")

results = []
t_start = time.perf_counter()
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    t0 = time.perf_counter()
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} — {item['question'][:60]}...", flush=True)
    sophon_msgs = flatten(item)

    blocks = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(run_cond, n, sophon_msgs, item["question"]): n for n in SOPHON_CONDS}
        for fut in futs:
            name, block = fut.result()
            blocks[name] = block

    row = {
        "question_id": item["question_id"],
        "question": item["question"],
        "gold": item["answer"],
        "conditions": {},
    }
    for cond in SOPHON_CONDS:
        ctx = blocks[cond]
        ans = call_claude(build_prompt(ctx, item["question"]))
        correct = judge_correctness(item["question"], item["answer"], ans)
        row["conditions"][cond] = {
            "ctx_tokens": count_tokens(ctx),
            "answer": ans[:300],
            "correct": correct,
        }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    dt = time.perf_counter() - t0
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in SOPHON_CONDS)
    print(f"   {summary}   ({dt:.0f}s)", flush=True)

(ROOT/"adv_audit_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s", flush=True)

# ---- aggregation ----
def pct(k, n): return (100*k/n) if n else 0
def wilson(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5) / d
    return (max(0,(c-h)*100), min(100,(c+h)*100))

total = len(results)
v030_k = sum(1 for r in results if r["conditions"]["V030"]["correct"] is True)
v032_k = sum(1 for r in results if r["conditions"]["V032_FULL"]["correct"] is True)

print()
print("="*70)
print(f"ADVERSARIAL-ONLY AUDIT — N = {total}")
print("="*70)
print(f"{'Condition':<14}{'acc':>8}  95% Wilson CI           correct/total")
for name, k in [("V030", v030_k), ("V032_FULL", v032_k)]:
    lo, hi = wilson(k, total)
    print(f"  {name:<12}  {pct(k,total):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]     {k}/{total}")

delta = pct(v032_k, total) - pct(v030_k, total)
print()
print(f"Δ(V032 − V030) = {delta:+.1f} pt")
print()
if abs(delta) <= 8:
    print(f"✅ Δ within ±8 pt → earlier -33 pt at N=6 was small-sample noise.")
elif delta <= -15:
    print(f"⚠️  Real adversarial regression. Consider implementing an LLM")
    print(f"    answerability check before release.")
else:
    print(f"ℹ️  Modest regression ({delta:.0f} pt). Advisory field in payload may suffice.")

print(f"\nsaved → {ROOT/'adv_audit_results.json'}")
