#!/usr/bin/env python3
"""
LOCOMO P1+P2 A/B bench — validates the v0.3.1 combo:

  * P1 (SOPHON_HYBRID=1)     : BM25 sparse-lexical + vector, RRF-fused
  * P2 (SOPHON_FACT_CARDS=1) : entity-indexed fact timeline via 1 Haiku call

Conditions:
  * BASELINE   — SOPHON_RETR v0.3.0 (vector-only, no fact cards)
  * P1_ONLY    — hybrid retrieval, no fact cards
  * P2_ONLY    — vector-only, with fact cards
  * P1_P2      — hybrid + fact cards (full v0.3.1 stack)
  * FULL       — ceiling (raw conversation)

Stratified sample across all 5 LOCOMO question types so we see where each
fix helps (P1 should win on single-hop/open-domain; P2 on temporal).
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "p1p2_ab_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLES_PER_TYPE = int(os.environ.get("SOPHON_P1P2_PER_TYPE", "4"))

CONDITIONS = ["BASELINE", "P1_ONLY", "P2_ONLY", "P1_P2", "FULL"]
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
                  "clientInfo":{"name":"p1p2","version":"0"}}}


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


def sophon_compress(messages, query, retriever_path, hybrid=False, fact_cards=False):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ, "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD}
    if hybrid:
        env["SOPHON_HYBRID"] = "1"
    if fact_cards:
        env["SOPHON_FACT_CARDS"] = "1"
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=600)
    return extract(out.get(1)) if out else None


def flatten(item):
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


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qtype, items in sorted(by_type.items()):
    random.shuffle(items)
    subset.extend(items[:SAMPLES_PER_TYPE])
print(f"[p1p2-ab] N={len(subset)} items, {SAMPLES_PER_TYPE}/type, seed=42")

results = []
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) — {item['question'][:55]}...", flush=True)
    sophon_msgs, full_text = flatten(item)

    # 4 Sophon passes + 1 FULL.
    conds_cfg = [
        ("BASELINE", False, False),
        ("P1_ONLY",  True,  False),
        ("P2_ONLY",  False, True),
        ("P1_P2",    True,  True),
    ]
    blocks = {}
    for name, hybrid, fc in conds_cfg:
        d = Path(tempfile.mkdtemp(prefix=f"locp_{name}_"))
        try:
            comp = sophon_compress(sophon_msgs, item["question"], d,
                                   hybrid=hybrid, fact_cards=fc)
            blocks[name] = build_block(comp) if comp else "(compress failed)"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    blocks["FULL"] = full_text

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
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in CONDITIONS)
    print(f"   {summary}")

(ROOT/"p1p2_ab_results.json").write_text(json.dumps(results, indent=2))

# ----- aggregation -----
import statistics
print()
print("="*95)
print(f"P1+P2 A/B — N = {len(results)}")
print("="*95)
TYPES = ["single_hop","multi_hop","temporal_reasoning","open_domain","adversarial"]
print(f"{'Type':<22}" + "".join(f"{c:>14}" for c in CONDITIONS) + f"{'n':>5}")
for qtype in TYPES:
    items = [r for r in results if r["question_type"]==qtype]
    line = f"{qtype:<22}"
    for c in CONDITIONS:
        correct = sum(1 for r in items if r["conditions"][c]["correct"] is True)
        total = len(items)
        pct = 100*correct/total if total else 0
        line += f"{pct:>13.1f}%"
    line += f"{len(items):>5}"
    print(line)
print("-"*95)
line = f"{'GLOBAL':<22}"
for c in CONDITIONS:
    correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    pct = 100*correct/len(results)
    line += f"{pct:>13.1f}%"
line += f"{len(results):>5}"
print(line)

print()
print("TOKEN ECONOMICS (mean ctx tokens)")
print("-"*95)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    print(f"  {c:<15} mean={statistics.mean(vals):>8.0f}  max={max(vals):>7}")

print()
print(f"saved → {ROOT/'p1p2_ab_results.json'}")
