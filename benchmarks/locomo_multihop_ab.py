#!/usr/bin/env python3
"""
LOCOMO multi-hop A/B bench — validates the P0 multi-hop LLM-in-the-loop
decomposition + RRF fusion added in v0.3.1.

Compares three conditions on LOCOMO multi_hop items only:

  * SOPHON_RETR          — baseline: single-pass retrieval (v0.3.0)
  * SOPHON_RETR_MH_LLM   — NEW: query decomposed via Haiku, sub-queries
                           retrieved separately, results fused via RRF
  * FULL                 — ceiling (full raw conversation)

Both Sophon conditions use the same SOPHON_LLM_CMD (Haiku) and the same
cold-cache tempdir per item. The only difference is SOPHON_MULTIHOP_LLM=1
on the second.

Positive signal: MH_LLM > RETR on multi_hop accuracy, FULL remains top.
Negative signal: MH_LLM ≤ RETR means the decomposer or fusion isn't
helping — investigate sub-query quality.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "multihop_ab_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
MODEL = "sonnet"
JUDGE = "sonnet"
SAMPLE_N = int(os.environ.get("SOPHON_MULTIHOP_N", "15"))

CONDITIONS = ["SOPHON_RETR", "SOPHON_RETR_MH_LLM", "FULL"]


def rpc(requests, env=None, timeout=180):
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
                  "clientInfo":{"name":"multihop","version":"0"}}}


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


def sophon_compress(messages, query, retriever_path, multihop_llm=False):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ, "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": "claude -p --model haiku --output-format json"}
    if multihop_llm:
        env["SOPHON_MULTIHOP_LLM"] = "1"
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


def build_retrieval_block(comp):
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


# ------- main ---------
all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
mh_items = [it for it in all_items if it["question_type"] == "multi_hop"]
random.shuffle(mh_items)
subset = mh_items[:SAMPLE_N]
print(f"[multihop-ab] N={len(subset)} items, seed=42")

results = []
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    print(f"[{idx+1}/{len(subset)}] {item['question_id']} — {item['question'][:60]}...", flush=True)
    sophon_msgs, full_text = flatten(item)

    # A: baseline retrieval
    dA = Path(tempfile.mkdtemp(prefix="locmh_a_"))
    try:
        comp_a = sophon_compress(sophon_msgs, item["question"], dA, multihop_llm=False)
        block_a = build_retrieval_block(comp_a) if comp_a else "(compress failed)"
        meta_a = (comp_a or {}).get("retrieved_chunks", {})
    finally:
        shutil.rmtree(dA, ignore_errors=True)

    # B: multi-hop LLM decomposition
    dB = Path(tempfile.mkdtemp(prefix="locmh_b_"))
    try:
        comp_b = sophon_compress(sophon_msgs, item["question"], dB, multihop_llm=True)
        block_b = build_retrieval_block(comp_b) if comp_b else "(compress failed)"
        meta_b = (comp_b or {}).get("retrieved_chunks", {})
    finally:
        shutil.rmtree(dB, ignore_errors=True)

    contexts = {
        "SOPHON_RETR":        block_a,
        "SOPHON_RETR_MH_LLM": block_b,
        "FULL":               full_text,
    }

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "decomposed_into": meta_b.get("decomposed_into"),
        "retr_meta_A": {
            "embedder": meta_a.get("embedder"),
            "n_chunks": len(meta_a.get("chunks", [])),
            "latency_ms": meta_a.get("latency_ms"),
        },
        "retr_meta_B": {
            "embedder": meta_b.get("embedder"),
            "n_chunks": len(meta_b.get("chunks", [])),
            "latency_ms": meta_b.get("latency_ms"),
        },
        "conditions": {},
    }
    for cond, ctx in contexts.items():
        prompt = build_prompt(ctx, item["question"])
        answer, lat = call_claude(prompt)
        correct, rationale = judge_correctness(item["question"], item["answer"], answer)
        row["conditions"][cond] = {
            "ctx_tokens": count_tokens(ctx),
            "latency_ms": round(lat),
            "answer": answer[:500],
            "correct": correct,
            "rationale": rationale,
        }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in CONDITIONS)
    subs = row.get("decomposed_into")
    subcount = len(subs) if subs else 0
    print(f"   {summary}  subqs={subcount}")

(ROOT/"multihop_ab_results.json").write_text(json.dumps(results, indent=2))

# ---- aggregation ----
import statistics
print()
print("="*80)
print(f"MULTI-HOP A/B — N = {len(results)}")
print("="*80)
print(f"{'Condition':<22}{'accuracy':>12}{'correct/total':>18}{'ctx_mean':>12}")
for c in CONDITIONS:
    correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    total = len(results)
    pct = 100*correct/total if total else 0
    ctx_mean = statistics.mean(r["conditions"][c]["ctx_tokens"] for r in results)
    print(f"{c:<22}{pct:>11.1f}%{correct:>6}/{total:<9}{ctx_mean:>12.0f}")

print()
print("DECOMPOSITION STATS")
print("-"*80)
with_sub = [r for r in results if r.get("decomposed_into")]
print(f"items decomposed: {len(with_sub)}/{len(results)}")
if with_sub:
    avg_subs = statistics.mean(len(r["decomposed_into"]) for r in with_sub)
    print(f"avg sub-queries per decomposed item: {avg_subs:.1f}")
    print("sample decompositions:")
    for r in with_sub[:3]:
        print(f"  Q: {r['question'][:80]}")
        for s in r["decomposed_into"]:
            print(f"    - {s[:80]}")

print()
print(f"saved → {ROOT/'multihop_ab_results.json'}")
