#!/usr/bin/env python3
"""
LOCOMO open-ended QA — adds the new SOPHON_RETR condition (compression +
retrieval) and compares against the previous SOPHON_COMP / FULL / NONE
baseline.

For each item:
  * NONE         — question only (floor)
  * SOPHON_COMP  — compress_history without retrieval (existing behaviour)
  * SOPHON_RETR  — compress_history WITH query → uses semantic retriever
                    (HashEmbedder) to inject top-k relevant chunks
  * FULL         — entire raw conversation (ceiling)

The retriever uses a fresh tempdir per item (cold cache) so we measure the
actual retrieval quality, not cross-item bleed.

This is the proof point for the semantic-retriever module: if it works, the
SOPHON_RETR accuracy should land somewhere between SOPHON_COMP and FULL.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "./benchmarks/data/locomo"))
RUNS = ROOT / "retrieval_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN", "sophon")
MODEL = "sonnet"
JUDGE = "sonnet"

SAMPLES_PER_TYPE = 12
CONDITIONS = ["NONE", "SOPHON_COMP", "SOPHON_RETR", "FULL"]


def rpc(requests, env=None, timeout=60):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    p = subprocess.run([SOPHON, "serve"], input=payload, capture_output=True,
                       text=True, timeout=timeout, env=env)
    out = {}
    for line in p.stdout.splitlines():
        if line.strip():
            d = json.loads(line)
            out[d.get("id")] = d
    return out


INIT = {"jsonrpc":"2.0","id":0,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
                  "clientInfo":{"name":"loo","version":"0"}}}


def call(name, args, rid):
    return {"jsonrpc":"2.0","id":rid,"method":"tools/call","params":{"name":name,"arguments":args}}


def extract(resp):
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res: return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def count_tokens(text):
    out = rpc([INIT, call("count_tokens",{"text":text},1)])
    return extract(out[1])["token_count"]


def sophon_compress(messages, query=None, retriever_path=None):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000}
    if query:
        args["query"] = query
        args["retrieval_top_k"] = 5
    env = None
    if retriever_path:
        import os
        env = {**os.environ, "SOPHON_RETRIEVER_PATH": str(retriever_path)}
    out = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=120)
    return extract(out[1])


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


def build_compressed_block(comp):
    """Same shape as the original open-ended bench: summary + facts + recent."""
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
    return "\n\n".join(parts) or "(empty)"


def build_retrieval_block(comp):
    """Compressed block + retrieved chunks (the new condition)."""
    base = build_compressed_block(comp)
    rc = comp.get("retrieved_chunks", {})
    if not rc or not rc.get("chunks"):
        return base
    retrieved_lines = []
    for sc in rc["chunks"]:
        chunk = sc.get("chunk", {})
        score = sc.get("score", 0.0)
        retrieved_lines.append(f"[score={score:.2f}] {chunk.get('content','')}")
    return f"{base}\n\nRETRIEVED CONTEXT:\n" + "\n".join(retrieved_lines)


def build_prompt(context_block, q):
    return f"""You are answering a question about a long conversation.

CONTEXT:
{context_block}

QUESTION: {q}

Answer concisely. If the context does not contain enough information to answer, say "I don't know."
"""


def call_claude(prompt, model=MODEL, timeout=120):
    t0 = time.perf_counter_ns()
    p = subprocess.run(["claude","-p","--model",model,"--output-format","json"],
                       input=prompt, capture_output=True, text=True, timeout=timeout)
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
    p = subprocess.run(["claude","-p","--model",JUDGE,"--output-format","json"],
                       input=prompt, capture_output=True, text=True, timeout=90)
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


# ----------------------------------------------------------------------
all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qtype, items in by_type.items():
    random.shuffle(items)
    subset.extend(items[:SAMPLES_PER_TYPE])
print(f"sampled {len(subset)} items (open-ended + retrieval)")

results = []
for idx, item in enumerate(subset):
    cache = RUNS / f"{item['question_id']}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached")
        continue

    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) ...", flush=True)
    sophon_msgs, full_text = flatten(item)

    # Compress without retrieval (baseline)
    comp_no_retr = sophon_compress(sophon_msgs)
    block_comp = build_compressed_block(comp_no_retr)

    # Compress WITH retrieval (new condition) — fresh tempdir per item
    retr_dir = Path(tempfile.mkdtemp(prefix="locomo_retr_"))
    try:
        comp_with_retr = sophon_compress(
            sophon_msgs,
            query=item["question"],
            retriever_path=retr_dir,
        )
        block_retr = build_retrieval_block(comp_with_retr)
        retrieval_meta = comp_with_retr.get("retrieved_chunks", {})
    finally:
        shutil.rmtree(retr_dir, ignore_errors=True)

    contexts = {
        "NONE":        "(no context)",
        "SOPHON_COMP": block_comp,
        "SOPHON_RETR": block_retr,
        "FULL":        full_text,
    }

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "retrieval_meta": {
            "embedder": retrieval_meta.get("embedder"),
            "total_searched": retrieval_meta.get("total_searched"),
            "latency_ms": retrieval_meta.get("latency_ms"),
            "n_chunks": len(retrieval_meta.get("chunks", [])),
        },
        "conditions": {},
    }
    for cond, ctx in contexts.items():
        prompt = build_prompt(ctx, item["question"])
        answer, lat = call_claude(prompt)
        correct, rationale = judge_correctness(item["question"], item["answer"], answer)
        row["conditions"][cond] = {
            "ctx_tokens": count_tokens(ctx) if ctx != "(no context)" else 0,
            "latency_ms": round(lat),
            "answer": answer[:500],
            "correct": correct,
            "rationale": rationale,
        }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    summary = " ".join(f"{c}={'✓' if row['conditions'][c]['correct'] is True else '✗'}" for c in CONDITIONS)
    n_chunks = row["retrieval_meta"]["n_chunks"]
    print(f"   {summary}  retrieved={n_chunks} chunks")

(ROOT/"retrieval_results.json").write_text(json.dumps(results, indent=2))

# ---- aggregation ----
import statistics
print()
print("="*90)
print(f"OPEN-ENDED + RETRIEVAL — N = {len(results)}")
print("="*90)
TYPES = ["single_hop","multi_hop","temporal_reasoning","open_domain","adversarial"]
print(f"{'Type':<22}" + "".join(f"{c:>15}" for c in CONDITIONS) + f"{'n':>6}")
for qtype in TYPES:
    items = [r for r in results if r["question_type"]==qtype]
    line = f"{qtype:<22}"
    for c in CONDITIONS:
        correct = sum(1 for r in items if r["conditions"][c]["correct"] is True)
        total = len(items)
        pct = 100*correct/total if total else 0
        line += f"{pct:>14.1f}%"
    line += f"{len(items):>6}"
    print(line)
print("-"*90)
line = f"{'GLOBAL':<22}"
for c in CONDITIONS:
    correct = sum(1 for r in results if r["conditions"][c]["correct"] is True)
    pct = 100*correct/len(results)
    line += f"{pct:>14.1f}%"
line += f"{len(results):>6}"
print(line)

print()
print("TOKEN ECONOMICS (mean ctx tokens)")
print("-"*90)
for c in CONDITIONS:
    vals = [r["conditions"][c]["ctx_tokens"] for r in results]
    print(f"  {c:<15} mean={statistics.mean(vals):>8.0f}  max={max(vals):>7}")

print()
print("RETRIEVAL META (mean over items)")
print("-"*90)
n_chunks = [r["retrieval_meta"]["n_chunks"] for r in results if r["retrieval_meta"]["n_chunks"] is not None]
latencies = [r["retrieval_meta"]["latency_ms"] for r in results if r["retrieval_meta"]["latency_ms"] is not None]
searched = [r["retrieval_meta"]["total_searched"] for r in results if r["retrieval_meta"]["total_searched"] is not None]
if n_chunks:
    print(f"  embedder              : {results[0]['retrieval_meta']['embedder']}")
    print(f"  chunks/item           : mean={statistics.mean(n_chunks):.1f}  max={max(n_chunks)}")
    print(f"  latency/item (ms)     : mean={statistics.mean(latencies):.0f}  max={max(latencies)}")
    print(f"  index size (chunks)   : mean={statistics.mean(searched):.0f}  max={max(searched)}")

print()
print("saved to", ROOT/"retrieval_results.json")
