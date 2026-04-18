#!/usr/bin/env python3
"""
LOCOMO cross-model A/B — tests whether Sophon's accuracy gains are
LLM-provider-agnostic. Compares V030 and V032_FULL contexts under two
different answerer/judge stacks:

    Stack A (reference): Claude Sonnet  answerer + Claude Sonnet  judge
    Stack B (challenger): user-picked CLI (codex/gpt/…) for both

Useful as a concrete answer to two questions:
  1. Does the V032 adversarial regression reproduce with GPT answerer?
     (GPT is often stricter about "I don't know" than Sonnet.)
  2. Does V032 keep its multi_hop / open_domain gains on other providers?

Configuration via env vars (all have sane defaults):
  SOPHON_BIN               — path to sophon binary
  SOPHON_LLM_CMD           — LLM command Sophon uses INTERNALLY (Haiku &
                             friends). Not parameterised per-stack here;
                             kept constant so the RETRIEVAL is held fixed
                             and only the ANSWERING layer changes.
  CROSS_ANSWERER_A         — Stack A answerer command
  CROSS_ANSWERER_B         — Stack B answerer command (different provider)
  CROSS_JUDGE              — judge command (shared across both stacks so
                             the comparator is constant)
  CROSS_N                  — per-type samples (default 4, stratified × 5
                             types = N=20)
  CROSS_TYPES              — comma-separated question types to include
                             (default: all five LOCOMO types)

Each command is a string; the script will split on whitespace and invoke
as subprocess, piping the prompt on stdin. Wrap providers with
`benchmarks/llm_cli.py` so they all emit plain text on stdout:

  CROSS_ANSWERER_A="python3 benchmarks/llm_cli.py --provider claude --model sonnet"
  CROSS_ANSWERER_B="python3 benchmarks/llm_cli.py --provider codex"
  CROSS_JUDGE="python3 benchmarks/llm_cli.py --provider claude --model sonnet"
"""
import json, os, random, re, shlex, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
RUNS = ROOT / "cross_model_runs"
RUNS.mkdir(parents=True, exist_ok=True)
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")

SOPHON_LLM_CMD = os.environ.get("SOPHON_LLM_CMD",
    "claude -p --model haiku --output-format json")

ANSWERER_A = os.environ.get("CROSS_ANSWERER_A",
    "python3 benchmarks/llm_cli.py --provider claude --model sonnet --timeout 180")
ANSWERER_B = os.environ.get("CROSS_ANSWERER_B",
    "python3 benchmarks/llm_cli.py --provider codex --timeout 300")
JUDGE = os.environ.get("CROSS_JUDGE",
    "python3 benchmarks/llm_cli.py --provider claude --model sonnet --timeout 180")

PER_TYPE = int(os.environ.get("CROSS_N", "4"))
TYPES = os.environ.get(
    "CROSS_TYPES",
    "adversarial,multi_hop,single_hop,open_domain,temporal_reasoning",
).split(",")

SOPHON_CONDS = ["V030", "V032_FULL"]


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
                  "clientInfo":{"name":"xmodel","version":"0"}}}


def call(name, args, rid):
    return {"jsonrpc":"2.0","id":rid,"method":"tools/call","params":{"name":name,"arguments":args}}


def extract(resp):
    if not resp: return None
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res: return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def count_tokens(text):
    out = rpc([INIT, call("count_tokens", {"text": text}, 1)])
    e = extract(out.get(1)) if out else None
    return e["token_count"] if e else 0


def sophon_compress(messages, query, retriever_path, mode):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": SOPHON_LLM_CMD}
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
        tag = f"[Session {i+1} | {ts}]"
        for turn in sess:
            role = turn.get("role","user")
            if role not in ("user","assistant","system"):
                role = "user"
            content = turn.get("content","") or ""
            sophon_msgs.append({"role": role, "content": f"{tag} {content}"})
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


def call_llm(cmd_str, prompt, timeout=240):
    """Run a free-form CLI command that reads stdin, returns stdout as text."""
    cmd = shlex.split(cmd_str)
    t0 = time.perf_counter_ns()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", (time.perf_counter_ns() - t0) / 1e6
    lat = (time.perf_counter_ns() - t0) / 1e6
    if p.returncode != 0:
        return f"(error: {p.stderr[:200]})", lat
    return p.stdout.strip(), lat


JUDGE_TEMPLATE = """You are scoring a QA system. Compare the candidate answer to the gold answer for the given question. Mark as correct if the candidate conveys the same factual answer as the gold (paraphrase is fine). Mark as wrong if the candidate is missing, contradicts the gold, or says "I don't know" when the gold has a concrete answer.

QUESTION: {question}

GOLD ANSWER: {gold}

CANDIDATE ANSWER: {candidate}

Respond with ONE line of strict JSON, nothing else:
{{"correct": true|false, "rationale": "<one short sentence>"}}
"""


def judge_one(question, gold, candidate):
    prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, candidate=candidate or "(empty)")
    txt, _ = call_llm(JUDGE, prompt, timeout=180)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m: return None
    try:
        parsed = json.loads(m.group(0))
        return bool(parsed.get("correct", False))
    except Exception:
        return None


def run_sophon_cond(name, messages, question):
    d = Path(tempfile.mkdtemp(prefix=f"xmodel_{name}_"))
    try:
        comp = sophon_compress(messages, question, d, mode=name)
        return name, build_block(comp) if comp else "(compress failed)"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------- main ----------
all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
by_type = {}
for it in all_items:
    by_type.setdefault(it["question_type"], []).append(it)

subset = []
for qt in TYPES:
    items = by_type.get(qt, [])
    if not items: continue
    random.shuffle(items)
    subset.extend(items[:PER_TYPE])
print(f"[xmodel] N={len(subset)} items, {PER_TYPE}/type, types={TYPES}")
print(f"[xmodel] SOPHON_LLM_CMD   = {SOPHON_LLM_CMD}")
print(f"[xmodel] CROSS_ANSWERER_A = {ANSWERER_A}")
print(f"[xmodel] CROSS_ANSWERER_B = {ANSWERER_B}")
print(f"[xmodel] CROSS_JUDGE      = {JUDGE}")
print()

results = []
t_start = time.perf_counter()
for idx, item in enumerate(subset):
    cache_key = f"{item['question_id']}_A-{hash(ANSWERER_A) & 0xffff:04x}_B-{hash(ANSWERER_B) & 0xffff:04x}"
    cache = RUNS / f"{cache_key}.json"
    if cache.exists():
        results.append(json.loads(cache.read_text()))
        print(f"[{idx+1}/{len(subset)}] {item['question_id']} — cached", flush=True)
        continue

    t0 = time.perf_counter()
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} ({item['question_type']}) — {item['question'][:50]}...", flush=True)
    sophon_msgs = flatten(item)

    blocks = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(run_sophon_cond, n, sophon_msgs, item["question"]): n
                for n in SOPHON_CONDS}
        for fut in futs:
            name, block = fut.result()
            blocks[name] = block

    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "results": {},
    }
    for sophon_cond in SOPHON_CONDS:
        ctx = blocks[sophon_cond]
        tokens = count_tokens(ctx)
        prompt = build_prompt(ctx, item["question"])
        for stack_name, stack_cmd in [("A", ANSWERER_A), ("B", ANSWERER_B)]:
            answer, lat = call_llm(stack_cmd, prompt, timeout=300)
            correct = judge_one(item["question"], item["answer"], answer)
            row["results"][f"{sophon_cond}@{stack_name}"] = {
                "ctx_tokens": tokens,
                "latency_ms": round(lat),
                "answer": answer[:300],
                "correct": correct,
            }
    cache.write_text(json.dumps(row, indent=2))
    results.append(row)
    dt = time.perf_counter() - t0
    summary = " ".join(
        f"{sc}@{st}={'✓' if row['results'][f'{sc}@{st}']['correct'] is True else '✗'}"
        for sc in SOPHON_CONDS for st in ["A", "B"]
    )
    print(f"   {summary}   dt={dt:.0f}s", flush=True)

(ROOT/"cross_model_results.json").write_text(json.dumps(results, indent=2))
print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.0f}s", flush=True)

# ----- aggregation -----
import statistics

def pct(k, n): return (100*k/n) if n else 0
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return (max(0,(c-h)*100), min(100,(c+h)*100))

labels = [f"{sc}@{st}" for sc in SOPHON_CONDS for st in ["A", "B"]]
print()
print("="*95)
print(f"CROSS-MODEL A/B — N = {len(results)}")
print("="*95)
print(f"{'Type':<22}" + "".join(f"{lab:>14}" for lab in labels) + f"{'n':>5}")
for qt in TYPES:
    items = [r for r in results if r["question_type"] == qt]
    if not items: continue
    line = f"{qt:<22}"
    for lab in labels:
        k = sum(1 for r in items if r["results"][lab]["correct"] is True)
        line += f"{pct(k, len(items)):>13.1f}%"
    line += f"{len(items):>5}"
    print(line)
print("-"*95)
line = f"{'GLOBAL':<22}"
totals = {}
for lab in labels:
    k = sum(1 for r in results if r["results"][lab]["correct"] is True)
    totals[lab] = (k, len(results))
    line += f"{pct(k, len(results)):>13.1f}%"
line += f"{len(results):>5}"
print(line)

print()
print("95% Wilson CI")
print("-"*95)
for lab in labels:
    k, n = totals[lab]
    lo, hi = wilson(k, n)
    print(f"  {lab:<14}  {pct(k, n):>5.1f}%  [{lo:>5.1f} — {hi:>5.1f}]    ({k}/{n})")

# Cross-stack deltas per Sophon condition (does stack change the picture?)
print()
print("Δ stack B − stack A (same Sophon config, different answerer)")
print("-"*95)
for sc in SOPHON_CONDS:
    ka, _ = totals[f"{sc}@A"]
    kb, _ = totals[f"{sc}@B"]
    print(f"  {sc:<12}  Δ = {pct(kb, len(results)) - pct(ka, len(results)):+.1f} pt")

# Cross-condition deltas per stack (does Sophon help uniformly across stacks?)
print()
print("Δ V032 − V030 (same answerer, different Sophon config)")
print("-"*95)
for st in ["A", "B"]:
    ka, _ = totals[f"V030@{st}"]
    kb, _ = totals[f"V032_FULL@{st}"]
    print(f"  stack {st}       Δ = {pct(kb, len(results)) - pct(ka, len(results)):+.1f} pt")

print()
print(f"saved → {ROOT/'cross_model_results.json'}")
