#!/usr/bin/env python3
"""
Audit LLM call failures (rate-limit, spawn, empty stdout) across the Sophon
compression pipeline. Runs 5 multi-hop items with SOPHON_DEBUG_LLM=1 and
counts every [sophon-llm] FAIL/WARN line emitted on stderr, grouped by
condition. Zero failures is the required precondition for trusting the
subsequent N=80 v0.3.1 headline bench.
"""
import json, os, random, re, shutil, subprocess, tempfile, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

random.seed(42)

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
SOPHON = os.environ.get("SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon")
HAIKU_CMD = "claude -p --model haiku --output-format json"

SAMPLE_N = int(os.environ.get("AUDIT_N", "5"))

SOPHON_CONDS = ["V030", "V031"]


def rpc(requests, env=None, timeout=600):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    try:
        p = subprocess.run([SOPHON, "serve"], input=payload, capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, ""
    out = {}
    for line in p.stdout.splitlines():
        if line.strip():
            try:
                d = json.loads(line)
                out[d.get("id")] = d
            except json.JSONDecodeError:
                continue
    return out, p.stderr


INIT = {"jsonrpc":"2.0","id":0,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
                  "clientInfo":{"name":"audit","version":"0"}}}


def call(name, args, rid):
    return {"jsonrpc":"2.0","id":rid,"method":"tools/call","params":{"name":name,"arguments":args}}


def extract(resp):
    if not resp: return None
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res: return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def sophon_compress(messages, query, retriever_path, hyde=False, fact_cards=False):
    args = {"messages": messages, "recent_window": 5, "max_tokens": 2000,
            "query": query, "retrieval_top_k": 5}
    env = {**os.environ,
           "SOPHON_RETRIEVER_PATH": str(retriever_path),
           "SOPHON_LLM_CMD": HAIKU_CMD,
           "SOPHON_DEBUG_LLM": "1"}
    if hyde:       env["SOPHON_HYDE"] = "1"
    if fact_cards: env["SOPHON_FACT_CARDS"] = "1"
    out, stderr = rpc([INIT, call("compress_history", args, 1)], env=env, timeout=600)
    if out is None:
        return None, stderr, "(timeout)"
    comp = extract(out.get(1)) if out else None
    return comp, stderr, None


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


def run_condition(name, messages, question):
    flags = {
        "V030": dict(hyde=False, fact_cards=False),
        "V031": dict(hyde=True,  fact_cards=True),
    }[name]
    d = Path(tempfile.mkdtemp(prefix=f"audit_{name}_"))
    t0 = time.perf_counter()
    try:
        comp, stderr, err = sophon_compress(messages, question, d, **flags)
        dt = time.perf_counter() - t0
        # Count FAIL/WARN lines for each condition
        fail_count = sum(1 for line in stderr.splitlines() if "[sophon-llm] FAIL" in line)
        warn_count = sum(1 for line in stderr.splitlines() if "[sophon-llm] WARN" in line)
        ok = comp is not None
        n_chunks = 0
        if comp:
            rc = comp.get("retrieved_chunks") or {}
            n_chunks = len(rc.get("chunks", []))
        return name, dt, fail_count, warn_count, ok, n_chunks, stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


all_items = [json.loads(l) for l in open(ROOT/"all_items.jsonl")]
mh_items = [it for it in all_items if it["question_type"] == "multi_hop"]
random.shuffle(mh_items)
subset = mh_items[:SAMPLE_N]
print(f"[audit-rate-limit] N={len(subset)} multi_hop items, SOPHON_DEBUG_LLM=1")
print()

all_stats = defaultdict(lambda: {"fails": 0, "warns": 0, "ok": 0, "n_runs": 0, "times": []})
sample_stderrs = defaultdict(list)

for idx, item in enumerate(subset):
    print(f"[{idx+1}/{len(subset)}] {item['question_id']} — {item['question'][:55]}...")
    sophon_msgs = flatten(item)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(run_condition, name, sophon_msgs, item["question"]): name
                for name in SOPHON_CONDS}
        for fut in futs:
            name, dt, fails, warns, ok, n_chunks, stderr = fut.result()
            all_stats[name]["fails"] += fails
            all_stats[name]["warns"] += warns
            all_stats[name]["ok"] += 1 if ok else 0
            all_stats[name]["n_runs"] += 1
            all_stats[name]["times"].append(dt)
            if stderr and len(sample_stderrs[name]) < 2:
                # save first couple of stderr samples per condition for inspection
                sample_stderrs[name].append(stderr[:2000])
            print(f"   {name}: ok={ok} fails={fails} warns={warns} chunks={n_chunks} dt={dt:.0f}s")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"{'Cond':<10}{'runs':>6}{'ok':>5}{'fails':>8}{'warns':>8}{'mean_dt':>10}{'max_dt':>10}")
for name in SOPHON_CONDS:
    s = all_stats[name]
    mean_dt = sum(s["times"])/len(s["times"]) if s["times"] else 0
    max_dt = max(s["times"]) if s["times"] else 0
    print(f"{name:<10}{s['n_runs']:>6}{s['ok']:>5}{s['fails']:>8}{s['warns']:>8}{mean_dt:>9.1f}s{max_dt:>9.1f}s")

print()
if any(all_stats[n]["fails"] for n in SOPHON_CONDS):
    print("⚠️  FAILURES DETECTED — sample stderr from one condition with fails:")
    for name in SOPHON_CONDS:
        if all_stats[name]["fails"] > 0:
            for st in sample_stderrs[name][:1]:
                fail_lines = [l for l in st.splitlines() if "FAIL" in l or "WARN" in l]
                for l in fail_lines[:5]:
                    print(f"  {l}")
            break
else:
    print("✅ Zero failures. Pipeline is clean for N=80 bench.")
