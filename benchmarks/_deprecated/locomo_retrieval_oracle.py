#!/usr/bin/env python3
"""
LOCOMO retrieval oracle — zero-LLM ablation bench.

Reads oracle_labels.json (gold session indices per item) and measures
retrieval quality across flag combinations. Pure retrieval metrics:

  * hit@K      — any gold chunk in top-K retrieved
  * recall@K   — fraction of gold sessions covered by top-K
  * MRR        — 1/rank of first gold chunk (0 if not retrieved)
  * latency    — wall clock per call (deterministic, no LLM subprocess)

No Haiku / Sonnet at runtime. All scoring is local string/index match.

Usage:
  python3 benchmarks/locomo_retrieval_oracle.py              # all configs
  SOPHON_ORACLE_CONFIGS=baseline,v5 python3 ... oracle.py   # subset
"""
import json, os, re, shutil, subprocess, tempfile, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from copy import deepcopy

ROOT = Path(os.environ.get("SOPHON_BENCH_LOCOMO", "/private/tmp/sophon_bench/locomo"))
LABELS_PATH = ROOT / "oracle_labels.json"
RESULTS_PATH = ROOT / "oracle_results.json"
SOPHON = os.environ.get(
    "SOPHON_BIN",
    "/Volumes/nvme/projet claude/mcp-Sophon/sophon/target/release/sophon",
)
TOP_K = int(os.environ.get("SOPHON_ORACLE_TOP_K", "10"))
WORKERS = int(os.environ.get("SOPHON_ORACLE_WORKERS", "4"))

# ---------- Rust-only flag bundles ----------
# No SOPHON_HYDE / SOPHON_FACT_CARDS / SOPHON_LLM_RERANK / SOPHON_TAIL_SUMMARY
# / SOPHON_ADAPTIVE / SOPHON_REACT / SOPHON_MULTIHOP_LLM — those invoke LLMs.
CONFIGS = {
    "baseline": {},  # plain HashEmbedder
    "multihop_heuristic": {"SOPHON_MULTIHOP": "1"},
    "hybrid": {"SOPHON_HYBRID": "1"},
    "entity_graph": {"SOPHON_ENTITY_GRAPH": "1"},
    "entity_weighted_P5": {
        "SOPHON_ENTITY_GRAPH": "1",
        "SOPHON_ENTITY_WEIGHTED": "1",
    },
    "chunk_entity_aware_P6": {"SOPHON_CHUNK_ENTITY_AWARE": "1"},
    "chunk_500": {"SOPHON_CHUNK_TARGET": "500", "SOPHON_CHUNK_MAX": "700"},
    "hybrid_plus_graph": {"SOPHON_HYBRID": "1", "SOPHON_ENTITY_GRAPH": "1"},
    "all_rust_on": {
        "SOPHON_HYBRID": "1",
        "SOPHON_ENTITY_GRAPH": "1",
        "SOPHON_ENTITY_WEIGHTED": "1",
        "SOPHON_MULTIHOP": "1",
        "SOPHON_CHUNK_ENTITY_AWARE": "1",
        "SOPHON_CHUNK_TARGET": "500",
        "SOPHON_CHUNK_MAX": "700",
    },
}


def rpc(requests, env=None, timeout=120):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    try:
        p = subprocess.run(
            [SOPHON, "serve"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
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


INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "oracle", "version": "0"},
    },
}


def call(name, args, rid):
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }


def extract(resp):
    if not resp:
        return None
    res = resp.get("result")
    if not res:
        return None
    if "structuredContent" in res:
        return res["structuredContent"]
    try:
        return json.loads(res["content"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def flatten_with_session_map(item):
    """Flatten sessions into a flat message list, tagging each message with
    `[Session N | timestamp]`. The tag is 1-based by convention (so
    "Session 1" = index 0). Chunk-to-session mapping is recovered at score
    time by regex-matching the tag in chunk content — `source_message_indices`
    isn't exposed in the MCP response payload."""
    sessions = item["haystack_sessions"]
    datetimes = item.get("haystack_session_datetimes", [""] * len(sessions))
    sophon_msgs = []
    for i, sess in enumerate(sessions):
        ts = datetimes[i] if i < len(datetimes) else ""
        session_tag = f"[Session {i+1} | {ts}]"
        for turn in sess:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = turn.get("content", "") or ""
            sophon_msgs.append({"role": role, "content": f"{session_tag} {content}"})
    return sophon_msgs


SESSION_TAG_RE = re.compile(r"\[Session (\d+) \|")


def extract_chunk_sessions(chunk_content):
    """Return the set of 0-based session indices referenced by `[Session N |`
    tags in the chunk content. A chunk may span multiple messages from
    different sessions (rare but possible after packing)."""
    return {int(m.group(1)) - 1 for m in SESSION_TAG_RE.finditer(chunk_content)}


def run_config(cfg_name, cfg_env, item, labels):
    gold_sessions = set(labels.get(item["question_id"]) or [])
    if not gold_sessions:
        return None  # skip unlabeled / unanswerable

    msgs = flatten_with_session_map(item)

    d = Path(tempfile.mkdtemp(prefix=f"oracle_{cfg_name}_"))
    env = {**os.environ, "SOPHON_RETRIEVER_PATH": str(d), **cfg_env}
    # Ensure no stray LLM flags sneak in from the caller's environment
    for k in [
        "SOPHON_LLM_CMD",
        "SOPHON_HYDE",
        "SOPHON_FACT_CARDS",
        "SOPHON_LLM_RERANK",
        "SOPHON_TAIL_SUMMARY",
        "SOPHON_ADAPTIVE",
        "SOPHON_REACT",
        "SOPHON_MULTIHOP_LLM",
    ]:
        env.pop(k, None)

    try:
        t0 = time.perf_counter()
        out = rpc(
            [
                INIT,
                call(
                    "compress_history",
                    {
                        "messages": msgs,
                        "recent_window": 0,  # irrelevant for oracle, just noise
                        "max_tokens": 99999,  # don't let budget shrink retrieval
                        "query": item["question"],
                        "retrieval_top_k": TOP_K,
                    },
                    1,
                ),
            ],
            env=env,
            timeout=120,
        )
        latency = time.perf_counter() - t0
    finally:
        shutil.rmtree(d, ignore_errors=True)

    comp = extract(out.get(1)) if out else None
    if not comp:
        return {"error": "compress_history returned nothing", "latency": latency}

    rc = comp.get("retrieved_chunks") or {}
    chunks = rc.get("chunks", [])
    retrieved_sessions_per_rank = [
        extract_chunk_sessions(sc.get("chunk", {}).get("content", "")) for sc in chunks
    ]

    # hit@K: any gold in any retrieved chunk
    hit_at_k = any(bool(gold_sessions & s) for s in retrieved_sessions_per_rank)
    # recall@K: fraction of gold sessions covered
    covered = set()
    for s in retrieved_sessions_per_rank:
        covered |= s
    recall = (
        len(covered & gold_sessions) / len(gold_sessions) if gold_sessions else 0.0
    )
    # MRR: 1/rank of first hit
    mrr = 0.0
    for rank, s in enumerate(retrieved_sessions_per_rank, start=1):
        if s & gold_sessions:
            mrr = 1.0 / rank
            break

    return {
        "hit_at_k": bool(hit_at_k),
        "recall": recall,
        "mrr": mrr,
        "latency": latency,
        "n_retrieved": len(chunks),
        "n_gold": len(gold_sessions),
    }


def main():
    all_items = {
        json.loads(l)["question_id"]: json.loads(l)
        for l in open(ROOT / "all_items.jsonl")
    }
    labels = json.loads(LABELS_PATH.read_text())
    # Keep only items with non-empty gold sessions
    labeled_qids = [qid for qid, sess in labels.items() if sess]
    items = [all_items[qid] for qid in labeled_qids if qid in all_items]
    print(
        f"[oracle] N={len(items)} items with gold labels (of {len(labels)} total labeled)",
        flush=True,
    )

    configs = CONFIGS
    env_filter = os.environ.get("SOPHON_ORACLE_CONFIGS")
    if env_filter:
        keep = set(env_filter.split(","))
        configs = {k: v for k, v in CONFIGS.items() if k in keep}
    print(f"[oracle] configs: {list(configs.keys())}")
    print(f"[oracle] top_k={TOP_K}, workers={WORKERS}")

    results = {cfg: [] for cfg in configs}
    for cfg_name, cfg_env in configs.items():
        t0 = time.perf_counter()
        print(f"\n[{cfg_name}] starting ({len(items)} items, {WORKERS} workers)")
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {
                pool.submit(run_config, cfg_name, cfg_env, it, labels): it["question_id"]
                for it in items
            }
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": str(e)}
                if res is not None:
                    res["question_id"] = qid
                    res["question_type"] = all_items[qid]["question_type"]
                    results[cfg_name].append(res)
                done += 1
                if done % 20 == 0 or done == len(items):
                    print(f"  [{cfg_name}] {done}/{len(items)}", flush=True)
        dt = time.perf_counter() - t0
        print(f"[{cfg_name}] done in {dt:.1f}s")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {RESULTS_PATH}")

    # ---- aggregation ----
    TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]
    print()
    print("=" * 110)
    print(f"RETRIEVAL ORACLE — N={len(items)} (items with gold labels), top_k={TOP_K}")
    print("=" * 110)
    header = f"{'config':<25} {'hit@K':>8} {'recall':>9} {'MRR':>7} {'lat_p50':>9} {'lat_p95':>9}"
    for qt in TYPES:
        header += f" {qt[:11]:>12}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        rows = [r for r in results[cfg] if "error" not in r]
        if not rows:
            continue
        hits = sum(1 for r in rows if r["hit_at_k"]) / len(rows) * 100
        recall = sum(r["recall"] for r in rows) / len(rows) * 100
        mrr = sum(r["mrr"] for r in rows) / len(rows)
        lats = sorted(r["latency"] for r in rows)
        p50 = lats[len(lats) // 2] * 1000
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] * 1000
        line = f"{cfg:<25} {hits:>7.1f}% {recall:>8.1f}% {mrr:>7.3f} {p50:>7.0f}ms {p95:>7.0f}ms"
        for qt in TYPES:
            qt_rows = [r for r in rows if r["question_type"] == qt]
            if qt_rows:
                qt_hit = sum(1 for r in qt_rows if r["hit_at_k"]) / len(qt_rows) * 100
                line += f" {qt_hit:>11.1f}%"
            else:
                line += f"{'-':>12}"
        print(line)

    # Per-type table (hit@K only, more readable)
    print()
    print("HIT@K by question type")
    print("-" * 110)
    print(f"{'config':<25}" + "".join(f"{qt[:11]:>13}" for qt in TYPES) + f"{'GLOBAL':>10}{'n':>5}")
    for cfg in configs:
        rows = [r for r in results[cfg] if "error" not in r]
        if not rows:
            continue
        line = f"{cfg:<25}"
        for qt in TYPES:
            qt_rows = [r for r in rows if r["question_type"] == qt]
            if qt_rows:
                v = sum(1 for r in qt_rows if r["hit_at_k"]) / len(qt_rows) * 100
                line += f"{v:>12.1f}%"
            else:
                line += f"{'-':>13}"
        global_v = sum(1 for r in rows if r["hit_at_k"]) / len(rows) * 100
        line += f"{global_v:>9.1f}%{len(rows):>5}"
        print(line)


if __name__ == "__main__":
    main()
