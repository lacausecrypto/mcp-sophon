#!/usr/bin/env python3
"""
Repository QA bench — measures Sophon's `navigate_codebase` against two
baselines on a small set of "where is X?" questions answered from the
repo itself.

Conditions:
  SOPHON    — navigate_codebase(root, query=...) digest
  GREP      — rg against the query string (mimics `git grep` / `rg`),
              we then stream the top file paths as the "retrieval"
  FULL      — concat every file under the repo, capped at a token
              budget, passed as context

Metrics (mechanical, no LLM needed):
  recall@1   — is the gold file the #1 hit?
  recall@3
  recall@5
  tokens     — bytes or cl100k_base tokens of the "context" each
              condition would hand to the answerer
  latency    — wall-clock of the retrieval step

No Sonnet judge here — ground truth is a file path the author of the
question asserts to be the canonical location of the concept. Mechanical
recall is faster, deterministic, and reproducible.

Target repo: the Sophon repo itself (we know it by heart).
"""
import json, os, re, shutil, statistics, subprocess, time
from pathlib import Path

# ---------------- Config ----------------
REPO_ROOT = Path(
    os.environ.get("SOPHON_REPO", "/Volumes/nvme/projet claude/mcp-Sophon/sophon")
).resolve()
SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(REPO_ROOT / "target/release/sophon"),
)
OUT_PATH = Path(os.environ.get("REPO_BENCH_OUT", "/tmp/repo_qa.json"))
FULL_BUDGET_TOKENS = int(os.environ.get("REPO_FULL_BUDGET", "32000"))


# ---------------- Ground truth ----------------
# Each item maps a natural-language query to one or more files that
# the user would accept as the correct answer. Several-answer items
# let us score a hit if ANY of the listed files is in the top-K —
# important because "where is the retriever?" can legitimately be
# lib.rs OR retriever.rs.
QA_ITEMS = [
    {
        "query": "Where is the EntityGraph implementation?",
        "gold": ["crates/semantic-retriever/src/entity_graph.rs"],
    },
    {
        "query": "Where is the HashEmbedder defined?",
        "gold": ["crates/semantic-retriever/src/embedder.rs"],
    },
    {
        "query": "Where is the BM25 index code?",
        "gold": ["crates/semantic-retriever/src/bm25.rs"],
    },
    {
        "query": "Where is the chunker that splits messages?",
        "gold": ["crates/semantic-retriever/src/chunker.rs"],
    },
    {
        "query": "Where is the block-based LLM summariser?",
        "gold": ["crates/memory-manager/src/summarizer.rs"],
    },
    {
        "query": "Where is the HyDE query rewriter?",
        "gold": ["crates/memory-manager/src/query_rewriter.rs"],
    },
    {
        "query": "Where is the ReAct iterative decider?",
        "gold": ["crates/memory-manager/src/react.rs"],
    },
    {
        "query": "Where is the graph fact extraction prompt?",
        "gold": ["crates/memory-manager/src/graph/extract.rs"],
    },
    {
        "query": "Where is the fact card renderer?",
        "gold": ["crates/memory-manager/src/fact_cards.rs"],
    },
    {
        "query": "Where is the LLM subprocess shell-out helper?",
        "gold": ["crates/memory-manager/src/llm_client.rs"],
    },
    {
        "query": "Where does SophonServer hold the graph memory?",
        "gold": ["crates/mcp-integration/src/server.rs"],
    },
    {
        "query": "Where is the compress_history MCP handler?",
        "gold": ["crates/mcp-integration/src/handlers.rs"],
    },
    {
        "query": "Where is the delta streamer protocol for file reads?",
        "gold": [
            "crates/delta-streamer/src/protocol.rs",
            "crates/delta-streamer/src/lib.rs",
        ],
    },
    {
        "query": "Where are the MCP tool schemas defined?",
        "gold": ["crates/mcp-integration/src/tools.rs"],
    },
    {
        "query": "Where is the output compressor filter dispatch?",
        "gold": [
            "crates/output-compressor/src/lib.rs",
            "crates/output-compressor/src/filters.rs",
            "crates/output-compressor/src/filters/mod.rs",
        ],
    },
    {
        "query": "Where is the question classifier for adaptive mode?",
        "gold": ["crates/memory-manager/src/question_classifier.rs"],
    },
    {
        "query": "Where is the Bm25Index scoring formula?",
        "gold": ["crates/semantic-retriever/src/bm25.rs"],
    },
    {
        "query": "Where does compress_history call run_retrieval?",
        "gold": ["crates/mcp-integration/src/handlers.rs"],
    },
    {
        "query": "Where is the reciprocal rank fusion code?",
        "gold": ["crates/semantic-retriever/src/fusion.rs"],
    },
    {
        "query": "Where is the navigator that scans the codebase?",
        "gold": [
            "crates/codebase-navigator/src/lib.rs",
            "crates/codebase-navigator/src/navigator.rs",
        ],
    },
]


# ---------------- RPC helpers ----------------
INIT = {
    "jsonrpc": "2.0", "id": 0, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "repo-qa", "version": "0"},
    },
}


def rpc(requests, env=None, timeout=120):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    t0 = time.perf_counter_ns()
    p = subprocess.run(
        [SOPHON, "serve"], input=payload, capture_output=True,
        text=True, timeout=timeout, env=env,
    )
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    out = {}
    for line in p.stdout.splitlines():
        if line.strip():
            try:
                d = json.loads(line)
                out[d.get("id")] = d
            except json.JSONDecodeError:
                continue
    return out, wall_ms


def call(name, args, rid):
    return {
        "jsonrpc": "2.0", "id": rid, "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }


def extract(resp):
    if not resp: return None
    res = resp.get("result")
    if not res: return None
    if "structuredContent" in res:
        return res["structuredContent"]
    return json.loads(res["content"][0]["text"])


def count_tokens(env, text):
    out, _ = rpc([INIT, call("count_tokens", {"text": text}, 1)], env=env)
    e = extract(out.get(1))
    return e["token_count"] if e else 0


# ---------------- Path helpers ----------------
def normalise_path(path_str):
    """Resolve a path string relative to REPO_ROOT into a stable form
    that can be compared against gold entries. Strips the repo prefix,
    lowercases, uses forward slashes."""
    if not path_str:
        return ""
    p = path_str.strip().replace("\\", "/")
    # Try to make it relative to the repo root.
    try:
        abs_p = (REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        rel = abs_p.relative_to(REPO_ROOT)
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return p


def extract_paths_from_digest(digest_text):
    """Heuristic: a navigate_codebase digest mentions file paths in the
    form `crates/foo/src/bar.rs`. We pull all such tokens in order of
    first appearance (dedup preserving order)."""
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_\-/]*\.(?:rs|py|toml|md|ts|tsx|js|go|java|rb|c|cpp|h|hpp|kt|swift|php)")
    seen = []
    seen_set = set()
    for m in pattern.finditer(digest_text or ""):
        candidate = normalise_path(m.group(0))
        if candidate and candidate not in seen_set:
            seen.append(candidate)
            seen_set.add(candidate)
    return seen


def score_hit(ordered_paths, gold_paths):
    """For each K in {1,3,5,10}, does any gold path appear in the
    first K ordered_paths? Returns dict of bool."""
    hits = {"recall@1": False, "recall@3": False, "recall@5": False, "recall@10": False}
    gold_norm = [normalise_path(g) for g in gold_paths]
    for k, label in [(1, "recall@1"), (3, "recall@3"), (5, "recall@5"), (10, "recall@10")]:
        for p in ordered_paths[:k]:
            if any(p == g or p.endswith(g) or g.endswith(p) for g in gold_norm):
                hits[label] = True
                break
    return hits


# ---------------- Conditions ----------------
def run_sophon(env, query):
    out, wall = rpc([INIT, call("navigate_codebase", {
        "root": str(REPO_ROOT),
        "query": query,
        "max_tokens": 3000,
    }, 1)], env=env, timeout=120)
    r = extract(out.get(1))
    digest = json.dumps(r) if r else ""
    paths = extract_paths_from_digest(digest)
    tokens = count_tokens(env, digest)
    return {
        "paths": paths,
        "tokens": tokens,
        "latency_ms": wall,
        "raw_len": len(digest),
    }


def run_grep(env, query):
    """Mimic the "run rg / grep and hand the user the hit list" baseline.
    Build a loose regex from query tokens (capitalised words get
    highest weight). Falls back from rg → BSD grep → Python walk so
    the bench still produces a meaningful baseline when rg is a shell
    wrapper rather than a real binary on PATH."""
    terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", query)
             if len(t) > 3 and t.lower() not in {"where", "what", "when", "which", "does", "call", "code", "mcp"}]
    if not terms:
        return {"paths": [], "tokens": 0, "latency_ms": 0, "raw_len": 0}

    pattern = "|".join(terms)
    t0 = time.perf_counter_ns()
    stdout = ""
    # Try rg first, then BSD grep -rEl, then fall back to a pure-Python
    # walk so this works in any environment.
    for cmd in (
        ["rg", "-l", "-i", pattern, str(REPO_ROOT)],
        ["grep", "-rEl", "-i", "--include=*.rs", "--include=*.py",
         "--include=*.toml", "--include=*.md", pattern, str(REPO_ROOT)],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode in (0, 1):  # grep returns 1 on zero matches
                stdout = proc.stdout
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    if not stdout:
        # Pure-Python fallback: walk + regex filter.
        rx = re.compile(pattern, re.IGNORECASE)
        hits = []
        for root, _, files in os.walk(REPO_ROOT):
            for f in files:
                if Path(f).suffix not in {".rs", ".py", ".toml", ".md"}:
                    continue
                path = Path(root) / f
                try:
                    body = path.read_text(errors="replace")
                except OSError:
                    continue
                if rx.search(body):
                    hits.append(str(path))
        stdout = "\n".join(hits)

    wall = (time.perf_counter_ns() - t0) / 1e6
    # rg / grep -l return one path per line
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    paths = []
    seen = set()
    for raw in lines[:40]:
        p = normalise_path(raw)
        if p and p not in seen:
            paths.append(p)
            seen.add(p)
    # Token cost: imagine you concat the full body of the top-5 hits
    concat = []
    for p in paths[:5]:
        try:
            concat.append((REPO_ROOT / p).read_text(errors="replace"))
        except OSError:
            pass
    tokens = count_tokens(env, "\n\n".join(concat)) if concat else 0
    return {"paths": paths, "tokens": tokens, "latency_ms": wall, "raw_len": sum(len(c) for c in concat)}


def run_full(env, query, full_text, full_tokens):
    """FULL = concat every .rs file under the repo up to a token
    budget. Served as "perfect recall" reference — the token cost is
    the interesting number, not the path list."""
    # FULL has no ranking — the answerer would get the whole blob. But
    # we can still score recall@K by listing files in repo-scan order
    # and checking position of gold.
    t0 = time.perf_counter_ns()
    # Walk repo in a deterministic order (sorted)
    paths = []
    for root, _, files in os.walk(REPO_ROOT):
        for f in sorted(files):
            if f.endswith(".rs") or f.endswith(".py"):
                p = normalise_path(os.path.join(root, f))
                if p:
                    paths.append(p)
    wall = (time.perf_counter_ns() - t0) / 1e6
    # FULL's "ranking" is arbitrary (alphabetical), so recall@K is
    # typically 0 for small K. The value comes from perfect token cost
    # (we pretend the LLM can read everything).
    return {"paths": paths, "tokens": full_tokens, "latency_ms": wall, "raw_len": len(full_text)}


def build_full_context():
    """Concat every code file, cap at FULL_BUDGET_TOKENS. Returns the
    concatenated text and the environment-independent char count."""
    exts = {".rs", ".py", ".toml", ".md"}
    pieces = []
    total = 0
    # Rough char → token ratio = 3.5; use 3 to stay under budget.
    char_budget = FULL_BUDGET_TOKENS * 3
    for root, _, files in os.walk(REPO_ROOT):
        for f in sorted(files):
            if Path(f).suffix not in exts:
                continue
            try:
                body = (Path(root) / f).read_text(errors="replace")
            except OSError:
                continue
            rel = normalise_path(os.path.join(root, f))
            blob = f"// ===== {rel} =====\n{body}\n"
            if total + len(blob) > char_budget:
                break
            pieces.append(blob)
            total += len(blob)
        if total > char_budget:
            break
    return "".join(pieces)


# ---------------- Main ----------------
def main():
    env = {**os.environ}
    if not REPO_ROOT.exists():
        print(f"ERROR: repo root does not exist: {REPO_ROOT}")
        return

    # Pre-build FULL context once (shared across all queries)
    full_text = build_full_context()
    full_tokens = count_tokens(env, full_text)
    print(f"[repo-qa] FULL context = {full_tokens} tokens ({len(full_text)} chars)")
    print(f"[repo-qa] repo = {REPO_ROOT}")
    print(f"[repo-qa] N = {len(QA_ITEMS)} queries")

    results = []
    conditions = ["SOPHON", "GREP", "FULL"]

    for idx, item in enumerate(QA_ITEMS):
        query = item["query"]
        gold = item["gold"]
        print(f"\n[{idx+1}/{len(QA_ITEMS)}] {query[:70]}")
        row = {"query": query, "gold": gold, "conditions": {}}

        for cond in conditions:
            if cond == "SOPHON":
                r = run_sophon(env, query)
            elif cond == "GREP":
                r = run_grep(env, query)
            elif cond == "FULL":
                r = run_full(env, query, full_text, full_tokens)
            else:
                continue
            hits = score_hit(r["paths"], gold)
            r.update(hits)
            row["conditions"][cond] = r
            marks = "".join("✓" if r[f"recall@{k}"] else "·" for k in (1, 3, 5, 10))
            print(f"   {cond:<8} {marks}  top5={r['paths'][:3]}  toks={r['tokens']}  dt={r['latency_ms']:.0f}ms")
        results.append(row)

    # ---------- aggregation ----------
    print()
    print("=" * 95)
    print(f"REPO QA — N = {len(results)} queries, repo = {REPO_ROOT.name}")
    print("=" * 95)
    print(f"{'Condition':<10}" + "".join(f"{f'recall@{k}':>11}" for k in (1, 3, 5, 10))
          + f"{'mean tokens':>14}" + f"{'mean dt ms':>12}")
    print("-" * 95)
    for cond in conditions:
        rows = [r["conditions"][cond] for r in results if cond in r["conditions"]]
        if not rows:
            continue
        line = f"{cond:<10}"
        for k in (1, 3, 5, 10):
            hits = sum(1 for r in rows if r.get(f"recall@{k}"))
            line += f"{hits / len(rows) * 100:>10.1f}%"
        line += f"{statistics.mean(r['tokens'] for r in rows):>14.0f}"
        line += f"{statistics.mean(r['latency_ms'] for r in rows):>11.0f}"
        print(line)

    print()
    print("Interpretation guide:")
    print("  SOPHON recall @ high K with SMALL tokens = ideal")
    print("  GREP   recall @ low K but requires extra judgement on many hits")
    print("  FULL   perfect recall in principle, but huge token cost")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
