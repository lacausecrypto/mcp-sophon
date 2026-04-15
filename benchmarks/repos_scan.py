#!/usr/bin/env python3
"""
Rigorous public-repo benchmark for Sophon's codebase-navigator.

Runs navigate_codebase against 5 well-known GitHub repos (SHAs pinned
in ../sha_pins.json), measures:
  * Fresh scan wall-clock
  * Incremental scan wall-clock (same session, 2nd call)
  * Files scanned / symbols found / graph edges
  * Output digest token count at a fixed budget

Emits a JSON report to stdout + a pretty table to stderr.
"""
import json, subprocess, time, os, sys
from pathlib import Path

BIN = os.environ.get("SOPHON_BIN", "sophon")
REPOS_DIR = Path(os.environ.get("SOPHON_BENCH_REPOS", "./benchmarks/data/repos"))
REPOS = ["serde", "flask", "express", "gin", "sinatra"]
MAX_TOKENS = 1500


def run_session(root: Path, n_calls: int) -> tuple[list[dict], float]:
    """Run `n_calls` navigate_codebase against the same root in one server."""
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "bench", "version": "0"}}},
    ] + [
        {"jsonrpc": "2.0", "id": i, "method": "tools/call",
         "params": {"name": "navigate_codebase",
                    "arguments": {"root": str(root), "max_tokens": MAX_TOKENS}}}
        for i in range(1, n_calls + 1)
    ]
    payload = "".join(json.dumps(r) + "\n" for r in requests)

    t0 = time.perf_counter_ns()
    p = subprocess.run([BIN, "serve"], input=payload, capture_output=True,
                       text=True, timeout=180)
    t1 = time.perf_counter_ns()
    total_ms = (t1 - t0) / 1e6

    out = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("id") in range(1, n_calls + 1):
            out.append(d["result"]["structuredContent"])
    return out, total_ms


def get_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()


def bench_repo(repo_name: str) -> dict:
    repo = REPOS_DIR / repo_name
    assert repo.exists(), f"{repo} not cloned"
    sha = get_sha(repo)
    scans, total_ms = run_session(repo, n_calls=2)
    fresh = scans[0]
    incr = scans[1]

    return {
        "repo": repo_name,
        "sha": sha,
        "fresh_scan": fresh["scan_result"],
        "incremental_scan": incr["scan_result"],
        "files_scanned": fresh["total_files_scanned"],
        "symbols_found": fresh["total_symbols_found"],
        "edges_in_graph": fresh["edges_in_graph"],
        "digest_tokens": fresh["total_tokens"],
        "truncated": fresh["truncated"],
        "session_wall_ms": round(total_ms),
    }


def main():
    results = []
    print("=" * 90, file=sys.stderr)
    print(
        f"{'repo':<10}{'files':>7}{'symbols':>10}{'edges':>8}"
        f"{'digest':>9}{'fresh_src':>12}{'incr(u/a/r)':>14}{'wall':>8}",
        file=sys.stderr,
    )
    print("-" * 90, file=sys.stderr)
    for repo in REPOS:
        r = bench_repo(repo)
        results.append(r)
        fresh_src = r["fresh_scan"].get("scan_source", "?")
        incr = r["incremental_scan"]
        incr_uar = f"{incr.get('unchanged', 0)}/{incr.get('added', 0)}/{incr.get('removed', 0)}"
        print(
            f"{repo:<10}{r['files_scanned']:>7}{r['symbols_found']:>10}"
            f"{r['edges_in_graph']:>8}{r['digest_tokens']:>9}{fresh_src:>12}"
            f"{incr_uar:>14}{r['session_wall_ms']:>7}ms",
            file=sys.stderr,
        )
    print("-" * 90, file=sys.stderr)
    # Emit machine-readable on stdout
    print(json.dumps({"bench": "public_repo_scan",
                      "max_tokens": MAX_TOKENS,
                      "repos": results}, indent=2))


if __name__ == "__main__":
    main()
