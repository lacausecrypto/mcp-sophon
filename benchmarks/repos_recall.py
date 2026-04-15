#!/usr/bin/env python3
"""
Recall@K benchmark for codebase-navigator against real GitHub commits.

Methodology:
  1. For each of the 5 cloned repos, take the N most recent commits
     that touch at least one source file our extractors know about.
  2. For each such commit:
     - query  = commit message (subject line only)
     - truth  = the set of files that commit actually modified, filtered
                to the ones our extractors would scan.
     - run navigate_codebase with root=repo, query=commit_msg,
       max_tokens big enough that no useful file is cut by budget.
     - take the top-K paths from the digest and check intersection
       with truth.
  3. Report recall@5 and recall@10 per repo + pooled across all repos.

This is an honest measure: it asks "when an agent is told a bug-fix
message from this repo and queries Sophon, does Sophon surface the
right files?". Lower bound, because PageRank without the full commit
history can only rank what the current tree structure implies.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

BIN = os.environ.get("SOPHON_BIN", "sophon")
REPOS_DIR = Path(os.environ.get("SOPHON_BENCH_REPOS", "./benchmarks/data/repos"))
REPOS = ["serde", "flask", "express", "gin", "sinatra"]
COMMITS_PER_REPO = 10
SUPPORTED_EXT = {
    "rs", "py", "pyi", "js", "jsx", "mjs", "cjs", "ts", "tsx", "go",
    "java", "kt", "kts", "swift", "c", "h", "cc", "cpp", "cxx",
    "hpp", "hh", "hxx", "rb", "rake", "gemspec", "php", "phtml",
}


def git_commits_with_files(repo: Path, n: int) -> list[dict]:
    """Return the last `n` commits that touched at least one supported file."""
    # `--name-only --pretty=format:"<SHA>|||<subject>"` gives us one
    # pair of "header + file list" per commit, separated by a blank line.
    out = subprocess.check_output(
        [
            "git", "-C", str(repo), "log",
            "--no-merges",
            f"-{n * 3}",  # over-sample so we can filter
            "--name-only",
            "--pretty=format:COMMIT|||%H|||%s",
        ],
        text=True,
    )
    commits: list[dict] = []
    current: dict | None = None
    for line in out.splitlines():
        if line.startswith("COMMIT|||"):
            if current and current["files"]:
                commits.append(current)
                if len(commits) >= n:
                    break
            parts = line.split("|||", 2)
            current = {"sha": parts[1], "subject": parts[2], "files": []}
        elif line.strip() and current is not None:
            ext = line.rsplit(".", 1)[-1].lower() if "." in line else ""
            if ext in SUPPORTED_EXT:
                current["files"].append(line.strip())
    if current and current["files"] and current not in commits:
        commits.append(current)
    return commits[:n]


def run_navigate(root: Path, query: str, max_tokens: int = 3000) -> list[str]:
    """Run navigate_codebase and return the ordered list of file paths."""
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "bench", "version": "0"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "navigate_codebase",
                    "arguments": {
                        "root": str(root),
                        "query": query,
                        "max_tokens": max_tokens,
                        "force_rescan": True,
                    }}},
    ]
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    p = subprocess.run([BIN, "serve"], input=payload, capture_output=True,
                       text=True, timeout=120)
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("id") == 1:
            files = d["result"]["structuredContent"].get("files", [])
            return [f["path"] for f in files]
    return []


def recall_at(predicted: list[str], truth: set[str], k: int) -> float:
    """Fraction of `truth` files that appear in `predicted[:k]`."""
    if not truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = sum(1 for t in truth if t in top_k)
    return hits / len(truth)


def bench_repo(repo_name: str) -> dict:
    repo = REPOS_DIR / repo_name
    commits = git_commits_with_files(repo, COMMITS_PER_REPO)
    per_commit: list[dict] = []
    for c in commits:
        truth = set(c["files"])
        t0 = time.perf_counter_ns()
        predicted = run_navigate(repo, c["subject"])
        t1 = time.perf_counter_ns()
        r5 = recall_at(predicted, truth, 5)
        r10 = recall_at(predicted, truth, 10)
        per_commit.append({
            "sha": c["sha"][:10],
            "subject": c["subject"][:80],
            "truth_files": sorted(truth),
            "top_10": predicted[:10],
            "recall_at_5": r5,
            "recall_at_10": r10,
            "latency_ms": round((t1 - t0) / 1e6),
        })
    r5s = [c["recall_at_5"] for c in per_commit]
    r10s = [c["recall_at_10"] for c in per_commit]
    return {
        "repo": repo_name,
        "n_commits": len(per_commit),
        "mean_recall_at_5": sum(r5s) / len(r5s) if r5s else 0.0,
        "mean_recall_at_10": sum(r10s) / len(r10s) if r10s else 0.0,
        "per_commit": per_commit,
    }


def main():
    results = []
    print("=" * 90, file=sys.stderr)
    print(f"{'repo':<10}{'n':>5}{'recall@5':>14}{'recall@10':>14}{'avg_lat_ms':>14}",
          file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for repo in REPOS:
        r = bench_repo(repo)
        results.append(r)
        avg_lat = (
            sum(c["latency_ms"] for c in r["per_commit"]) / len(r["per_commit"])
            if r["per_commit"] else 0.0
        )
        print(
            f"{repo:<10}{r['n_commits']:>5}"
            f"{r['mean_recall_at_5'] * 100:>13.1f}%"
            f"{r['mean_recall_at_10'] * 100:>13.1f}%"
            f"{avg_lat:>12.0f}ms",
            file=sys.stderr,
        )
    print("-" * 90, file=sys.stderr)
    # Pooled average
    all_r5 = [c["recall_at_5"] for r in results for c in r["per_commit"]]
    all_r10 = [c["recall_at_10"] for r in results for c in r["per_commit"]]
    pooled5 = sum(all_r5) / len(all_r5) if all_r5 else 0.0
    pooled10 = sum(all_r10) / len(all_r10) if all_r10 else 0.0
    print(f"{'POOLED':<10}{len(all_r5):>5}{pooled5 * 100:>13.1f}%{pooled10 * 100:>13.1f}%",
          file=sys.stderr)
    print(json.dumps({"bench": "public_repo_recall",
                      "commits_per_repo": COMMITS_PER_REPO,
                      "repos": results,
                      "pooled_recall_at_5": pooled5,
                      "pooled_recall_at_10": pooled10,
                      }, indent=2))


if __name__ == "__main__":
    main()
