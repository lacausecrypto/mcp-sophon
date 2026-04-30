#!/usr/bin/env python3
"""
Real shell command outputs → `compress_output` measurement.

Closes the methodological gap that `real_session_capture.py` left:
that bench measures only what `git` captures (diffs + commit
messages), which is ~5-10 % of a real Claude Code session's tool
traffic. The other 30-40 % is shell command output — `cargo test`,
`gh run watch`, `npm install`, `kubectl get pods`, `ls -la`, etc.

This script runs a curated set of commands that match what an
agent loop ACTUALLY produces during this repo's development, and
measures `compress_output` against the real stdout/stderr captured
locally — no canned samples, no fabricated payloads.

What it runs (cwd = repo root, all read-only or idempotent):

  cargo test --workspace --lib --tests --exclude prompt-compressor
  cargo build --release --bin sophon -p mcp-integration
  cargo check --workspace
  git status
  git log --oneline -n 30
  git log -n 5
  git diff HEAD~3
  ls -laR sophon/crates  (capped depth)
  find sophon/crates -name "*.rs" | head -30
  grep -rn "TODO\\|FIXME" sophon/crates --include='*.rs'
  gh run list --limit 20         (network — skipped if `gh` missing)
  gh repo view --json …          (network — skipped if `gh` missing)

For each command we capture stdout+stderr, count tokens, run
through compress_output, report the saved %. Aggregate is
weighted by raw tokens.

All commands are SAFE: read-only or idempotent, no `git push`, no
state mutation. Cargo commands need ulimit raised on macOS — the
script does that automatically.

Running
=======

    python3 benchmarks/real_session_shell.py
    python3 benchmarks/real_session_shell.py --skip-cargo
    python3 benchmarks/real_session_shell.py --json
    python3 benchmarks/real_session_shell.py --anonymise
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOPHON_BIN = os.environ.get("SOPHON_BIN", str(ROOT / "sophon/target/release/sophon"))
SOPHON_DIR = ROOT / "sophon"


# Commands grouped by category so the report can break down per type.
# Each entry: (label, command_list, working_dir, category, slow_flag).
COMMANDS = [
    # --- Cargo (slow but representative — agents call these often) ---
    (
        "cargo test --workspace --lib --tests --exclude prompt-compressor",
        ["cargo", "test", "--workspace", "--lib", "--tests", "--exclude", "prompt-compressor", "--quiet"],
        SOPHON_DIR,
        "cargo",
        True,
    ),
    (
        "cargo check --workspace",
        ["cargo", "check", "--workspace", "--quiet"],
        SOPHON_DIR,
        "cargo",
        True,
    ),
    # --- git (fast, common) ---
    (
        "git status",
        ["git", "status"],
        ROOT,
        "git",
        False,
    ),
    (
        "git log --oneline -n 30",
        ["git", "log", "--oneline", "-n", "30"],
        ROOT,
        "git",
        False,
    ),
    (
        "git log -n 5 (verbose)",
        ["git", "log", "-n", "5"],
        ROOT,
        "git",
        False,
    ),
    (
        "git diff HEAD~3",
        ["git", "diff", "HEAD~3"],
        ROOT,
        "git",
        False,
    ),
    (
        "git log --stat -n 5",
        ["git", "log", "--stat", "-n", "5"],
        ROOT,
        "git",
        False,
    ),
    # --- filesystem search (high frequency in agent loops) ---
    (
        "ls -laR sophon/crates (capped)",
        ["bash", "-c", "ls -laR sophon/crates 2>&1 | head -200"],
        ROOT,
        "filesystem",
        False,
    ),
    (
        "find sophon/crates -name '*.rs' (head 60)",
        ["bash", "-c", "find sophon/crates -name '*.rs' | head -60"],
        ROOT,
        "filesystem",
        False,
    ),
    (
        "grep -rn TODO|FIXME sophon/crates",
        ["bash", "-c", "grep -rn 'TODO\\|FIXME' sophon/crates --include='*.rs' || true"],
        ROOT,
        "filesystem",
        False,
    ),
    (
        "wc -l sophon/crates/*/src/*.rs",
        ["bash", "-c", "wc -l sophon/crates/*/src/*.rs 2>/dev/null | head -50"],
        ROOT,
        "filesystem",
        False,
    ),
    # --- gh (network, skip if missing) ---
    (
        "gh run list --limit 20",
        ["gh", "run", "list", "--limit", "20"],
        ROOT,
        "github",
        False,
    ),
    (
        "gh repo view --json description,homepageUrl",
        ["gh", "repo", "view", "--json", "description,homepageUrl"],
        ROOT,
        "github",
        False,
    ),
]


def raise_ulimit():
    """macOS defaults to 256 open files which kills cargo. Match
    the project's documented `ulimit -n 16384` recommendation."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(16384, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def capture_command(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[str, int]:
    """Run a command, capture combined stdout+stderr, return (output,
    elapsed_ms). Failure is non-fatal — we record the error output
    just like a real agent would receive it."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        out = p.stdout
    except subprocess.TimeoutExpired as e:
        out = f"(timeout after {timeout}s)\n{e.stdout or ''}"
    except FileNotFoundError as e:
        out = f"(command not found: {e})"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return out, elapsed_ms


def rpc_compress_output(command: str, output: str) -> dict:
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "shell-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_output",
                "arguments": {"command": command, "output": output},
            },
        },
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    p = subprocess.run([SOPHON_BIN, "serve"], input=raw, capture_output=True, text=True, timeout=120)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id") == 1:
            return d.get("result", {}).get("structuredContent") or {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cargo", action="store_true", help="Skip cargo commands (slow on cold cache).")
    parser.add_argument("--skip-network", action="store_true", help="Skip gh commands (need network + auth).")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--anonymise",
        action="store_true",
        help=(
            "Scrub the command label in the output (replace with "
            "`<command #N>`). Useful when sharing results from a "
            "private repo where command lines might reference "
            "internal paths or secret-bearing args."
        ),
    )
    args = parser.parse_args()

    if not Path(SOPHON_BIN).exists():
        print(f"sophon binary not found at {SOPHON_BIN!r}. Run cargo build --release.", file=sys.stderr)
        return 2

    raise_ulimit()

    # Filter the command list.
    selected = []
    for label, cmd, cwd, cat, slow in COMMANDS:
        if slow and args.skip_cargo:
            continue
        if cat == "github" and (args.skip_network or not shutil.which("gh")):
            continue
        selected.append((label, cmd, cwd, cat, slow))

    print(f"Running {len(selected)} real shell commands → compress_output…", file=sys.stderr)

    rows = []
    for i, (label, cmd, cwd, cat, _slow) in enumerate(selected, 1):
        print(f"  [{i:>2}/{len(selected)}] {label}", file=sys.stderr)
        output, elapsed_ms = capture_command(cmd, cwd)
        if not output.strip():
            print(f"    (empty output, skipping)", file=sys.stderr)
            continue

        result = rpc_compress_output(label, output)
        raw_tokens = int(result.get("original_tokens") or 0)
        compressed_tokens = int(result.get("compressed_tokens") or 0)
        saved = (raw_tokens - compressed_tokens) / raw_tokens * 100 if raw_tokens else 0.0
        printable_label = f"<command #{i}>" if args.anonymise else label
        rows.append({
            "label": printable_label,
            "category": cat,
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "saved_pct": saved,
            "filter": result.get("filter_name", "?"),
            "command_elapsed_ms": elapsed_ms,
            "raw_bytes": len(output),
        })

    # Aggregates
    total_raw = sum(r["raw_tokens"] for r in rows)
    total_compressed = sum(r["compressed_tokens"] for r in rows)
    aggregate_saved = (total_raw - total_compressed) / total_raw * 100 if total_raw else 0.0

    # Per-category breakdown
    by_cat = {}
    for r in rows:
        b = by_cat.setdefault(r["category"], {"raw": 0, "compressed": 0, "n": 0})
        b["raw"] += r["raw_tokens"]
        b["compressed"] += r["compressed_tokens"]
        b["n"] += 1
    cat_rows = []
    for cat, b in by_cat.items():
        s = (b["raw"] - b["compressed"]) / b["raw"] * 100 if b["raw"] else 0.0
        cat_rows.append({"category": cat, "n": b["n"], "raw_tokens": b["raw"], "compressed_tokens": b["compressed"], "saved_pct": s})
    cat_rows.sort(key=lambda c: c["raw_tokens"], reverse=True)

    summary = {
        "commands_run": len(rows),
        "total_raw_tokens": total_raw,
        "total_compressed_tokens": total_compressed,
        "aggregate_saved_pct": aggregate_saved,
        "per_category": cat_rows,
        "per_command": rows,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 96)
    print(f"Real shell command outputs → compress_output ({len(rows)} commands)")
    print("=" * 96)

    print(f"  {'command':<55} {'cat':<10} {'raw':>7} {'out':>5} {'saved%':>8} {'filter':<14}")
    print(f"  {'-'*55} {'-'*10} {'-'*7} {'-'*5} {'-'*8} {'-'*14}")
    for r in sorted(rows, key=lambda r: -r["saved_pct"]):
        label = r["label"][:54]
        print(
            f"  {label:<55} {r['category']:<10} "
            f"{r['raw_tokens']:>7} {r['compressed_tokens']:>5} "
            f"{r['saved_pct']:>7.1f}% {r['filter']:<14}"
        )

    print()
    print("Per-category (weighted by raw tokens):")
    print(f"  {'category':<14} {'n':>3} {'raw':>8} {'compressed':>11} {'saved%':>8}")
    print(f"  {'-'*14} {'-'*3} {'-'*8} {'-'*11} {'-'*8}")
    for c in cat_rows:
        print(
            f"  {c['category']:<14} {c['n']:>3} {c['raw_tokens']:>8} "
            f"{c['compressed_tokens']:>11} {c['saved_pct']:>7.1f}%"
        )

    print()
    print(f"  Aggregate (weighted, all categories):  "
          f"{total_raw:>7} → {total_compressed:<5}   {aggregate_saved:>5.1f} % saved")

    print()
    print("Honest scope:")
    print("  * These are REAL local outputs, no canned samples.")
    print("  * Run on YOUR machine with YOUR repo state — numbers are")
    print("    local, not portable. Re-run to refresh.")
    print("  * Cargo + gh dominate when run hot vs cold cache.")
    print("  * --anonymise scrubs command labels but not output content")
    print("    fed to sophon (in-process).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
