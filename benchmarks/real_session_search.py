#!/usr/bin/env python3
"""
Real search outputs → compress_output measurement.

Search loops are a major chunk of agent tool traffic that
real_session_shell only sampled briefly. A typical Claude Code
session calls grep / find / glob / ripgrep dozens of times per
hour while the agent is locating callers, references, and
patterns. This script puts compress_output through that real
shape: real searches against THIS repo's working tree, no canned
samples.

What it runs (read-only, all relative to repo root):

  grep   — multi-pattern recursive grep, with and without context
  find   — find by name pattern, with -exec, with size filters
  rg     — ripgrep variants if installed (preferred by many agents)
  ls     — `ls -laR` capped, plus deep `find` analogues
  wc     — `wc -l` over the workspace

Each search is run twice — once exactly and once via Sophon's
compress_output — so the report can compare token cost of the
raw output to the compressed form.

Running
=======

    python3 benchmarks/real_session_search.py
    python3 benchmarks/real_session_search.py --json
    python3 benchmarks/real_session_search.py --anonymise
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOPHON_BIN = os.environ.get("SOPHON_BIN", str(ROOT / "sophon/target/release/sophon"))


# (label, [argv], category, optional executable required)
# All commands are read-only; each is a search shape that an
# agent loop is likely to issue while spelunking through code.
SEARCHES = [
    # Plain greps
    (
        "grep -rn 'fn ' sophon/crates",
        ["bash", "-c", "grep -rn 'fn ' sophon/crates --include='*.rs' | head -200 || true"],
        "grep",
        None,
    ),
    (
        "grep -rn 'pub use' sophon/crates",
        ["bash", "-c", "grep -rn 'pub use' sophon/crates --include='*.rs' || true"],
        "grep",
        None,
    ),
    (
        "grep -rn 'TODO|FIXME|XXX' sophon/crates",
        ["bash", "-c", "grep -rn 'TODO\\|FIXME\\|XXX' sophon/crates --include='*.rs' || true"],
        "grep",
        None,
    ),
    (
        "grep -rn -B 2 -A 2 'CancellationToken' sophon/crates",
        ["bash", "-c", "grep -rn -B 2 -A 2 'CancellationToken' sophon/crates --include='*.rs' || true"],
        "grep",
        None,
    ),
    (
        "grep -in 'compress' README.md CHANGELOG.md",
        ["bash", "-c", "grep -in 'compress' README.md CHANGELOG.md || true"],
        "grep",
        None,
    ),
    # find variants
    (
        "find sophon -name '*.rs' (head 200)",
        ["bash", "-c", "find sophon -name '*.rs' | head -200"],
        "find",
        None,
    ),
    (
        "find sophon -name '*.rs' -size +5k",
        ["bash", "-c", "find sophon -name '*.rs' -size +5k 2>/dev/null"],
        "find",
        None,
    ),
    (
        "find . -name 'Cargo.toml' (excl target)",
        ["bash", "-c", "find . -name 'Cargo.toml' -not -path './sophon/target/*' 2>/dev/null"],
        "find",
        None,
    ),
    (
        "find . -name '*.json' (head 80)",
        ["bash", "-c", "find . -name '*.json' -not -path './sophon/target/*' 2>/dev/null | head -80"],
        "find",
        None,
    ),
    # ls deep traversals
    (
        "ls -la sophon/crates",
        ["bash", "-c", "ls -la sophon/crates"],
        "ls",
        None,
    ),
    (
        "ls -laR sophon/crates/mcp-integration",
        ["bash", "-c", "ls -laR sophon/crates/mcp-integration"],
        "ls",
        None,
    ),
    # wc
    (
        "wc -l sophon/crates/*/src/*.rs (head 60)",
        ["bash", "-c", "wc -l sophon/crates/*/src/*.rs 2>/dev/null | head -60"],
        "wc",
        None,
    ),
    # ripgrep if available
    (
        "rg -n 'CompressionStrategy' sophon/crates",
        ["bash", "-c", "rg -n 'CompressionStrategy' sophon/crates --type rust 2>/dev/null"],
        "rg",
        "rg",
    ),
    (
        "rg --files sophon/crates -t rust",
        ["bash", "-c", "rg --files sophon/crates -t rust 2>/dev/null | head -200"],
        "rg",
        "rg",
    ),
]


def run_search(cmd: list[str], cwd: Path = ROOT, timeout: int = 60) -> str:
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
        return p.stdout
    except subprocess.TimeoutExpired as e:
        return f"(timeout)\n{e.stdout or ''}"
    except FileNotFoundError as e:
        return f"(command not found: {e})"


def rpc_compress_output(command: str, output: str) -> dict:
    payload = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "search-bench", "version": "0"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "compress_output",
                    "arguments": {"command": command, "output": output}}},
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--anonymise", action="store_true",
                        help="Replace search labels with `<search #N>`.")
    args = parser.parse_args()

    if not Path(SOPHON_BIN).exists():
        print(f"sophon binary not found at {SOPHON_BIN!r}", file=sys.stderr)
        return 2

    rows = []
    selected = []
    for label, cmd, cat, required in SEARCHES:
        if required and not shutil.which(required):
            print(f"  skipping {label} (missing: {required})", file=sys.stderr)
            continue
        selected.append((label, cmd, cat))

    print(f"Running {len(selected)} real searches → compress_output…", file=sys.stderr)

    for i, (label, cmd, cat) in enumerate(selected, 1):
        print(f"  [{i:>2}/{len(selected)}] {label}", file=sys.stderr)
        output = run_search(cmd)
        if not output.strip():
            print(f"    (empty, skipping)", file=sys.stderr)
            continue
        result = rpc_compress_output(label, output)
        raw_tokens = int(result.get("original_tokens") or 0)
        compressed_tokens = int(result.get("compressed_tokens") or 0)
        saved = (raw_tokens - compressed_tokens) / raw_tokens * 100 if raw_tokens else 0.0
        printable_label = f"<search #{i}>" if args.anonymise else label
        rows.append({
            "label": printable_label,
            "category": cat,
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "saved_pct": saved,
            "filter": result.get("filter_name", "?"),
            "raw_lines": output.count("\n") + (1 if output and not output.endswith("\n") else 0),
        })

    if not rows:
        print("no rows captured", file=sys.stderr)
        return 1

    total_raw = sum(r["raw_tokens"] for r in rows)
    total_compressed = sum(r["compressed_tokens"] for r in rows)
    aggregate_saved = (total_raw - total_compressed) / total_raw * 100 if total_raw else 0.0

    by_cat: dict[str, dict] = {}
    for r in rows:
        b = by_cat.setdefault(r["category"], {"raw": 0, "compressed": 0, "n": 0})
        b["raw"] += r["raw_tokens"]
        b["compressed"] += r["compressed_tokens"]
        b["n"] += 1
    cat_rows = [
        {
            "category": cat,
            "n": b["n"],
            "raw_tokens": b["raw"],
            "compressed_tokens": b["compressed"],
            "saved_pct": (b["raw"] - b["compressed"]) / b["raw"] * 100 if b["raw"] else 0.0,
        }
        for cat, b in by_cat.items()
    ]
    cat_rows.sort(key=lambda c: c["raw_tokens"], reverse=True)

    summary = {
        "searches_run": len(rows),
        "total_raw_tokens": total_raw,
        "total_compressed_tokens": total_compressed,
        "aggregate_saved_pct": aggregate_saved,
        "per_category": cat_rows,
        "per_search": rows,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 96)
    print(f"Real search outputs → compress_output ({len(rows)} searches)")
    print("=" * 96)
    print(f"  {'search':<55} {'cat':<6} {'raw':>7} {'out':>5} {'saved%':>8} {'filter':<14}")
    print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*14}")
    for r in sorted(rows, key=lambda r: -r["saved_pct"]):
        print(
            f"  {r['label'][:54]:<55} {r['category']:<6} "
            f"{r['raw_tokens']:>7} {r['compressed_tokens']:>5} "
            f"{r['saved_pct']:>7.1f}% {r['filter']:<14}"
        )

    print("\nPer-category (weighted by raw tokens):")
    print(f"  {'category':<10} {'n':>3} {'raw':>8} {'compressed':>11} {'saved%':>8}")
    print(f"  {'-'*10} {'-'*3} {'-'*8} {'-'*11} {'-'*8}")
    for c in cat_rows:
        print(
            f"  {c['category']:<10} {c['n']:>3} {c['raw_tokens']:>8} "
            f"{c['compressed_tokens']:>11} {c['saved_pct']:>7.1f}%"
        )
    print()
    print(f"  Aggregate: {total_raw:>7} → {total_compressed:<5}  "
          f"{aggregate_saved:>5.1f} % saved")
    print()
    print("Honest scope:")
    print("  * Search outputs are highly state-dependent (your repo's grep")
    print("    hits depend on what's actually in the tree).")
    print("  * grep with `-B/-A` context expands the output but typically")
    print("    compresses much harder because the surrounding context is")
    print("    redundant. Worth measuring both shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
