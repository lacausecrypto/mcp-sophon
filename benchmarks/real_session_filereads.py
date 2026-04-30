#!/usr/bin/env python3
"""
Real file-read content → compress_prompt measurement.

Closes a second methodological gap: real Claude Code sessions
spend most of their tokens on file reads — `Read` tool calls that
return file contents which then sit in the prompt across many
follow-up turns. `compress_output_per_command` measured the win
on shell stdout; this script measures the win on file contents
fed through `compress_prompt`.

Methodology
===========

For each representative file in this repo (chosen to cover
size + language variety), we:

  1. Read the file as-is.
  2. Build a synthetic agent prompt:
       <file content>
       Question: <query>
  3. Run it through `compress_prompt` with the query.
  4. Compare original tokens vs compressed tokens.

We do this with **three queries per file** to surface query-
dependence: a specific question (narrow scope), a broad question
("what does this do?"), and a refactoring question. The same
file with different queries produces different compressed
outputs because compress_prompt routes by topic match.

Files measured
==============

  small  (~70 lines)   sophon/Cargo.toml                — TOML, dense
  medium (~300 lines)  memory-manager/src/lib.rs        — Rust API
  large  (~600 lines)  output-compressor/src/strategy.rs — Rust impl
  XL     (~1000 lines) semantic-retriever/src/chunker.rs — Rust impl
  prose  (~520 lines)  README.md                        — Markdown

Running
=======

    python3 benchmarks/real_session_filereads.py
    python3 benchmarks/real_session_filereads.py --json
    python3 benchmarks/real_session_filereads.py --anonymise
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOPHON_BIN = os.environ.get("SOPHON_BIN", str(ROOT / "sophon/target/release/sophon"))


# (relative path, "size bucket", language tag)
FILES = [
    ("sophon/Cargo.toml", "small", "toml"),
    ("sophon/crates/memory-manager/src/lib.rs", "medium", "rust"),
    ("sophon/crates/output-compressor/src/strategy.rs", "large", "rust"),
    ("sophon/crates/semantic-retriever/src/chunker.rs", "xl", "rust"),
    ("README.md", "prose", "markdown"),
    ("benchmarks/real_session_capture.py", "medium", "python"),
]

# Queries tested against EVERY file. The same file with a
# different query gets a different compressed prompt because
# the topic-routing keeps only sections relevant to the query.
QUERIES = [
    ("specific", "How is the cancellation token registered and removed?"),
    ("broad", "What does this file do and how does it fit into the project?"),
    ("refactor", "Refactor this code to improve its testability."),
]


def rpc_compress_prompt(prompt: str, query: str, max_tokens: int = 2000) -> dict:
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "filereads-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_prompt",
                "arguments": {
                    "prompt": prompt,
                    "query": query,
                    "max_tokens": max_tokens,
                },
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


def rpc_count_tokens(text: str) -> int:
    payload = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "filereads-bench", "version": "0"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "count_tokens", "arguments": {"text": text}}},
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    p = subprocess.run([SOPHON_BIN, "serve"], input=raw, capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id") == 1:
            sc = d.get("result", {}).get("structuredContent") or {}
            return int(sc.get("token_count") or 0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="compress_prompt budget per call (default 2000)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--anonymise",
        action="store_true",
        help=(
            "Replace file paths with `file_<i>.<ext>` placeholders "
            "in the output. Content is still fed to sophon for "
            "measurement but the labels are scrubbed for sharing."
        ),
    )
    args = parser.parse_args()

    if not Path(SOPHON_BIN).exists():
        print(f"sophon binary not found at {SOPHON_BIN!r}", file=sys.stderr)
        return 2

    rows = []
    for i, (rel_path, size_bucket, lang) in enumerate(FILES, 1):
        full = ROOT / rel_path
        if not full.exists():
            print(f"  skipping missing file: {rel_path}", file=sys.stderr)
            continue
        content = full.read_text(errors="replace")
        raw_tokens = rpc_count_tokens(content)

        if args.anonymise:
            label = f"file_{i:02d}.{full.suffix.lstrip('.')}"
        else:
            label = rel_path

        print(f"  [{i}/{len(FILES)}] {label} ({raw_tokens} tokens, {size_bucket}/{lang})", file=sys.stderr)

        for tag, q in QUERIES:
            result = rpc_compress_prompt(content, q, max_tokens=args.max_tokens)
            compressed_tokens = int(result.get("token_count") or 0)
            saved = (raw_tokens - compressed_tokens) / raw_tokens * 100 if raw_tokens else 0.0
            included = result.get("included_sections", [])
            excluded = result.get("excluded_sections", [])
            rows.append({
                "file": label,
                "size_bucket": size_bucket,
                "language": lang,
                "query_tag": tag,
                "raw_tokens": raw_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_pct": saved,
                "included_section_count": len(included) if isinstance(included, list) else 0,
                "excluded_section_count": len(excluded) if isinstance(excluded, list) else 0,
            })

    if not rows:
        print("no rows — nothing measured", file=sys.stderr)
        return 1

    # Aggregates
    total_raw = sum(r["raw_tokens"] for r in rows)
    total_compressed = sum(r["compressed_tokens"] for r in rows)
    aggregate_saved = (total_raw - total_compressed) / total_raw * 100 if total_raw else 0.0

    # Per-language breakdown
    by_lang: dict[str, dict] = {}
    for r in rows:
        b = by_lang.setdefault(r["language"], {"raw": 0, "compressed": 0, "n": 0})
        b["raw"] += r["raw_tokens"]
        b["compressed"] += r["compressed_tokens"]
        b["n"] += 1
    lang_rows = [
        {
            "language": L,
            "n": b["n"],
            "raw_tokens": b["raw"],
            "compressed_tokens": b["compressed"],
            "saved_pct": (b["raw"] - b["compressed"]) / b["raw"] * 100 if b["raw"] else 0.0,
        }
        for L, b in by_lang.items()
    ]
    lang_rows.sort(key=lambda r: r["raw_tokens"], reverse=True)

    # Per-query-tag breakdown (does query specificity matter?)
    by_tag: dict[str, dict] = {}
    for r in rows:
        b = by_tag.setdefault(r["query_tag"], {"raw": 0, "compressed": 0, "n": 0})
        b["raw"] += r["raw_tokens"]
        b["compressed"] += r["compressed_tokens"]
        b["n"] += 1
    tag_rows = [
        {
            "query_tag": T,
            "n": b["n"],
            "raw_tokens": b["raw"],
            "compressed_tokens": b["compressed"],
            "saved_pct": (b["raw"] - b["compressed"]) / b["raw"] * 100 if b["raw"] else 0.0,
        }
        for T, b in by_tag.items()
    ]

    summary = {
        "files_measured": len({r["file"] for r in rows}),
        "queries_per_file": len(QUERIES),
        "total_measurements": len(rows),
        "aggregate_raw_tokens": total_raw,
        "aggregate_compressed_tokens": total_compressed,
        "aggregate_saved_pct": aggregate_saved,
        "per_language": lang_rows,
        "per_query_tag": tag_rows,
        "per_measurement": rows,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 96)
    print(f"Real file-read content → compress_prompt "
          f"({summary['files_measured']} files × {summary['queries_per_file']} queries = "
          f"{summary['total_measurements']} measurements)")
    print("=" * 96)

    print(f"\n  {'file':<55} {'q':<9} {'raw':>6} {'out':>5} {'saved%':>8}")
    print(f"  {'-'*55} {'-'*9} {'-'*6} {'-'*5} {'-'*8}")
    for r in rows:
        print(
            f"  {r['file'][:54]:<55} {r['query_tag']:<9} "
            f"{r['raw_tokens']:>6} {r['compressed_tokens']:>5} "
            f"{r['saved_pct']:>7.1f}%"
        )

    print("\nPer-language (weighted by raw tokens):")
    print(f"  {'language':<12} {'n':>3} {'raw':>7} {'compressed':>11} {'saved%':>8}")
    print(f"  {'-'*12} {'-'*3} {'-'*7} {'-'*11} {'-'*8}")
    for L in lang_rows:
        print(
            f"  {L['language']:<12} {L['n']:>3} {L['raw_tokens']:>7} "
            f"{L['compressed_tokens']:>11} {L['saved_pct']:>7.1f}%"
        )

    print("\nPer-query-shape (does specificity matter?):")
    print(f"  {'query':<12} {'n':>3} {'raw':>7} {'compressed':>11} {'saved%':>8}")
    print(f"  {'-'*12} {'-'*3} {'-'*7} {'-'*11} {'-'*8}")
    for T in tag_rows:
        print(
            f"  {T['query_tag']:<12} {T['n']:>3} {T['raw_tokens']:>7} "
            f"{T['compressed_tokens']:>11} {T['saved_pct']:>7.1f}%"
        )

    print()
    print(f"  Aggregate: {total_raw:>6} → {total_compressed:<5} "
          f"{aggregate_saved:>5.1f} % saved")
    print()
    print("Honest scope:")
    print("  * Files chosen for size + language variety, not random sample.")
    print("  * Queries are 3 typical agent shapes — specific / broad / refactor.")
    print("  * compress_prompt routes by section topic; results vary if your")
    print("    queries are vocabulary-distant from the source content.")
    print("  * Run on YOUR repo state — re-run after editing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
