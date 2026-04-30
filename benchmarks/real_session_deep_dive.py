#!/usr/bin/env python3
"""
Deep-dive companion to `real_session_capture.py`.

Runs 4 complementary measurements on the same git history:

  A. Per-commit-type breakdown  — feat / fix / chore / style / docs /
                                   test / perf / bench all compress
                                   differently because the diff /
                                   message ratio differs.
  B. Bombs                       — top 5 largest commits by raw tokens,
                                   showing what compression does to
                                   the worst-case turns.
  C. USD translation             — apply Anthropic Claude 3.5 Sonnet
                                   input pricing ($3/MT) + prompt-cache
                                   read ($0.30/MT) to the saved tokens
                                   and report concrete dollar savings.
  D. Rolling-summary effect      — replay the session WITH and WITHOUT
                                   SOPHON_ROLLING_SUMMARY=1 enabled,
                                   compare snapshot tokens.

Goal: turn the headline 93 % from `real_session_capture.py` into a
texture — where does the compression come from, where doesn't it,
what does it cost the user to skip Sophon in pure $.

Running
=======

    python3 benchmarks/real_session_deep_dive.py
    python3 benchmarks/real_session_deep_dive.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOPHON = os.environ.get("SOPHON_BIN", str(ROOT / "sophon/target/release/sophon"))

# Anthropic Claude pricing (per million tokens, USD). Source:
# anthropic.com/pricing as of 2026-04. Update via CLI flag if it
# drifts.
#
# Defaults reflect **Opus 4.7** because that's what this repo's
# real session was actually run on — using Sonnet pricing here
# would understate the dollar savings by ~5×.
#
# Sonnet 4.6 / 3.5 (kept for the `--model sonnet` override):
#   input        $ 3.00 / MT
#   cache_read   $ 0.30 / MT
#   output       $15.00 / MT
#
# Haiku 4.5 (kept for `--model haiku`):
#   input        $ 0.80 / MT  (approximate; check anthropic.com)
#   cache_read   $ 0.08 / MT
#   output       $ 4.00 / MT
PRICING = {
    "opus": {
        "input":      15.00,
        "cache_read":  1.50,
        "output":     75.00,
    },
    "sonnet": {
        "input":       3.00,
        "cache_read":  0.30,
        "output":     15.00,
    },
    "haiku": {
        "input":       0.80,
        "cache_read":  0.08,
        "output":      4.00,
    },
}
DEFAULT_MODEL = "opus"


def git_commits(since: str, until: str) -> list[dict]:
    fmt = "%H%x1f%s%x1f%b%x1f%an%x1f%ai%x1e"
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--reverse", f"--format={fmt}", f"{since}..{until}"],
        text=True,
    )
    out = []
    for entry in raw.split("\x1e"):
        entry = entry.lstrip("\n")
        if not entry.strip():
            continue
        parts = entry.split("\x1f")
        if len(parts) < 5:
            continue
        sha, subject, body, author, date = parts[:5]
        diff = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "--format=", "--unified=3", sha],
            text=True,
            errors="replace",
        )
        files = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "--name-only", "--format=", sha],
            text=True,
        ).splitlines()
        out.append({
            "sha": sha[:8],
            "subject": subject,
            "body": body,
            "diff": diff,
            "files": [f for f in files if f.strip()],
        })
    return out


def commit_type(subject: str) -> str:
    """Conventional-commit prefix or `(other)` if it doesn't match."""
    for prefix in ("feat", "fix", "chore", "style", "docs", "test", "perf", "bench", "ci", "refactor"):
        if subject.startswith(prefix + "(") or subject.startswith(prefix + ":"):
            return prefix
    return "(other)"


def is_sophon_own_repo() -> bool:
    """Same heuristic as real_session_capture.is_sophon_own_repo —
    duplicated rather than imported because both scripts are
    standalone CLIs we want users to pip-install / curl directly."""
    cargo = ROOT / "sophon" / "Cargo.toml"
    if not cargo.exists():
        return False
    try:
        text = cargo.read_text(errors="ignore")
        if "sophon-core" not in text and "mcp-integration" not in text:
            return False
    except OSError:
        return False
    try:
        remote = subprocess.check_output(
            ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return "mcp-sophon" in remote
    except subprocess.CalledProcessError:
        return False


def rpc(name: str, arguments: dict, env: dict | None = None) -> dict:
    payload = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "deep-dive", "version": "0"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": name, "arguments": arguments}},
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=120, env=full_env)
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


def commit_to_messages(c: dict, max_diff_chars: int = 4000) -> list[dict]:
    msgs = [
        {"role": "user", "content": f"Task: {c['subject']}"},
        {"role": "assistant", "content": (c["body"] or "(no body)") + f"\nFiles: {', '.join(c['files'])}"},
    ]
    diff = c["diff"]
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + f"\n... (truncated, original {len(c['diff'])} chars)"
    if c["files"]:
        msgs.append({"role": "assistant", "content": json.dumps({
            "type": "tool_use", "name": "apply_diff", "input": {"files": c["files"][:5]}
        })})
        msgs.append({"role": "user", "content": json.dumps({
            "type": "tool_result", "tool_use_id": c["sha"], "content": diff,
        })})
    return msgs


# ============================================================================
# A. Per-commit-type breakdown
# ============================================================================
def per_type(commits: list[dict], anonymise: bool = False) -> dict:
    """Per-commit-type aggregate. Bucket key is the conventional-
    commit type (`feat`/`fix`/…) which is structural, not personal,
    so it's safe to print regardless of `--anonymise`.
    Per-bucket the only stored ID is the SHA — only used internally
    here, never returned, so no anonymisation needed at this level."""
    _ = anonymise  # bucket output has no identifying fields
    buckets: dict[str, list] = defaultdict(list)
    for c in commits:
        result = rpc("compress_output", {"command": "git diff", "output": c["diff"][:16000]})
        buckets[commit_type(c["subject"])].append({
            "raw": int(result.get("original_tokens") or 0),
            "compressed": int(result.get("compressed_tokens") or 0),
        })
    rows = []
    for t, entries in buckets.items():
        agg_raw = sum(e["raw"] for e in entries)
        agg_compressed = sum(e["compressed"] for e in entries)
        saved = (agg_raw - agg_compressed) / agg_raw * 100 if agg_raw else 0.0
        rows.append({
            "type": t,
            "n": len(entries),
            "raw_tokens": agg_raw,
            "compressed_tokens": agg_compressed,
            "saved_pct": saved,
        })
    rows.sort(key=lambda r: r["raw_tokens"], reverse=True)
    return {"per_type": rows}


# ============================================================================
# B. Bombs — top 5 by raw tokens
# ============================================================================
def bombs(commits: list[dict], top_n: int = 5, anonymise: bool = False) -> dict:
    """Top-N largest commits by raw tokens. The output dict carries
    `sha` + `subject`, both of which can leak identifying details
    when this script is run against a private repo. With
    `anonymise=True`, both fields are scrubbed: SHA → commit_NNN
    placed by sort-position, subject → '<type>: <redacted>'.
    The compression measurement is unaffected (we still feed the
    real diff to sophon)."""
    enriched = []
    for i, c in enumerate(commits):
        result = rpc("compress_output", {"command": "git diff", "output": c["diff"][:32000]})
        raw = int(result.get("original_tokens") or 0)
        compressed = int(result.get("compressed_tokens") or 0)
        if anonymise:
            sha_out = f"commit_{i + 1:03d}"
            subject_out = f"{commit_type(c['subject'])}: <redacted>"
        else:
            sha_out = c["sha"]
            subject_out = c["subject"][:70]
        enriched.append({
            "sha": sha_out,
            "subject": subject_out,
            "raw_tokens": raw,
            "compressed_tokens": compressed,
            "saved_pct": (raw - compressed) / raw * 100 if raw else 0.0,
            "filter": result.get("filter_name", "?"),
        })
    enriched.sort(key=lambda e: e["raw_tokens"], reverse=True)
    return {"bombs": enriched[:top_n]}


# ============================================================================
# C. USD translation
# ============================================================================
def usd_savings(
    tail_raw: int,
    tail_compressed: int,
    per_diff_raw: int,
    per_diff_compressed: int,
    model: str,
) -> dict:
    """Two cost models, parametrised by `model` (`opus` / `sonnet` /
    `haiku`):
       1. Naive: every token costs the input rate (no caching, fresh
          request). Saved tokens directly translate to $.
       2. Realistic: with prompt caching, the conversation history
          is read at the cache-read rate after the first call.
          Sophon's compression reduces the tail that has to be
          re-read each turn. For a 25-turn loop where the cached
          prefix is read 25× and Sophon shrinks the tail, saved $
          = saved_tail_tokens * 25 * cache_read_price + saved_diff
          _tokens * input_price (diffs are per-turn, not cached)."""
    rates = PRICING[model]
    saved_history_tokens = tail_raw - tail_compressed
    saved_diffs_tokens = per_diff_raw - per_diff_compressed
    return {
        "model": model,
        "saved_history_tokens": saved_history_tokens,
        "saved_diffs_tokens": saved_diffs_tokens,
        "naive_input_usd":
            (saved_history_tokens + saved_diffs_tokens) / 1_000_000 * rates["input"],
        "with_caching_25turn_reads_usd":
            saved_history_tokens / 1_000_000 * 25 * rates["cache_read"]
            + saved_diffs_tokens / 1_000_000 * rates["input"],
        "pricing_input_per_mt_usd": rates["input"],
        "pricing_cache_read_per_mt_usd": rates["cache_read"],
    }


# ============================================================================
# D. Rolling-summary effect on real session
# ============================================================================
def rolling_compare(commits: list[dict]) -> dict:
    """Build the full message stream, run compress_history WITHOUT
    rolling and WITH SOPHON_ROLLING_SUMMARY=1. Compare snapshot
    tokens. The rolling path persists state across calls so we
    do a single tail snapshot per scenario for a fair comparison."""
    accumulated: list[dict] = []
    for c in commits:
        accumulated.extend(commit_to_messages(c))
    args = {"messages": accumulated, "max_tokens": 4000}

    no_rolling = rpc("compress_history", args, env={"SOPHON_NO_LLM_SUMMARY": "1"})
    with_rolling = rpc("compress_history", args, env={
        "SOPHON_ROLLING_SUMMARY": "1",
        "SOPHON_ROLLING_THRESHOLD": "20",
        "SOPHON_NO_LLM_SUMMARY": "1",
    })
    return {
        "rolling_off_tokens": int(no_rolling.get("token_count") or 0),
        "rolling_on_tokens": int(with_rolling.get("token_count") or 0),
        "rolling_off_msgs": int(no_rolling.get("original_message_count") or 0),
        "rolling_on_msgs": int(with_rolling.get("original_message_count") or 0),
    }


# ============================================================================
# Driver
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="b197760~1")
    parser.add_argument("--until", default="HEAD")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--model",
        choices=list(PRICING.keys()),
        default=DEFAULT_MODEL,
        help=(
            "Pricing model to use for the USD translation. Defaults to "
            "`opus` because that's what this repo's real session was "
            "run on — using sonnet/haiku here would understate savings."
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--anonymise",
        action="store_true",
        help=(
            "Scrub identifying fields (commit SHA → commit_NNN, "
            "subject → '<type>: <redacted>') before any stdout / "
            "JSON output. Diffs are still fed to sophon for "
            "measurement but never printed. Use when running "
            "against a private repo."
        ),
    )
    args = parser.parse_args()

    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}", file=sys.stderr)
        return 2

    if not args.anonymise and not is_sophon_own_repo():
        print(
            "\n"
            "================================================================\n"
            " WARNING: --anonymise is OFF and this looks like a NON-Sophon repo.\n"
            "\n"
            " Section [B] bombs will print real commit SHAs + subjects.\n"
            " --json output will include the same per-commit identifiers.\n"
            " Diffs are passed to sophon for measurement (in-process; not logged).\n"
            "\n"
            " If this repo is private or sensitive, re-run with `--anonymise`.\n"
            "================================================================\n",
            file=sys.stderr,
        )

    print(f"Walking git log {args.since}..{args.until}…", file=sys.stderr)
    commits = git_commits(args.since, args.until)
    print(f"  → {len(commits)} commits", file=sys.stderr)
    if not commits:
        return 1

    print("[A] Per-commit-type breakdown…", file=sys.stderr)
    a = per_type(commits, anonymise=args.anonymise)

    print("[B] Bombs (top largest by raw tokens)…", file=sys.stderr)
    b = bombs(commits, args.top_n, anonymise=args.anonymise)

    print("[D] Rolling-summary on/off compare…", file=sys.stderr)
    d = rolling_compare(commits)

    # For section C, we re-derive aggregates from per-type.
    tail_total_raw = sum(e["raw_tokens"] for e in a["per_type"])
    tail_total_compressed = sum(e["compressed_tokens"] for e in a["per_type"])
    print(f"[C] USD translation (model={args.model})…", file=sys.stderr)
    full_blob = "\n".join(m["content"] for c0 in commits for m in commit_to_messages(c0))
    tail_raw = int(rpc("count_tokens", {"text": full_blob}).get("token_count") or 0)
    tail_compressed = d["rolling_off_tokens"]
    c = usd_savings(tail_raw, tail_compressed, tail_total_raw, tail_total_compressed, args.model)

    summary = {
        "git_range": "(redacted)" if args.anonymise else f"{args.since}..{args.until}",
        "commits": len(commits),
        **a,
        **b,
        **c,
        "rolling_compare": d,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 88)
    print(f"Real-session deep dive — git {args.since}..{args.until}, {len(commits)} commits")
    print("=" * 88)

    print("\n[A] Per-commit-type compression on the diff body")
    print(f"  {'type':<10} {'n':>3} {'raw':>7} {'compressed':>10} {'saved%':>8}")
    print(f"  {'-'*10} {'-'*3} {'-'*7} {'-'*10} {'-'*8}")
    for row in a["per_type"]:
        print(
            f"  {row['type']:<10} {row['n']:>3} {row['raw_tokens']:>7} "
            f"{row['compressed_tokens']:>10} {row['saved_pct']:>7.1f}%"
        )

    print("\n[B] Bombs — top largest commits")
    print(f"  {'sha':<10} {'raw':>7} {'compr':>6} {'saved%':>8}  filter         subject")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*8}  {'-'*14} {'-'*40}")
    for row in b["bombs"]:
        print(
            f"  {row['sha']:<10} {row['raw_tokens']:>7} "
            f"{row['compressed_tokens']:>6} {row['saved_pct']:>7.1f}%  "
            f"{row['filter']:<14} {row['subject']}"
        )

    print(f"\n[C] USD savings (Claude {c['model'].title()} pricing)")
    print(f"  Tail (compress_history) saved tokens:        {c['saved_history_tokens']:>8}")
    print(f"  Diffs (compress_output) saved tokens:        {c['saved_diffs_tokens']:>8}")
    print(
        f"  Naive input savings (single read, "
        f"${c['pricing_input_per_mt_usd']:.2f}/MT):  "
        f"${c['naive_input_usd']:>7.4f}"
    )
    print(
        f"  With prompt caching (25-turn reads, "
        f"${c['pricing_cache_read_per_mt_usd']:.2f}/MT):  "
        f"${c['with_caching_25turn_reads_usd']:>7.4f}"
    )
    print("    Pass --model sonnet or --model haiku to re-price for those tiers.")

    print("\n[D] Rolling summary on/off (heuristic mode, threshold=20)")
    print(f"  rolling OFF  →  {d['rolling_off_tokens']:>5} tokens (msg count {d['rolling_off_msgs']})")
    print(f"  rolling ON   →  {d['rolling_on_tokens']:>5} tokens (msg count {d['rolling_on_msgs']})")
    drift = d["rolling_off_tokens"] - d["rolling_on_tokens"]
    if d["rolling_off_tokens"]:
        drift_pct = drift / d["rolling_off_tokens"] * 100
        print(f"  delta:           {drift:+5} tokens ({drift_pct:+.1f} %)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
