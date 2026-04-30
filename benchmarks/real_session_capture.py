#!/usr/bin/env python3
"""
Real-session capture bench — replay this repo's actual git history
through Sophon's `compress_history` + `compress_output` and report
what the compression would have done on a *real* agent-driven
software-engineering session.

Why this exists
===============

All other benches in this repo are synthetic. Headline numbers
(68 % session compression, 90 % aggregate compress_output, 74 %
tool-call dedup) are produced from canned samples that we wrote
ourselves to exercise specific code paths. They prove the
machinery works; they do not prove the *pitch* — that compression
matters in real life.

This bench takes the easiest available real workload — the git
history of v0.5.0 → v0.5.4 development on this very repo, which
is ~30 commits of an agent loop (Claude / Claude Code on the
operator's machine) doing genuine software engineering: feature
design, refactor, tests, fmt fixes, CI fixes, release chore. The
commits are public; the diffs are public; nothing is fabricated.

What we measure
===============

For each commit in the range, we build a "session turn" that
models what an agent loop produces:

    user      = "Task: <commit subject>"  (the directive)
    assistant = <commit body>             (the reasoning)
    tool_use  = {name:"write_file", input:{path:<file>}}  per file
    tool_result = <diff body, truncated to per-file blob>

Each turn is appended to a synthetic agent session that grows
turn by turn. At every checkpoint we measure:

  raw_tokens          tokens in the entire session
  compressed_tokens   tokens after `compress_history(max_tokens=4000)`
  saved_pct           (raw - compressed) / raw

…plus a per-commit `compress_output` measurement on the diff body
itself (so we get a number for "how much would Sophon save on
the typical write-file tool result a coding agent emits?").

What this is honest about
=========================

* The "user" turn is synthesised from the commit subject — a real
  agent loop would have a richer user prompt. Our synthesis is
  intentionally minimal so we don't inflate the compression
  number.
* The "assistant" turn is the commit body, which is verbose by
  design (this repo's commit style includes mechanics +
  measurements + caveats). That is REPRESENTATIVE of agent
  output that explains itself, which is the typical case for
  Claude Code; it would NOT be representative of a terse agent
  that just emits diffs.
* `tool_result` content is the file diff. Real tool results
  vary — some are file contents, some are command outputs, some
  are error messages. The diff is one realistic shape.

Reading the result
==================

The headline number is `tail_compress_pct` — what compression
would have done if you handed the WHOLE accumulated session to
Sophon at the end. That maps onto "I had a 4-hour Claude Code
session, here's the compression". Per-commit numbers show how
the compression scales with session length.

Running
=======

    python3 benchmarks/real_session_capture.py
    python3 benchmarks/real_session_capture.py --since v0.5.0~1 --json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(ROOT / "sophon/target/release/sophon"),
)


def git_commits(since: str, until: str) -> list[dict]:
    """Walk the commit range and return a list of structured records.
    Uses `\x1e` (record separator) between commits and `\x1f` (unit
    separator) between fields so commit bodies that contain newlines
    are parsed correctly."""
    fmt = "%H%x1f%s%x1f%b%x1f%an%x1f%ai%x1e"
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--reverse", f"--format={fmt}", f"{since}..{until}"],
        text=True,
    )
    commits = []
    for entry in raw.split("\x1e"):
        if not entry.strip():
            continue
        # Strip leading newline that git inserts between records.
        entry = entry.lstrip("\n")
        parts = entry.split("\x1f")
        if len(parts) < 5:
            continue
        sha, subject, body, author, date = parts[:5]
        # Get the diff for this commit, capped to a sane size per file
        # so a single huge JSONL change doesn't dominate.
        diff = subprocess.check_output(
            [
                "git", "-C", str(ROOT),
                "show", "--format=", "--unified=3", sha,
            ],
            text=True,
            errors="replace",
        )
        files = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "--name-only", "--format=", sha],
            text=True,
        ).splitlines()
        commits.append({
            "sha": sha[:8],
            "subject": subject,
            "body": body,
            "author": author,
            "date": date,
            "diff": diff,
            "files": [f for f in files if f.strip()],
        })
    return commits


def rpc_compress_history(messages: list[dict], max_tokens: int = 4000) -> dict:
    """Run `compress_history` over the synthetic session and return
    the structuredContent dict (raw + compressed tokens)."""
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "real-session-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_history",
                "arguments": {"messages": messages, "max_tokens": max_tokens},
            },
        },
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=120)
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


def rpc_compress_output(command: str, output: str) -> dict:
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "real-session-bench", "version": "0"},
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
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=60)
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
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "real-session-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "count_tokens",
                "arguments": {"text": text},
            },
        },
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=30)
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


def commit_to_messages(commit: dict, max_diff_chars: int = 4000) -> list[dict]:
    """Render one commit as 2-4 messages: user task, assistant
    explanation, write_file tool_use(s), tool_result diff."""
    msgs = [
        {"role": "user", "content": f"Task: {commit['subject']}"},
        {
            "role": "assistant",
            "content": (commit["body"] or "(no commit body)")
            + f"\nFiles touched: {', '.join(commit['files'])}",
        },
    ]
    # Model the diff as one tool_use + tool_result pair (the agent
    # ran a write_file or apply_diff). Cap the diff body so a single
    # mega-commit doesn't blow the per-turn budget.
    diff = commit["diff"]
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + f"\n... (diff truncated, original {len(commit['diff'])} chars)"
    if commit["files"]:
        tool_use = {
            "type": "tool_use",
            "name": "apply_diff",
            "input": {"files": commit["files"][:5]},
        }
        tool_result = {
            "type": "tool_result",
            "tool_use_id": commit["sha"],
            "content": diff,
        }
        msgs.append({"role": "assistant", "content": json.dumps(tool_use)})
        msgs.append({"role": "user", "content": json.dumps(tool_result)})
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default="b197760~1",
        help="Lower bound (exclusive) of git range. Default: just before "
        "the v0.5.0 work started in this repo.",
    )
    parser.add_argument(
        "--until",
        default="HEAD",
        help="Upper bound (inclusive) of git range. Default: HEAD.",
    )
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}", file=sys.stderr)
        return 2

    print(f"Walking git log {args.since}..{args.until}…", file=sys.stderr)
    commits = git_commits(args.since, args.until)
    print(f"  → {len(commits)} commits", file=sys.stderr)
    if not commits:
        print("no commits in range — try a wider --since", file=sys.stderr)
        return 1

    # Per-commit compress_output measurement on the diff body itself.
    per_diff = []
    print(f"Measuring compress_output on each commit's diff ({len(commits)} calls)…", file=sys.stderr)
    for i, c in enumerate(commits):
        diff = c["diff"]
        if len(diff) > args.max_diff_chars * 4:
            diff = diff[: args.max_diff_chars * 4]
        if not diff.strip():
            continue
        result = rpc_compress_output("git diff", diff)
        per_diff.append({
            "sha": c["sha"],
            "subject": c["subject"][:60],
            "raw_tokens": int(result.get("original_tokens") or 0),
            "compressed_tokens": int(result.get("compressed_tokens") or 0),
            "ratio": float(result.get("ratio") or 1.0),
            "filter": result.get("filter_name", "?"),
        })
        if (i + 1) % 5 == 0:
            print(f"  diff {i+1}/{len(commits)} done", file=sys.stderr)

    # Build the running session message-by-message + checkpoint
    # compress_history at every Nth commit.
    print(f"Replaying session through compress_history checkpoints (every {args.checkpoint} commits)…", file=sys.stderr)
    accumulated: list[dict] = []
    checkpoints = []
    for i, c in enumerate(commits):
        accumulated.extend(commit_to_messages(c, args.max_diff_chars))
        if (i + 1) % args.checkpoint == 0 or i == len(commits) - 1:
            raw_blob = "\n".join(m["content"] for m in accumulated)
            raw_tokens = rpc_count_tokens(raw_blob)
            ch = rpc_compress_history(accumulated, max_tokens=args.max_tokens)
            compressed_tokens = int(ch.get("token_count") or 0)
            saved_pct = (raw_tokens - compressed_tokens) / raw_tokens * 100 if raw_tokens else 0.0
            checkpoints.append({
                "after_commit": i + 1,
                "session_messages": len(accumulated),
                "raw_tokens": raw_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_pct": saved_pct,
            })
            print(
                f"  commit {i+1:>2}/{len(commits)}: "
                f"messages={len(accumulated):>3}  "
                f"raw={raw_tokens:>6}  compressed={compressed_tokens:>5}  "
                f"saved={saved_pct:>5.1f}%",
                file=sys.stderr,
            )

    # Aggregates
    diff_raw = sum(d["raw_tokens"] for d in per_diff)
    diff_compressed = sum(d["compressed_tokens"] for d in per_diff)
    diff_saved_pct = (diff_raw - diff_compressed) / diff_raw * 100 if diff_raw else 0.0

    last_ck = checkpoints[-1] if checkpoints else {}
    summary = {
        "git_range": f"{args.since}..{args.until}",
        "commits": len(commits),
        "session_total_messages": len(accumulated),
        "tail_raw_tokens": last_ck.get("raw_tokens", 0),
        "tail_compressed_tokens": last_ck.get("compressed_tokens", 0),
        "tail_compress_pct": last_ck.get("saved_pct", 0.0),
        "per_diff_raw_tokens": diff_raw,
        "per_diff_compressed_tokens": diff_compressed,
        "per_diff_aggregate_saved_pct": diff_saved_pct,
        "checkpoints": checkpoints,
        "per_diff": per_diff,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 84)
    print(f"Real-session capture — git {args.since}..{args.until}")
    print("=" * 84)
    print(f"  Commits replayed:                {summary['commits']}")
    print(f"  Total session messages:          {summary['session_total_messages']}")
    print()
    print(f"  Tail compress_history (whole session):")
    print(f"    raw tokens:                    {summary['tail_raw_tokens']:>8}")
    print(f"    compressed (max=4000):         {summary['tail_compressed_tokens']:>8}")
    print(f"    saved:                         {summary['tail_compress_pct']:>7.1f} %")
    print()
    print(f"  Per-diff compress_output aggregate:")
    print(f"    raw tokens:                    {summary['per_diff_raw_tokens']:>8}")
    print(f"    compressed:                    {summary['per_diff_compressed_tokens']:>8}")
    print(f"    saved:                         {summary['per_diff_aggregate_saved_pct']:>7.1f} %")
    print()
    print("  Compression growth as session lengthens:")
    print(f"    {'after #':>8} {'msgs':>6} {'raw':>7} {'compr':>6} {'saved%':>8}")
    for ck in checkpoints:
        print(
            f"    {ck['after_commit']:>8} "
            f"{ck['session_messages']:>6} "
            f"{ck['raw_tokens']:>7} "
            f"{ck['compressed_tokens']:>6} "
            f"{ck['saved_pct']:>7.1f}%"
        )
    print()
    print("Honest scope:")
    print("  * 'user' turns synthesised from commit subjects (terse).")
    print("  * 'assistant' turns are commit bodies (verbose by this repo's style).")
    print("  * tool_result content is the actual git diff (real activity).")
    print("  * Numbers reflect THIS repo's commit cadence + verbosity; your")
    print("    own session shape may differ. The bench script is the spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
