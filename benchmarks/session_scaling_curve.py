#!/usr/bin/env python3
"""
Session scaling curve — how Sophon behaves as a session grows.

The v0.4.0 bench suite measured `compress_history` / `update_memory`
at a fixed session length. That's a single point on a curve — it
doesn't tell you how Sophon degrades (or doesn't) as the session
grows from 0 to 1000+ turns. This bench produces the curve.

For each checkpoint we measure:

    update_memory_ms        latency to append the newest turn to
                            the JSONL store + regen stats
    compress_history_ms     latency of a full `compress_history` call
                            with `max_tokens=2000`
    token_count             size of the compressed payload returned
    rss_mb                  resident-set size of the `sophon serve`
                            process (`ps -o rss=`)

Everything is driven through **one** long-lived `sophon serve`
process over stdio so we measure steady-state behaviour, not
cold-start overhead. No LLM calls, no API keys.

Inputs
    --max-turns   how far to push the session (default: 600)
    --checkpoint  sample cadence (default: every 25 turns)
    --csv         write per-checkpoint rows to a file (default: stdout)
    --json        machine-readable output at the end

What a healthy curve looks like
    * `update_memory_ms` stays ~flat (append-only JSONL).
    * `compress_history_ms` is ~flat *until* the threshold, then
      scales with `max_tokens` not `session_length` (Sophon caps
      the payload regardless of turn count).
    * `token_count` saturates near `max_tokens`.
    * `rss_mb` grows linearly in the JSONL store but bounded
      (index is in-memory O(n_chunks * dim)).

A regression that makes `compress_history_ms` O(turns²) or lets
`rss_mb` blow up would immediately show as a superlinear curve.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)


@dataclass
class Checkpoint:
    turn: int
    update_memory_ms: float
    compress_history_ms: float
    token_count: int
    rss_mb: float | None
    history_len: int


def _rss_of_pid(pid: int) -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return int(out.split()[0]) / 1024.0 if out else None
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _init_msg(rid: int = 0) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "scaling-bench", "version": "0"},
        },
    }


def _update_memory(rid: int, msgs: list[dict]) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {
            "name": "update_memory",
            "arguments": {"messages": msgs, "return_snapshot": False},
        },
    }


def _compress_history(rid: int, msgs: list[dict]) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {
            "name": "compress_history",
            "arguments": {
                "messages": msgs,
                "max_tokens": 2000,
                "recent_window": 5,
            },
        },
    }


def _send_and_read(proc: subprocess.Popen, msg: dict) -> tuple[dict, float]:
    assert proc.stdin and proc.stdout
    t0 = time.perf_counter()
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    try:
        resp = json.loads(line)
    except json.JSONDecodeError:
        resp = {}
    return resp, dt_ms


def synthetic_turn(index: int) -> list[dict]:
    """Two messages per turn: user + assistant. Deterministic-seeded
    so re-runs produce bit-identical inputs."""
    topics = ["retrieval", "compression", "caching", "chunking", "mcp", "agents", "rust", "tokenizer"]
    t = topics[index % len(topics)]
    user = (
        f"Turn {index}: diagnose {t} behaviour under load. "
        f"Earlier we saw p99 spikes around {100 + (index * 7) % 400} ms."
    )
    assistant = (
        f"Looking at the {t} path, the bottleneck is upstream of the chunker. "
        f"We should instrument the {t}_index to capture per-{t} latency quantiles "
        f"and correlate with {topics[(index + 3) % len(topics)]} queue depth."
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def run_curve(max_turns: int, checkpoint_every: int) -> list[Checkpoint]:
    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}. Set SOPHON_BIN.", file=sys.stderr)
        sys.exit(2)

    proc = subprocess.Popen(
        [SOPHON, "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    try:
        # initialise
        _send_and_read(proc, _init_msg(0))

        checkpoints: list[Checkpoint] = []
        all_msgs: list[dict] = []
        rid = 1
        for turn in range(1, max_turns + 1):
            new_msgs = synthetic_turn(turn)
            all_msgs.extend(new_msgs)

            # update_memory latency is measured every turn (cheap).
            _, update_ms = _send_and_read(proc, _update_memory(rid, new_msgs))
            rid += 1

            if turn % checkpoint_every == 0 or turn == max_turns or turn == 1:
                resp, compress_ms = _send_and_read(
                    proc, _compress_history(rid, all_msgs)
                )
                rid += 1
                body = resp.get("result", {}).get("structuredContent", {}) or {}
                token_count = int(body.get("token_count") or 0)
                history_len = int(body.get("original_message_count") or len(all_msgs))
                rss = _rss_of_pid(proc.pid)
                checkpoints.append(
                    Checkpoint(
                        turn=turn,
                        update_memory_ms=update_ms,
                        compress_history_ms=compress_ms,
                        token_count=token_count,
                        rss_mb=rss,
                        history_len=history_len,
                    )
                )
                print(
                    f"  turn {turn:>4}  "
                    f"update={update_ms:>6.1f} ms  "
                    f"compress={compress_ms:>7.1f} ms  "
                    f"tokens={token_count:>5}  "
                    f"rss={rss or float('nan'):>6.1f} MB",
                    file=sys.stderr,
                )
        return checkpoints
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-turns", type=int, default=600)
    parser.add_argument("--checkpoint", type=int, default=25)
    parser.add_argument(
        "--csv", type=str, default=None, help="Write per-checkpoint rows to file."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    args = parser.parse_args()

    print(f"Running scaling curve: 1 → {args.max_turns} turns, checkpoint every {args.checkpoint}…", file=sys.stderr)
    cps = run_curve(args.max_turns, args.checkpoint)

    if not cps:
        print("no checkpoints — max-turns must be ≥ checkpoint cadence", file=sys.stderr)
        return 2

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(cps[0]).keys()))
            writer.writeheader()
            for c in cps:
                writer.writerow(asdict(c))
        print(f"Wrote {len(cps)} checkpoints to {args.csv}", file=sys.stderr)

    # Growth ratios = last vs first checkpoint
    first, last = cps[0], cps[-1]
    growth = {
        "turns": last.turn / first.turn,
        "update_ms": last.update_memory_ms / first.update_memory_ms
        if first.update_memory_ms
        else float("inf"),
        "compress_ms": last.compress_history_ms / first.compress_history_ms
        if first.compress_history_ms
        else float("inf"),
        "token_count": last.token_count / max(1, first.token_count),
        "rss_mb": (last.rss_mb or 0) / (first.rss_mb or 1) if first.rss_mb else 1.0,
    }

    compress_samples = [c.compress_history_ms for c in cps]
    update_samples = [c.update_memory_ms for c in cps]

    summary = {
        "max_turns": args.max_turns,
        "checkpoints": len(cps),
        "first_checkpoint": asdict(first),
        "last_checkpoint": asdict(last),
        "growth_ratios": growth,
        "compress_history_ms_p50": statistics.median(compress_samples),
        "compress_history_ms_p99": sorted(compress_samples)[
            min(len(compress_samples) - 1, int(len(compress_samples) * 0.99))
        ],
        "update_memory_ms_p50": statistics.median(update_samples),
        "update_memory_ms_p99": sorted(update_samples)[
            min(len(update_samples) - 1, int(len(update_samples) * 0.99))
        ],
        "all_checkpoints": [asdict(c) for c in cps],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 84)
    print(
        f"Session scaling curve — {args.max_turns} turns, "
        f"{len(cps)} checkpoints"
    )
    print("=" * 84)
    print(f"  {'turn':>5} {'update ms':>10} {'compress ms':>12} {'tokens':>7} {'RSS MB':>8}")
    print(f"  {'-' * 5} {'-' * 10} {'-' * 12} {'-' * 7} {'-' * 8}")
    for c in cps:
        print(
            f"  {c.turn:>5} "
            f"{c.update_memory_ms:>10.1f} "
            f"{c.compress_history_ms:>12.1f} "
            f"{c.token_count:>7} "
            f"{(c.rss_mb or 0):>8.1f}"
        )

    print()
    print("Growth ratios (last / first):")
    print(f"  turns                  {growth['turns']:>7.1f}x")
    print(f"  update_memory latency  {growth['update_ms']:>7.2f}x  (flat → healthy)")
    print(f"  compress_history ms    {growth['compress_ms']:>7.2f}x  (bounded by budget)")
    print(f"  token_count            {growth['token_count']:>7.2f}x  (saturates at max_tokens)")
    print(f"  RSS                    {growth['rss_mb']:>7.2f}x  (linear-ish in store)")

    print()
    print("Latency distribution over the run:")
    print(
        f"  compress_history  p50={summary['compress_history_ms_p50']:>7.1f} ms   "
        f"p99={summary['compress_history_ms_p99']:>7.1f} ms"
    )
    print(
        f"  update_memory     p50={summary['update_memory_ms_p50']:>7.1f} ms   "
        f"p99={summary['update_memory_ms_p99']:>7.1f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
