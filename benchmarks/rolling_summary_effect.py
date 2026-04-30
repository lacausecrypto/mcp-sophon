#!/usr/bin/env python3
"""
Rolling-summary effect bench (phase 2B).

Quantifies the win from `SOPHON_ROLLING_SUMMARY=1`: instead of
re-summarising the full history on every `compress_history` call,
the summary is built once at ingest time and stitched to the live
recent window at query time.

What we measure
    no-rolling baseline  : plain `compress_history` per call,
                           no rolling state. Equivalent to v0.5.1
                           behaviour and any caller without the
                           env var set.
    rolling enabled      : `SOPHON_ROLLING_SUMMARY=1` —
                           `update_memory` builds the summary at
                           ingest, `compress_history` serves the
                           cached state.

For each, drive a single long-lived `sophon serve` from turn 1 to N
and at every checkpoint sample the wall-clock of:
    - `update_memory` (ingest cost — should grow when rolling fires)
    - `compress_history` (query cost — should drop when rolling
       active)

The headline numbers users care about are:
    * `compress_history p50 / p99` — does query time stay flat?
    * `update_memory p99` — what spikes did rolling introduce?
    * total time for N queries vs total time without rolling

No API keys, no LLM call. The deterministic heuristic path is
forced via `SOPHON_NO_LLM_SUMMARY=1` so the bench is reproducible
in CI; pass `--with-llm` to also measure the LLM path if your
machine has `claude -p --model haiku` on `$PATH`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)


# ---------------- RPC helpers ----------------
def _init_msg(rid: int = 0) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "rolling-bench", "version": "0"},
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


# ---------------- workload ----------------
def synthetic_turn(index: int) -> list[dict]:
    """Two messages per turn (user + assistant). Topic-shifting
    enough that the heuristic summariser produces non-trivial
    output."""
    topics = [
        "retrieval",
        "compression",
        "caching",
        "chunking",
        "MCP protocol",
        "tokenizer",
        "rust internals",
        "agent loops",
    ]
    t = topics[index % len(topics)]
    user = (
        f"Turn {index}: walk me through how {t} interacts with the rest "
        f"of the pipeline. We were measuring p99 around {(index * 17) % 400 + 60} ms."
    )
    assistant = (
        f"On {t} specifically: the main coupling point is upstream of the "
        f"chunker; instrumenting {t}_index per quantile and correlating with "
        f"{topics[(index + 3) % len(topics)]} queue depth is what surfaces "
        f"the pathological cases. Common culprit at turn {index} is allocator "
        f"contention under sustained load."
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


@dataclass
class RunResult:
    label: str
    update_ms_p50: float
    update_ms_p99: float
    compress_ms_p50: float
    compress_ms_p99: float
    total_compress_ms: float
    total_update_ms: float
    rolling_fired: bool = False
    extra: dict = field(default_factory=dict)


def run_session(
    label: str,
    max_turns: int,
    checkpoint_every: int,
    env_overrides: dict[str, str],
) -> RunResult:
    """Drive one long-lived `sophon serve` end-to-end."""
    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}. Set SOPHON_BIN.", file=sys.stderr)
        sys.exit(2)

    env = os.environ.copy()
    env.update(env_overrides)

    # Each run gets its own JSONL store so they don't share rolling
    # state across runs.
    with tempfile.TemporaryDirectory() as tmpdir:
        env.setdefault("SOPHON_MEMORY_PATH", str(Path(tmpdir) / "memory.jsonl"))

        proc = subprocess.Popen(
            [SOPHON, "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        assert proc.stdin and proc.stdout
        update_samples: list[float] = []
        compress_samples: list[float] = []

        try:
            _send_and_read(proc, _init_msg(0))
            all_msgs: list[dict] = []
            rid = 1
            for turn in range(1, max_turns + 1):
                new = synthetic_turn(turn)
                all_msgs.extend(new)
                _, update_ms = _send_and_read(proc, _update_memory(rid, new))
                rid += 1
                update_samples.append(update_ms)
                if turn % checkpoint_every == 0 or turn == max_turns:
                    _, compress_ms = _send_and_read(
                        proc, _compress_history(rid, all_msgs)
                    )
                    rid += 1
                    compress_samples.append(compress_ms)
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Detect whether rolling fired by checking if the sidecar exists.
        sidecar = Path(env["SOPHON_MEMORY_PATH"] + ".sophon-summary.json")
        rolling_fired = sidecar.exists()

    update_sorted = sorted(update_samples)
    compress_sorted = sorted(compress_samples)
    return RunResult(
        label=label,
        update_ms_p50=statistics.median(update_samples) if update_samples else 0.0,
        update_ms_p99=update_sorted[
            min(len(update_sorted) - 1, int(len(update_sorted) * 0.99))
        ] if update_sorted else 0.0,
        compress_ms_p50=statistics.median(compress_samples) if compress_samples else 0.0,
        compress_ms_p99=compress_sorted[
            min(len(compress_sorted) - 1, int(len(compress_sorted) * 0.99))
        ] if compress_sorted else 0.0,
        total_update_ms=sum(update_samples),
        total_compress_ms=sum(compress_samples),
        rolling_fired=rolling_fired,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run with SOPHON_LLM_CMD if `claude -p --model haiku` is on $PATH.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    runs: list[RunResult] = []

    # Run 1: baseline (no rolling).
    print(f"Run 1/2: baseline (rolling off, {args.max_turns} turns, checkpoint every {args.checkpoint})…", file=sys.stderr)
    runs.append(
        run_session(
            "baseline (no rolling)",
            args.max_turns,
            args.checkpoint,
            env_overrides={"SOPHON_NO_LLM_SUMMARY": "1"},
        )
    )

    # Run 2: rolling on.
    print(f"Run 2/2: rolling on (threshold={args.threshold})…", file=sys.stderr)
    runs.append(
        run_session(
            f"rolling enabled (threshold={args.threshold}, heuristic)",
            args.max_turns,
            args.checkpoint,
            env_overrides={
                "SOPHON_ROLLING_SUMMARY": "1",
                "SOPHON_ROLLING_THRESHOLD": str(args.threshold),
                "SOPHON_NO_LLM_SUMMARY": "1",
            },
        )
    )

    # Optional LLM run.
    if args.with_llm:
        if not os.environ.get("SOPHON_LLM_CMD") and not _which("claude"):
            print("--with-llm requested but `claude` not on $PATH and "
                  "$SOPHON_LLM_CMD not set; skipping LLM run.", file=sys.stderr)
        else:
            print("Run 3/3: rolling on with LLM (slow!)…", file=sys.stderr)
            runs.append(
                run_session(
                    "rolling enabled + LLM",
                    args.max_turns,
                    args.checkpoint,
                    env_overrides={
                        "SOPHON_ROLLING_SUMMARY": "1",
                        "SOPHON_ROLLING_THRESHOLD": str(args.threshold),
                        # Don't set NO_LLM_SUMMARY here.
                    },
                )
            )

    if args.json:
        print(json.dumps(
            [
                {
                    "label": r.label,
                    "update_ms_p50": r.update_ms_p50,
                    "update_ms_p99": r.update_ms_p99,
                    "compress_ms_p50": r.compress_ms_p50,
                    "compress_ms_p99": r.compress_ms_p99,
                    "total_update_ms": r.total_update_ms,
                    "total_compress_ms": r.total_compress_ms,
                    "rolling_fired": r.rolling_fired,
                }
                for r in runs
            ],
            indent=2,
        ))
        return 0

    print()
    print("=" * 92)
    print(f"Rolling summary effect — {args.max_turns} turns, checkpoint every {args.checkpoint}")
    print("=" * 92)
    header = (
        f"  {'run':<42} "
        f"{'compress p50':>13} "
        f"{'compress p99':>13} "
        f"{'update p99':>11} "
        f"{'rolling?':>9}"
    )
    print(header)
    print(f"  {'-' * 42} {'-' * 13} {'-' * 13} {'-' * 11} {'-' * 9}")
    for r in runs:
        print(
            f"  {r.label[:41]:<42} "
            f"{r.compress_ms_p50:>11.2f}ms "
            f"{r.compress_ms_p99:>11.2f}ms "
            f"{r.update_ms_p99:>9.2f}ms "
            f"{'yes' if r.rolling_fired else 'no':>9}"
        )

    if len(runs) >= 2:
        baseline, rolling = runs[0], runs[1]
        print()
        print("Delta (rolling vs baseline):")
        delta_compress_p50 = baseline.compress_ms_p50 - rolling.compress_ms_p50
        delta_compress_p99 = baseline.compress_ms_p99 - rolling.compress_ms_p99
        delta_total = baseline.total_compress_ms - rolling.total_compress_ms
        speedup_p50 = (
            baseline.compress_ms_p50 / rolling.compress_ms_p50
            if rolling.compress_ms_p50 > 0 else float("inf")
        )
        print(f"  compress p50 : {delta_compress_p50:+7.2f} ms  ({speedup_p50:.2f}x speedup)")
        print(f"  compress p99 : {delta_compress_p99:+7.2f} ms")
        print(f"  total compress wall-clock: {delta_total:+8.2f} ms over the run")
        print(f"  update p99 cost paid for rolling: {rolling.update_ms_p99:.2f} ms (vs {baseline.update_ms_p99:.2f} ms baseline)")

    print()
    print("Methodology:")
    print(f"  * Driver: one long-lived `sophon serve` per run (steady-state, no cold-start noise).")
    print(f"  * Workload: {args.max_turns} synthetic turns (user/assistant pairs cycling 8 topics).")
    print(f"  * `compress_history` called every {args.checkpoint} turns with max_tokens=2000.")
    print(f"  * SOPHON_NO_LLM_SUMMARY=1 forces deterministic heuristic summariser.")
    print(f"  * Rolling threshold: {args.threshold} messages (un-summarised tail).")
    return 0


def _which(name: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        c = os.path.join(p, name)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


if __name__ == "__main__":
    sys.exit(main())
