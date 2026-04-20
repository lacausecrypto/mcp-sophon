#!/usr/bin/env python3
"""
Cold start + memory footprint — "single binary" differentiator bench.

Sophon's positioning rests on a promise neither mem0 nor Letta can
match: a **single Rust binary** that starts fast and stays lean.
This bench quantifies that promise against the two obvious
alternatives — a Python process importing mem0 / importing
sentence-transformers — so the README number isn't just a
marketing claim.

What we measure
    binary_size_bytes    disk footprint of the release binary
    cold_start_ms        time from process spawn to `initialize`
                         JSON-RPC response on stdio
    rss_idle_mb          resident-set size right after initialize
    rss_after_n_mb       RSS after running 100 `count_tokens` calls
                         — catches leaks / GC tails

Everything is wall-clock + `ps -o rss`. No API keys, no network.
Python baselines are opt-in via `--include-python-baseline` and
only run if the target modules import cleanly on this machine.

Baseline candidates
    * `python -c "import mem0"`    cold import of mem0ai
    * `python -c "import sentence_transformers; sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')"`
      cold load of a typical retrieval model
    * `python -c "import langchain"`  heavyweight agent stack

If any import fails we skip that row with a `(not installed)`
marker rather than crashing.

Running
    python cold_start_and_footprint.py                  # sophon only
    python cold_start_and_footprint.py --include-python-baseline
    python cold_start_and_footprint.py --json           # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)

REPEATS = 5  # cold-start is noisy; we p50 over N runs


@dataclass
class ColdStartResult:
    label: str
    cold_start_ms_p50: float
    cold_start_ms_p99: float
    binary_size_bytes: int | None
    rss_idle_mb: float | None
    rss_after_100_mb: float | None
    notes: list[str] = field(default_factory=list)


def _rss_of_pid(pid: int) -> float | None:
    """Return RSS in MB for `pid`, or None if it no longer exists."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        kb = int(out.split()[0]) if out else 0
        return kb / 1024.0
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _init_rpc() -> str:
    msg = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "cold-start-bench", "version": "0"},
        },
    }
    return json.dumps(msg) + "\n"


def _count_tokens_rpc(rid: int, text: str) -> str:
    msg = {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": "count_tokens", "arguments": {"text": text}},
    }
    return json.dumps(msg) + "\n"


def measure_sophon_cold_start() -> tuple[float, float | None]:
    """Spawn `sophon serve`, time the first `initialize` response,
    and capture RSS before shutting it down. Returns (ms, rss_mb)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [SOPHON, "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(_init_rpc())
    proc.stdin.flush()
    # Block on the first response line — that's our "ready" signal.
    line = proc.stdout.readline()
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        json.loads(line)
    except json.JSONDecodeError:
        ms = float("nan")
    rss = _rss_of_pid(proc.pid)
    proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return ms, rss


def measure_sophon_steady_rss(n_calls: int = 100) -> float | None:
    """Keep a `sophon serve` process alive, drive N `count_tokens`
    calls through it, and snapshot RSS at the end. Shuts down clean."""
    proc = subprocess.Popen(
        [SOPHON, "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    try:
        proc.stdin.write(_init_rpc())
        proc.stdin.flush()
        proc.stdout.readline()  # initialize response

        sample = "the quick brown fox jumps over the lazy dog." * 5
        for i in range(n_calls):
            proc.stdin.write(_count_tokens_rpc(i + 1, sample))
            proc.stdin.flush()
            proc.stdout.readline()

        rss = _rss_of_pid(proc.pid)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return rss


def measure_python_import(label: str, module_code: str) -> ColdStartResult:
    """Spawn `python -c <code>`, time until process exits. This isn't
    a fair "first-query" comparison because Python services still need
    a server loop, but it IS a fair "what does it cost just to load
    this stack into memory" number — which is what a Sophon-free
    user would have to pay per process restart."""
    notes: list[str] = []
    ms_samples: list[float] = []
    rss_samples: list[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    module_code
                    + "\n"
                    + "import os,sys; sys.stdout.write('ready\\n'); sys.stdout.flush(); "
                    + "sys.stdin.readline()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return ColdStartResult(
                label=label,
                cold_start_ms_p50=float("nan"),
                cold_start_ms_p99=float("nan"),
                binary_size_bytes=None,
                rss_idle_mb=None,
                rss_after_100_mb=None,
                notes=["python binary not found"],
            )
        assert proc.stdin and proc.stdout
        ready = proc.stdout.readline()
        ms = (time.perf_counter() - t0) * 1000.0
        if "ready" not in ready:
            # import error — read stderr for diagnosis
            stderr_tail = ""
            if proc.stderr:
                stderr_tail = proc.stderr.read()[:200]
            proc.kill()
            proc.wait()
            return ColdStartResult(
                label=label,
                cold_start_ms_p50=float("nan"),
                cold_start_ms_p99=float("nan"),
                binary_size_bytes=None,
                rss_idle_mb=None,
                rss_after_100_mb=None,
                notes=[f"(not installed) {stderr_tail.strip().splitlines()[0] if stderr_tail else ''}"],
            )
        rss = _rss_of_pid(proc.pid)
        ms_samples.append(ms)
        if rss is not None:
            rss_samples.append(rss)
        proc.stdin.write("q\n")
        proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    ms_sorted = sorted(ms_samples)
    return ColdStartResult(
        label=label,
        cold_start_ms_p50=ms_sorted[len(ms_sorted) // 2],
        cold_start_ms_p99=ms_sorted[min(len(ms_sorted) - 1, int(len(ms_sorted) * 0.99))],
        binary_size_bytes=None,
        rss_idle_mb=statistics.median(rss_samples) if rss_samples else None,
        rss_after_100_mb=None,
        notes=notes,
    )


def measure_sophon() -> ColdStartResult:
    if not Path(SOPHON).exists():
        return ColdStartResult(
            label="sophon (release)",
            cold_start_ms_p50=float("nan"),
            cold_start_ms_p99=float("nan"),
            binary_size_bytes=None,
            rss_idle_mb=None,
            rss_after_100_mb=None,
            notes=[f"binary not found at {SOPHON!r} — set SOPHON_BIN"],
        )

    binary_size = Path(SOPHON).stat().st_size

    ms_samples: list[float] = []
    rss_idle_samples: list[float] = []
    for _ in range(REPEATS):
        ms, rss = measure_sophon_cold_start()
        ms_samples.append(ms)
        if rss is not None:
            rss_idle_samples.append(rss)
    ms_sorted = sorted(ms_samples)

    rss_after = measure_sophon_steady_rss(n_calls=100)

    return ColdStartResult(
        label="sophon (release)",
        cold_start_ms_p50=ms_sorted[len(ms_sorted) // 2],
        cold_start_ms_p99=ms_sorted[min(len(ms_sorted) - 1, int(len(ms_sorted) * 0.99))],
        binary_size_bytes=binary_size,
        rss_idle_mb=statistics.median(rss_idle_samples) if rss_idle_samples else None,
        rss_after_100_mb=rss_after,
    )


def format_size(bytes_: int | None) -> str:
    if bytes_ is None:
        return "—"
    mb = bytes_ / 1_000_000
    return f"{mb:5.1f} MB"


def format_ms(v: float) -> str:
    if v != v:  # NaN
        return "   —"
    return f"{v:6.1f}"


def format_rss(v: float | None) -> str:
    if v is None:
        return "    —"
    return f"{v:6.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-python-baseline",
        action="store_true",
        help="Also measure cold-start of Python mem0 / sentence-transformers / "
        "langchain for comparison. Skipped rows for modules that fail to import.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args()

    results: list[ColdStartResult] = [measure_sophon()]

    if args.include_python_baseline:
        results.append(measure_python_import("python + mem0", "import mem0"))
        results.append(
            measure_python_import(
                "python + sentence-transformers (MiniLM-L6-v2 load)",
                "from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer('all-MiniLM-L6-v2')",
            )
        )
        results.append(measure_python_import("python + langchain", "import langchain"))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "label": r.label,
                        "cold_start_ms_p50": r.cold_start_ms_p50,
                        "cold_start_ms_p99": r.cold_start_ms_p99,
                        "binary_size_bytes": r.binary_size_bytes,
                        "rss_idle_mb": r.rss_idle_mb,
                        "rss_after_100_mb": r.rss_after_100_mb,
                        "notes": r.notes,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return 0

    print("=" * 94)
    print("Cold start + memory footprint bench")
    print("=" * 94)
    print(
        f"  {'stack':<52} "
        f"{'size':>8} "
        f"{'cold p50':>9} "
        f"{'cold p99':>9} "
        f"{'RSS idle':>10}"
    )
    print(f"  {'-' * 52} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 10}")
    for r in results:
        print(
            f"  {r.label:<52} "
            f"{format_size(r.binary_size_bytes):>8} "
            f"{format_ms(r.cold_start_ms_p50):>9}ms "
            f"{format_ms(r.cold_start_ms_p99):>8}ms "
            f"{format_rss(r.rss_idle_mb):>10}"
        )
        for note in r.notes:
            print(f"    └─ {note}")

    # Sophon-specific: RSS after 100 calls
    sophon = next((r for r in results if "sophon" in r.label), None)
    if sophon and sophon.rss_after_100_mb is not None:
        print()
        print(
            f"  sophon RSS after 100 count_tokens calls: "
            f"{sophon.rss_after_100_mb:.1f} MB "
            f"(delta: {(sophon.rss_after_100_mb - (sophon.rss_idle_mb or 0)):+.1f} MB)"
        )
    print()
    print("Methodology:")
    print(f"  * REPEATS = {REPEATS}; p50 and p99 over the run set.")
    print("  * Cold-start = process spawn → first JSON-RPC initialize reply on stdout.")
    print("  * RSS via `ps -o rss=` against the running pid (MB = kB ÷ 1024).")
    print("  * Python import baseline is opt-in via --include-python-baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
