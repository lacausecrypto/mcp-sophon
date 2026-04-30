#!/usr/bin/env python3
"""
Run the full bench suite, compare to a stored baseline, fail loudly
on regressions.

Used by `.github/workflows/bench-nightly.yml` to catch silent
performance drift in pull requests. The committed baseline lives at
`benchmarks/baseline.json` and is updated explicitly by maintainers
via `--update-baseline` after a deliberate change.

What this driver does
=====================

1. Calls each `--json` bench in `BENCHES` and pulls a flat dict of
   key metrics from each.
2. Compares against the baseline. Each metric has a direction
   (`min` = lower is better; `max` = higher is better) and a
   tolerance percentage.
3. Exits non-zero if any metric drifted past tolerance in the
   wrong direction. Prints a digest of every comparison so the
   CI log surfaces both regressions AND wins.

Wins: the script also flags improvements >5 % so maintainers
notice when the baseline should be refreshed.

Caveats
=======

* Cold-start metrics carry real wall-clock variance (OS file cache,
  scheduler jitter). The default tolerance for those is generous
  (15 %); compression-ratio metrics get a tight tolerance (2 %)
  because they're deterministic per binary.
* The bench scripts must support `--json` for machine-readable
  output. Any new bench added here without `--json` will be
  flagged as missing rather than silently passing.

Running locally
===============

    python3 benchmarks/run_all_with_baseline.py
    python3 benchmarks/run_all_with_baseline.py --update-baseline
    python3 benchmarks/run_all_with_baseline.py --quick   # skip slow ones
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "benchmarks" / "baseline.json"


@dataclass
class Metric:
    """One numeric metric extracted from a bench, with how to
    interpret it for regression detection."""
    name: str
    extractor: str  # JSON-path-ish: dot for keys, [N] for arrays, [-1] for last
    direction: str  # "max" (higher = better) or "min" (lower = better)
    tolerance_pct: float  # max allowed drift in the wrong direction
    bench: str  # which bench produced this metric


def extract(data, path: str):
    """Tiny JSON path: `key.subkey[0].field`. Supports `[N]` and
    `[-1]` for array indexing. Returns None on any missing step."""
    cur = data
    # Tokenise — split on `.` then handle `[N]` suffixes.
    for raw_step in path.split("."):
        # Pull bracket suffixes off the bare key.
        idx_parts = []
        key = raw_step
        while "[" in key and key.endswith("]"):
            bracket = key.rfind("[")
            idx_parts.insert(0, key[bracket + 1 : -1])
            key = key[:bracket]
        if key:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
        for idx in idx_parts:
            try:
                cur = cur[int(idx)]
            except (IndexError, KeyError, ValueError, TypeError):
                return None
        if cur is None:
            return None
    return cur


# ---------------------------------------------------------------------------
# Bench registry — what to run, what to extract, how to judge
# ---------------------------------------------------------------------------
BENCHES = {
    "cold_start_and_footprint": {
        "cmd": ["python3", str(ROOT / "benchmarks/cold_start_and_footprint.py"), "--json"],
        "metrics": [
            Metric("binary_size_bytes", "[0].binary_size_bytes", "min", 5.0, "cold_start"),
            Metric("cold_start_ms_p50", "[0].cold_start_ms_p50", "min", 25.0, "cold_start"),
            Metric("cold_start_ms_p99", "[0].cold_start_ms_p99", "min", 30.0, "cold_start"),
            Metric("rss_idle_mb", "[0].rss_idle_mb", "min", 25.0, "cold_start"),
        ],
        "slow": False,
    },
    "compress_output_per_command": {
        "cmd": ["python3", str(ROOT / "benchmarks/compress_output_per_command.py"), "--json"],
        # The bench prints a list of records. We compute the
        # weighted aggregate post-hoc in score_run().
        "metrics": [
            Metric("aggregate_saved_pct", "_synthetic.aggregate_saved", "max", 2.0, "per_command"),
            Metric("mean_saved_pct", "_synthetic.mean_saved", "max", 5.0, "per_command"),
        ],
        "slow": False,
    },
    "session_scaling_curve": {
        "cmd": [
            "python3",
            str(ROOT / "benchmarks/session_scaling_curve.py"),
            "--max-turns", "100",
            "--checkpoint", "25",
            "--json",
        ],
        "metrics": [
            Metric("compress_history_p50_ms", "compress_history_ms_p50", "min", 30.0, "scaling"),
            Metric("compress_history_p99_ms", "compress_history_ms_p99", "min", 50.0, "scaling"),
            Metric("update_memory_p99_ms", "update_memory_ms_p99", "min", 50.0, "scaling"),
        ],
        "env": {"SOPHON_MEMORY_PATH": ""},  # filled per-run
        "slow": True,
    },
}


def run_bench(name: str, cfg: dict) -> dict:
    """Execute a bench with --json and return its parsed output,
    or `{"_error": "..."}` if it failed."""
    env = os.environ.copy()
    if "env" in cfg:
        for k, v in cfg["env"].items():
            if k == "SOPHON_MEMORY_PATH" and not v:
                # Per-run scratch file so scaling bench doesn't share
                # state across CI invocations.
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".jsonl", delete=False, dir=tempfile.gettempdir()
                )
                tmp.close()
                env[k] = tmp.name
            else:
                env[k] = v
    print(f"  → running {name}", file=sys.stderr)
    try:
        result = subprocess.run(
            cfg["cmd"],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"_error": "timeout (>600s)"}
    if result.returncode != 0:
        return {
            "_error": f"exit {result.returncode}",
            "_stderr_tail": result.stderr[-500:],
        }
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"_error": f"non-JSON stdout: {e}", "_stdout_head": result.stdout[:200]}

    # Bench-specific post-processing: for the per-command bench we
    # compute the weighted aggregate ourselves rather than asking
    # the bench to expose another field.
    if name == "compress_output_per_command":
        agg_raw = sum(r.get("raw_tokens", 0) for r in data)
        agg_compressed = sum(r.get("compressed_tokens", 0) for r in data)
        weighted = (
            (agg_raw - agg_compressed) / agg_raw if agg_raw else 0.0
        )
        per_cmd_savings = [
            (r["raw_tokens"] - r["compressed_tokens"]) / r["raw_tokens"]
            for r in data
            if r.get("raw_tokens", 0) > 0
        ]
        mean_saved = sum(per_cmd_savings) / len(per_cmd_savings) if per_cmd_savings else 0.0
        data = {
            "_records": data,
            "_synthetic": {
                "aggregate_saved": weighted * 100,
                "mean_saved": mean_saved * 100,
            },
        }
    return data


def score_run(
    bench_outputs: dict[str, dict],
    baseline: dict,
) -> tuple[list[str], list[str], list[str], dict]:
    """Compare bench outputs to baseline. Return (regressions, wins,
    notes, fresh_baseline)."""
    regressions: list[str] = []
    wins: list[str] = []
    notes: list[str] = []
    fresh: dict = {}

    for bench_name, cfg in BENCHES.items():
        data = bench_outputs.get(bench_name, {})
        if "_error" in data:
            regressions.append(
                f"  [{bench_name}] BENCH FAILED: {data['_error']}"
            )
            continue
        for m in cfg["metrics"]:
            current = extract(data, m.extractor)
            if current is None:
                regressions.append(
                    f"  [{bench_name}.{m.name}] EXTRACTION FAILED at path "
                    f"{m.extractor!r}; bench output schema may have drifted."
                )
                continue
            try:
                current = float(current)
            except (TypeError, ValueError):
                regressions.append(
                    f"  [{bench_name}.{m.name}] non-numeric value: {current!r}"
                )
                continue
            fresh.setdefault(bench_name, {})[m.name] = current
            base_v = baseline.get(bench_name, {}).get(m.name)
            if base_v is None:
                notes.append(
                    f"  [{bench_name}.{m.name}] new metric, no baseline (recording {current:.3f})"
                )
                continue
            base_v = float(base_v)
            # Compute drift in the BAD direction.
            if base_v == 0:
                # Avoid div/0 — treat any change as 100 %.
                pct_drift = float("inf") if current != 0 else 0.0
            elif m.direction == "min":  # lower is better; positive drift = regression
                pct_drift = (current - base_v) / abs(base_v) * 100
            else:  # "max" — higher is better; negative drift = regression
                pct_drift = (base_v - current) / abs(base_v) * 100
            if pct_drift > m.tolerance_pct:
                regressions.append(
                    f"  [{bench_name}.{m.name}] REGRESSION {pct_drift:+.1f}% "
                    f"(was {base_v:.3f}, now {current:.3f}, tolerance {m.tolerance_pct}%)"
                )
            elif pct_drift < -5.0:
                # Improvement worth flagging
                wins.append(
                    f"  [{bench_name}.{m.name}] IMPROVED {-pct_drift:.1f}% "
                    f"(was {base_v:.3f}, now {current:.3f})"
                )
    return regressions, wins, notes, fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Replace `benchmarks/baseline.json` with the current run results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip benches marked `slow: True`.",
    )
    parser.add_argument("--json", action="store_true", help="Emit comparison as JSON.")
    args = parser.parse_args()

    print("Running bench suite for baseline comparison…", file=sys.stderr)
    bench_outputs: dict[str, dict] = {}
    for name, cfg in BENCHES.items():
        if args.quick and cfg.get("slow"):
            print(f"  → skipping {name} (--quick)", file=sys.stderr)
            continue
        bench_outputs[name] = run_bench(name, cfg)

    baseline = {}
    if BASELINE.exists():
        with open(BASELINE) as f:
            baseline = json.load(f)

    regressions, wins, notes, fresh = score_run(bench_outputs, baseline)

    if args.update_baseline:
        # Merge fresh values into existing baseline (don't drop benches
        # that weren't run via --quick).
        merged = dict(baseline)
        for k, v in fresh.items():
            merged[k] = {**merged.get(k, {}), **v}
        BASELINE.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nBaseline updated → {BASELINE}", file=sys.stderr)

    if args.json:
        print(json.dumps({
            "regressions": regressions,
            "wins": wins,
            "notes": notes,
            "current": fresh,
            "baseline": baseline,
        }, indent=2))
        return 1 if regressions else 0

    print()
    print("=" * 80)
    print("Benchmark vs baseline summary")
    print("=" * 80)
    if regressions:
        print(f"\n❌ {len(regressions)} regression(s):")
        for line in regressions:
            print(line)
    if wins:
        print(f"\n✓ {len(wins)} improvement(s) (worth refreshing the baseline):")
        for line in wins:
            print(line)
    if notes:
        print(f"\n  {len(notes)} note(s):")
        for line in notes:
            print(line)
    if not regressions and not wins and not notes:
        print("\n  All metrics within tolerance.")
    print()

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
