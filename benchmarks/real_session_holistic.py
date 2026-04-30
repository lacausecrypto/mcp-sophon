#!/usr/bin/env python3
"""
Holistic real-session bench — runs all four real-data benches and
synthesises a single weighted picture of what Sophon does to a
real Claude Code agent session.

The four sub-benches were built incrementally to plug methodological
gaps:

  real_session_capture.py   — compress_history over git log replay
  real_session_shell.py     — compress_output on real shell stdout
  real_session_filereads.py — compress_prompt on real source files
  real_session_search.py    — compress_output on real grep/find outputs

Each measures one tool-traffic dimension. Run separately they
return four different aggregate numbers that aren't directly
comparable — different APIs, different inputs, different shapes.

This driver runs all four as subprocesses (with `--json`),
parses their JSON, and produces:

  1. A single table with every dimension side-by-side.
  2. A "tool-traffic-weighted" estimate that blends the four
     numbers by a typical-agent-session weighting (configurable
     via `--weights`). The default mix reflects this repo's own
     observed shape; users can override for their workload.
  3. A `--anonymise` switch that propagates to all sub-benches.

This script is read-only. Sub-benches all run sequentially because
the cargo + sophon subprocesses don't parallelise cleanly under
the macOS file-descriptor limit.

Running
=======

    python3 benchmarks/real_session_holistic.py
    python3 benchmarks/real_session_holistic.py --skip-cargo
    python3 benchmarks/real_session_holistic.py --json
    python3 benchmarks/real_session_holistic.py \\
        --weights "history=0.35,shell=0.30,filereads=0.20,search=0.15"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default tool-traffic weights based on observed shape of this
# repo's own dev sessions. Sums to 1.0. Override via --weights.
#
# These are FRACTIONS of total session token volume that flow
# through each shape, NOT compression ratios.
DEFAULT_WEIGHTS = {
    "history":   0.35,  # accumulated message stream → compress_history
    "shell":     0.30,  # cargo / gh / git command outputs
    "filereads": 0.20,  # Read tool calls returning file contents
    "search":    0.15,  # grep / find / glob in agent loops
}


SUB_BENCHES = [
    ("history",   "real_session_capture.py",   ["--checkpoint", "10"]),
    ("shell",     "real_session_shell.py",     []),
    ("filereads", "real_session_filereads.py", []),
    ("search",    "real_session_search.py",    []),
]


def parse_weights(s: str) -> dict[str, float]:
    """Parse `name=val,name=val,...` and validate the result sums
    close to 1.0. Falls back to DEFAULT_WEIGHTS on any parse error
    so the bench keeps running."""
    weights: dict[str, float] = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            continue
        name, val = pair.split("=", 1)
        try:
            weights[name.strip()] = float(val.strip())
        except ValueError:
            return DEFAULT_WEIGHTS.copy()
    total = sum(weights.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    # Normalise to 1.0 in case the user gave round numbers.
    return {k: v / total for k, v in weights.items()}


def run_subbench(name: str, script: str, extra_args: list[str], anonymise: bool) -> dict:
    """Run a sub-bench with --json and return its parsed output."""
    cmd = ["python3", str(ROOT / "benchmarks" / script), "--json"] + extra_args
    if anonymise:
        cmd.append("--anonymise")
    print(f"  [{name}] running {script}…", file=sys.stderr)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        return {"_error": f"exit {p.returncode}", "_stderr_tail": p.stderr[-500:]}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        return {"_error": f"non-JSON stdout: {e}", "_stdout_head": p.stdout[:200]}


def extract_numbers(name: str, data: dict) -> dict:
    """Pull the comparable headline numbers from each sub-bench's
    JSON shape (each one has a slightly different field name for
    `aggregate saved %`)."""
    if "_error" in data:
        return {"name": name, "error": data["_error"]}

    if name == "history":
        # capture has compress_history (tail) + per-diff aggregate
        last_ck = (data.get("checkpoints") or [{}])[-1]
        return {
            "name": name,
            "raw_tokens": int(last_ck.get("raw_tokens") or 0),
            "compressed_tokens": int(last_ck.get("compressed_tokens") or 0),
            "saved_pct": float(last_ck.get("saved_pct") or 0.0),
            "n_units": int(data.get("commits") or 0),
            "unit_label": "commits",
        }
    if name == "shell":
        return {
            "name": name,
            "raw_tokens": int(data.get("total_raw_tokens") or 0),
            "compressed_tokens": int(data.get("total_compressed_tokens") or 0),
            "saved_pct": float(data.get("aggregate_saved_pct") or 0.0),
            "n_units": int(data.get("commands_run") or 0),
            "unit_label": "commands",
        }
    if name == "filereads":
        return {
            "name": name,
            "raw_tokens": int(data.get("aggregate_raw_tokens") or 0),
            "compressed_tokens": int(data.get("aggregate_compressed_tokens") or 0),
            "saved_pct": float(data.get("aggregate_saved_pct") or 0.0),
            "n_units": int(data.get("total_measurements") or 0),
            "unit_label": "file × query runs",
        }
    if name == "search":
        return {
            "name": name,
            "raw_tokens": int(data.get("total_raw_tokens") or 0),
            "compressed_tokens": int(data.get("total_compressed_tokens") or 0),
            "saved_pct": float(data.get("aggregate_saved_pct") or 0.0),
            "n_units": int(data.get("searches_run") or 0),
            "unit_label": "searches",
        }
    return {"name": name, "error": "unknown sub-bench"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cargo", action="store_true",
                        help="Pass --skip-cargo to real_session_shell.")
    parser.add_argument("--skip-network", action="store_true",
                        help="Pass --skip-network to real_session_shell.")
    parser.add_argument("--anonymise", action="store_true",
                        help="Propagate --anonymise to every sub-bench.")
    parser.add_argument(
        "--weights",
        default=",".join(f"{k}={v}" for k, v in DEFAULT_WEIGHTS.items()),
        help=("Tool-traffic weights, comma-separated `name=fraction`. "
              "Auto-normalised to sum 1.0. Default reflects this "
              "repo's observed shape."),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    weights = parse_weights(args.weights)

    results = {}
    for name, script, extra in SUB_BENCHES:
        sub_extra = list(extra)
        if name == "shell":
            if args.skip_cargo:
                sub_extra.append("--skip-cargo")
            if args.skip_network:
                sub_extra.append("--skip-network")
        results[name] = run_subbench(name, script, sub_extra, args.anonymise)

    extracted = {n: extract_numbers(n, d) for n, d in results.items()}

    # Weighted estimate: assume each dimension carries `weight[name]`
    # fraction of total session token volume; the saved % under
    # that mix is just the linear combination of the four saved %s.
    weighted_saved = sum(
        weights.get(n, 0.0) * extracted[n].get("saved_pct", 0.0)
        for n in extracted
        if "error" not in extracted[n]
    )

    summary = {
        "weights": weights,
        "weighted_estimated_saved_pct": weighted_saved,
        "per_dimension": extracted,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print("=" * 96)
    print("Holistic real-session bench — 4-dimension cross-section")
    print("=" * 96)
    print(f"  {'dimension':<14} {'units':>20} {'raw':>9} {'compressed':>11} {'saved%':>8} {'weight':>8}")
    print(f"  {'-'*14} {'-'*20} {'-'*9} {'-'*11} {'-'*8} {'-'*8}")
    for name, _, _ in SUB_BENCHES:
        e = extracted[name]
        if "error" in e:
            print(f"  {name:<14} {'(error)':<20} {'':<9} {'':<11} {'':<8} {weights.get(name, 0.0):>7.2f}")
            continue
        unit_str = f"{e['n_units']} {e['unit_label']}"
        print(
            f"  {name:<14} {unit_str:<20} "
            f"{e['raw_tokens']:>9} {e['compressed_tokens']:>11} "
            f"{e['saved_pct']:>7.1f}% {weights.get(name, 0.0):>7.2f}"
        )

    print()
    print(f"  Weighted blend (default mix):   "
          f"{weighted_saved:>5.1f} % saved on a typical session")
    print()
    print(f"  Weight mix used (sums to 1.0):")
    for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<12} {v:>5.2f}")

    print()
    print("Honest scope:")
    print("  * `weighted_estimated_saved_pct` blends the 4 dimensions linearly.")
    print("    Real sessions don't split exactly into these four buckets; the")
    print("    weights are an APPROXIMATION based on this repo's observed shape.")
    print("  * Plug in your own --weights to model your workload.")
    print("  * Each sub-bench is independently meaningful; the blended number")
    print("    is for napkin-math only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
