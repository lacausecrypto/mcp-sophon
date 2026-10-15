#!/usr/bin/env python3
"""Report HEAD-vs-base performance on the *same* CI runner.

This is informational; the separate historical-baseline gate remains strict.
Cold-start latency is noisy (5 samples); do not treat tiny differences as
proof of improvement or regression.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = [
    ("Release binary", "cold", 0, "binary_size_bytes", "bytes"),
    ("Cold start p50", "cold", 0, "cold_start_ms_p50", "ms"),
    ("Cold start p99", "cold", 0, "cold_start_ms_p99", "ms"),
    ("Idle RSS", "cold", 0, "rss_idle_mb", "MB"),
    ("Memory update p99", "scaling", None, "update_memory_ms_p99", "ms"),
    ("History compression p50", "scaling", None, "compress_history_ms_p50", "ms"),
    ("History compression p99", "scaling", None, "compress_history_ms_p99", "ms"),
]


def read_metric(data: dict, source: str, index: int | None, key: str) -> float:
    item = data[source]
    if index is not None:
        item = item[index]
    value = float(item[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {source}.{key}: {value}")
    return value


def percent_change(base: float, head: float) -> float:
    if base == 0:
        if head == 0:
            return 0.0
        raise ValueError("cannot compare nonzero HEAD against zero base")
    return (head - base) * 100 / base


def render(base: dict, head: dict) -> str:
    rows = [
        "## Sophon: PR HEAD vs main on one runner",
        "",
        "Lower is better for all reported metrics. Positive delta means HEAD is",
        "slower/larger. This is a diagnostic comparison, **not a baseline reset**.",
        "",
        "| Metric | main | PR HEAD | Change |",
        "|---|---:|---:|---:|",
    ]
    for title, source, index, key, unit in METRICS:
        a = read_metric(base, source, index, key)
        b = read_metric(head, source, index, key)
        change = percent_change(a, b)
        if unit == "bytes":
            a_str = f"{a / 1_000_000:.3f} MB"
            b_str = f"{b / 1_000_000:.3f} MB"
        else:
            a_str = f"{a:.3f} {unit}"
            b_str = f"{b:.3f} {unit}"
        rows.append(f"| {title} | {a_str} | {b_str} | {change:+.1f}% |")
    rows += [
        "",
        "_Cold-start latency is measured with only five launches and has substantial",
        "runner noise. Repeat runs before attributing small differences to code._",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base-cold", "head-cold", "base-scaling", "head-scaling"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    base = {
        "cold": json.loads(args.base_cold.read_text()),
        "scaling": json.loads(args.base_scaling.read_text()),
    }
    head = {
        "cold": json.loads(args.head_cold.read_text()),
        "scaling": json.loads(args.head_scaling.read_text()),
    }
    print(render(base, head))


if __name__ == "__main__":
    main()
