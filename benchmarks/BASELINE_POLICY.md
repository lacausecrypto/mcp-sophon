# Performance baseline policy

## Calibration recorded on 2026-10-15

The previous `benchmarks/baseline.json` came from an earlier, smaller Sophon
build (roughly 5.252 MB). It no longer represented the current `main`
implementation (6.377 MB), and the benchmark workflow was already failing on
`main` for at least ten consecutive days before the stabilization PR.

We archived the previous values intact in
[`baseline-pre-2026-10-15.json`](./baseline-pre-2026-10-15.json).

The new `baseline.json` uses **the main binary, not the PR binary**, measured
on the **same ubuntu-latest runner and with the same Rust toolchain** as the PR
HEAD in [comparison run 38703110954](https://github.com/lacausecrypto/mcp-sophon/actions/runs/38703110954).
The commit compared on main was `9ff478b885c67bd5446f2e9b8205f89b42b9447d`.
Raw JSON files are retained as an artifact on that workflow run.
New latency/RSS metrics are rounded to 3 decimals; binary size is exact.

The archived baseline and [ten days of failed main benchmark runs](https://github.com/lacausecrypto/mcp-sophon/actions/runs/38533498029)
provide historical evidence that the size and latency drift predated the PR.
For example, the October 14 main run reported a 6,376,720-byte release binary
and a 69.278 ms cold-start median against the old 34.086 ms reference.

## What this does and does not mean

- This is a **reference-point correction**, **not an optimization**.
- The pre-calibration performance gap remains historically visible.
- **No tolerance percentages were increased.** The existing size, startup,
  throughput and latency metrics retain the same strict detection rules.
- The compressor output-quality/savings thresholds are unchanged from the
  previous baseline; the same-runner run did not recalibrate those metrics.
- The main-vs-PR comparison reports relative changes separately and does not
  gate the workflow. Run it again if a small change is close to runner noise.
- Cold start and p99 latency are especially sensitive to runner scheduling,
  system caches and sample count. Interpret small changes cautiously.

For future changes, only update the baseline after demonstrating that the new
reference is intentional, repeatable, and attributable to a specific version.
Preserve the previous reference before replacing it.
