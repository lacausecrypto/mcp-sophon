# Deprecated benchmarks

Scripts archived on 2026-04-20 as part of the v0.5.0 re-scope toward
pure compression. They were written during the v0.4.0 recall-chasing
sprint (LOCOMO accuracy targeting, HyDE / fact-cards / entity-graph
stacks) and are no longer representative of Sophon's positioning.

Sophon now positions itself **orthogonally** to mem0 / Letta / Zep:

- Metrics we optimise: tokens saved %, latency p50/p99, binary size,
  canary preservation, MCP compliance.
- Metrics we no longer chase: LOCOMO absolute accuracy, head-to-head
  neural recall benchmarks.

See `CHANGELOG.md` entry for v0.5.0 for the full rationale.

These files are kept for historical reproducibility only. They are
not part of the supported benchmark suite and may reference code
paths, flags, or modules that have been removed.
