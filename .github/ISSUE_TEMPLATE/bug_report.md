---
name: Bug report
about: Report incorrect behavior, wrong numbers, or broken tooling
title: "[BUG] "
labels: bug
assignees: ''
---

**What you ran**
The exact command line, JSON-RPC payload, or tool call arguments.

**What you expected**
One sentence.

**What you got**
Paste stderr, the JSON response, or the relevant token counts.

**Environment**
- OS + arch (`uname -a` or `systeminfo`):
- Rust version (`rustc --version`) if built from source:
- Sophon version (`sophon --version`):
- How installed (npm / release binary / source):

**Reproducer**
If the bug affects a benchmark number in BENCHMARK.md, a minimal script
that plugs into the existing bench runners is ideal.
