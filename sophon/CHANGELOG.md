# Changelog

All notable changes to Sophon are documented here. Versions follow semantic
versioning (pre-1.0: minor = features/behaviour changes, patch = fixes).

## [0.6.0] — 2026-06-19

Audit-driven correctness, security and robustness pass (see `../audit18.md`).
End-to-end correctness bench (N=51 real-repo tasks, blind LLM judge): key-fact
recall vs naive truncation at equal budget went from **+4.2 pts (CI spans 0)**
to **+20.6 pts (95% CI [+13.1, +28.9])**; `history` op 19.1% → 32.1%, `output`
op −16.3 → −2.0. 448 tests green, clippy clean.

### Added / Changed
- **`compress_prompt` now uses its budget and is query-aware by default.** A
  deterministic BM25 lexical scorer (no ML, no model download) ranks sections
  against the *actual* query terms — including identifiers the frozen keyword
  dictionary never matched — and a budget backfill fills spare budget with the
  next-most-relevant sections. This kills the catastrophic under-fill ("9
  tokens emitted of a ~700 budget") that lost to plain truncation. Budget math
  is now header-aware so output never overshoots `max_tokens`.
- **`compress_history` is query-aware.** When the `query` argument is provided,
  the summary of dropped older turns is ranked by relevance to the question
  (then information density), so the line that answers the query lands in the
  summary instead of losing the budget race. No query → unchanged behaviour.
- **`compress_output` safety floor.** List-like filters (`generic`, `grep`)
  never preserve less of the original than a plain head-truncation to the same
  budget would; over-aggressive aggregation that dropped the answer now falls
  back to truncation (so compress_output is, at worst, a tie — never an
  anti-value).
- **Server survives tool panics.** Release profile switched `panic = "abort"`
  → `"unwind"` so the existing per-tool `spawn_blocking`/`JoinSet` isolation
  actually works: a panic in any tool returns a JSON-RPC internal error and
  keeps the server (and every other session) alive, instead of aborting the
  whole process.

### Fixed
- **Path traversal (F1):** delta file read/write is confined to a filesystem
  root (`SOPHON_FS_ROOT`, default: working directory). Prompt-injected
  `../../.ssh/id_rsa` is rejected; containment is decided on the canonical
  (symlink-resolved) path, which also blocks symlink escapes without
  over-rejecting legitimately symlinked roots (e.g. macOS `/tmp`).
- **Multi-edit corruption (F2):** overlapping structured edits are rejected
  up-front (`EditError::OverlappingEdits`) instead of silently splicing over
  each other and destroying data; disjoint adjacent edits still apply.
- **CRLF preservation (F3):** files written via delta ops keep their original
  line endings instead of being silently converted CRLF → LF.
- **Fragment decode collision (F4):** content that legitimately contains
  `[FRAGMENT:…]` no longer fails the whole decode — unknown references are left
  verbatim (decode is now infallible).
- **Embedder honesty (F5/F6):** requesting an ML embedder
  (`SOPHON_EMBEDDER=bge`) that wasn't compiled in now logs an explicit warning
  and falls back to the deterministic hash embedder, instead of silently
  pretending to be semantic. Tool descriptions corrected (`--features bge`).
- **Multibyte panic:** the per-section embedding input is truncated on a UTF-8
  char boundary, fixing a `[..500]` byte-slice panic on multibyte prompt text.

[0.6.0]: https://github.com/lacausecrypto/mcp-sophon/releases/tag/v0.6.0
