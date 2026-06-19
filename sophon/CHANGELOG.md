# Changelog

All notable changes to Sophon are documented here. Versions follow semantic
versioning (pre-1.0: minor = features/behaviour changes, patch = fixes).

## [Unreleased]

Compression-power & token-economy pass (see `../audit19-plan.md`). Five
verified levers from a fresh adversarial audit; every change ships with a
regression test, full suite green (470+ tests), clippy clean.

Measured effect (N=51 real-repo correctness bench, blind judge, vs the
v0.6.0 baseline): **prompt** key-fact recall **48.8% → 51.3%** (the parser and
query-aware-truncation fixes). The sophon−truncate *margin* is statistically
flat (+20.6 → +19.7, CIs overlap) because the truncation baseline is
calibrated to Sophon's emitted token count and shifts with it. `output` is
unchanged on this bench (its samples are uncolored, so the ANSI fix below
can't show — it is proven separately by a colored-input regression test).
The envelope/accounting fixes are correctness/honesty improvements a
recall bench does not capture.

### Added
- **`build` output filter (T2.2).** `cargo build`/`clippy`/`check`, `tsc`,
  `eslint`, `biome`, `make`, `ninja`, `bazel`, `go build`, and `… run
  build` scripts now route to a dedicated filter instead of the generic
  (truncation-floored) fallback. It drops `Compiling`/`Downloading`/
  progress noise while keeping every diagnostic (errors, warnings, the
  `-->`/`|`/`note:` context) and the final `Finished` success line.
  Deterministic per-command bench: `cargo build --release` 107 → 18 tokens
  (83% saved).
- **Carriage-return progress collapse (T2.1).** Download/build progress
  bars that redraw a line in place with `\r` are folded to their final
  state (CRLF line endings preserved). Runs as a pre-pass after the ANSI
  strip.
- **Stack-trace folding (T2.3).** A new `FoldStackFrames` strategy (wired
  into the cargo-test, pytest and build filters) folds the repetitive
  middle of deep Python/Node/Java/Rust traces — keeping the exception
  message and the first/last few frames — only when a trace exceeds 8
  frames, so short traces are untouched.

### Changed
- **`compress_prompt` de-duplicates redundant sections (T4.2).** When two
  selected sections carry the same content (normalized: case- and
  whitespace-insensitive) — e.g. a block repeated under two headers — only
  one keeps the budget; backfill won't re-spend it on a near-copy either.
  Neutral on the bench (its 24 real source files have no duplicate
  sections, so no regression) — the win is on genuinely redundant prompts.
- **`compress_output` strips ANSI/terminal escapes up-front (T1.1).** Real
  tool output is colored, and the per-filter regexes are anchored
  (`^test … ok$`), so colored input defeated every filter and
  `compress_output` degenerated to ~truncation. A single pre-pass unblocks
  all filters (colored `cargo test` goes from ratio ~0.98 to <0.5). The
  stripped text is the canonical base for the safety floor too, so the
  floor no longer spuriously reverts the strip.
- **`compress_prompt` truncates long sections query-aware (T1.3).** When a
  section overflows the budget, Sophon now keeps the lines that overlap the
  query (in original order) instead of a blind prefix, so an answer buried
  mid/late in a long section survives. Falls back to the prefix cut with no
  query or no overlap; always clamped to the budget.
- **`encode_fragments` no longer echoes full fragment content (T0.3).** The
  response now summarizes newly-registered fragments (id/hash/token_count/
  category) instead of re-sending the bytes the placeholder just removed —
  which made the first encode net-negative while `tokens_saved` still
  claimed a win. Content stays in the store, retrievable via
  `decode_fragments`.

### Performance
- **`compress_history` no longer builds a thrown-away index (T3.1).** The
  embedding-heavy `SemanticIndex` was built on every call and then stripped
  from the response unless the caller passed `include_index` — pure wasted
  work on the default path. It is now built only when actually requested.

### Fixed
- **History relevance matches identifier components, not substrings (T3.3).**
  The query-aware history summary scored a line by raw substring, so `age`
  matched `page` and `storage` and pulled irrelevant lines into the summary.
  It now matches whole tokens and their snake_case/camelCase parts, keeping
  the intended wins (`age` ↦ `oldest_entry_age_seconds`, `cache` ↦
  `cache.rs`) without the false positives. History recall 32.1% → 32.7% on
  the N=51 bench; overall sophon−truncate +19.7 → +20.0 pts.
- **Parser XML hijack (T1.2).** A single incidental `<tag>…</tag>` in a
  Markdown prompt used to flip the whole parse to the XML branch, dropping
  every `##` section (bench prompt-002: 17 tokens emitted of a ~700
  budget). XML now wins only when it actually structures the prompt (covers
  ≥50% of it) or when there is no Markdown/numbered alternative.
- **Delta token accounting (T0.2).** `read_file_delta` counted its cost on
  the Rust `{:?}` Debug form, not the JSON actually sent on the wire — a
  different string whose token count diverges from reality (and can push a
  delta past the real budget). It now counts the serialized JSON.

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
