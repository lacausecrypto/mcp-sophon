pub mod fact_cards;
pub mod fact_extractor;
pub mod graph;
pub mod index;
pub mod llm_client;
pub mod llm_reranker;
pub mod message;
pub mod multihop;
pub mod query_decomposer;
pub mod query_rewriter;
pub mod question_classifier;
pub mod react;
pub mod reconstructor;
pub mod summarizer;
pub mod tail_summary;

use std::{fs, io, path::PathBuf};

pub use fact_cards::{extract_fact_cards, FactCards, FactEvent};
pub use fact_extractor::extract_facts;
pub use llm_client::{call_llm, llm_cmd_is_configured, DEFAULT_LLM_CMD};
pub use llm_reranker::rerank_chunks;
pub use message::{CompressedMemory, Fact, FactCategory, Message, Role};
pub use multihop::is_likely_multihop;
pub use query_decomposer::decompose_query;
pub use query_rewriter::hyde_rewrite_query;
pub use question_classifier::{classify_question, QuestionMode};
pub use react::{react_decide, ReactDecision};
pub use reconstructor::expand_memory;
pub use summarizer::{
    compress_history, compress_history_with_rolling, refresh_rolling_summary, MemoryConfig,
    RollingSummary, DEFAULT_ROLLING_REFRESH_THRESHOLD,
};
pub use tail_summary::summarise_tail;

/// High-level facade for conversation memory compression.
///
/// Three usage modes coexist:
/// - **Stateless batch**: `compress(messages)` — client provides the full
///   history each call. Cheap, no internal state.
/// - **Stateful session (RAM)**: `append(new_msgs)` + `snapshot()` — the
///   manager owns the accumulated history within a single `sophon serve`
///   run, lost on restart.
/// - **Persistent session (disk)**: `with_persistence(path)` or
///   `load(path)` — the history is JSONL-backed at `path`, appends are
///   flushed to disk, and a later `sophon serve` can resume from the file.
#[derive(Debug, Clone)]
pub struct MemoryManager {
    config: MemoryConfig,
    history: Vec<Message>,
    persistence_path: Option<PathBuf>,
    /// Pre-computed rolling summary. Populated after `append()`
    /// crosses the refresh threshold when `SOPHON_ROLLING_SUMMARY=1`
    /// is set. `None` means the feature is inactive (default) — in
    /// which case every public read path is byte-identical to the
    /// pre-2B implementation.
    rolling: Option<RollingSummary>,
    /// Refresh threshold in messages. Defaults to
    /// `DEFAULT_ROLLING_REFRESH_THRESHOLD` (50); env var
    /// `SOPHON_ROLLING_THRESHOLD` overrides.
    rolling_threshold: usize,
    /// Whether the rolling-summary feature is active. Read once
    /// at construction from `SOPHON_ROLLING_SUMMARY=1`. Stored as
    /// a flag rather than re-read every append for thread-safety
    /// and reproducibility under bench harnesses that toggle env
    /// vars.
    rolling_enabled: bool,
}

fn rolling_summary_path(history_path: &std::path::Path) -> PathBuf {
    let mut p = history_path.to_path_buf();
    let new_name = match p.file_name() {
        Some(n) => format!("{}.sophon-summary.json", n.to_string_lossy()),
        None => "sophon-summary.json".to_string(),
    };
    p.set_file_name(new_name);
    p
}

fn read_rolling_sidecar(history_path: &std::path::Path) -> Option<RollingSummary> {
    let p = rolling_summary_path(history_path);
    let raw = fs::read_to_string(&p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_rolling_sidecar(
    history_path: &std::path::Path,
    rolling: &RollingSummary,
) -> std::io::Result<()> {
    let p = rolling_summary_path(history_path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(rolling)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    sophon_core::atomic_file::replace(&p, raw.as_bytes())
}

fn delete_rolling_sidecar(history_path: &std::path::Path) {
    let p = rolling_summary_path(history_path);
    let _ = fs::remove_file(&p);
}

fn rolling_enabled_from_env() -> bool {
    std::env::var("SOPHON_ROLLING_SUMMARY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn rolling_threshold_from_env() -> usize {
    std::env::var("SOPHON_ROLLING_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 10) // sanity floor
        .unwrap_or(DEFAULT_ROLLING_REFRESH_THRESHOLD)
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            persistence_path: None,
            rolling: None,
            rolling_threshold: rolling_threshold_from_env(),
            rolling_enabled: rolling_enabled_from_env(),
        }
    }

    /// Attach a JSONL persistence path. If the file exists, its contents
    /// are loaded into the in-memory history. Subsequent `append()` and
    /// `reset()` calls sync to disk.
    ///
    /// Also loads the rolling-summary sidecar at
    /// `<path>.sophon-summary.json` if it exists, so that a restarted
    /// `sophon serve` resumes exactly where the previous run left off
    /// without re-running the LLM summariser on the entire backlog.
    pub fn with_persistence<P: Into<PathBuf>>(mut self, path: P) -> std::io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if path.exists() {
            let raw = fs::read_to_string(&path)?;
            for line in raw.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(msg) = serde_json::from_str::<Message>(line) {
                    self.history.push(msg);
                }
            }
        }
        // Load sidecar only if rolling is enabled — otherwise we'd
        // resurrect stale state when the user disables the feature.
        if self.rolling_enabled {
            self.rolling = read_rolling_sidecar(&path);
            // Discard a sidecar that's ahead of the live history (the
            // history file was truncated externally).
            if let Some(r) = &self.rolling {
                if self.history.is_empty() || r.summarized_until > self.history.len() {
                    self.rolling = None;
                    delete_rolling_sidecar(&path);
                }
            }
        }
        self.persistence_path = Some(path);
        Ok(self)
    }

    pub fn persistence_path(&self) -> Option<&PathBuf> {
        self.persistence_path.as_ref()
    }

    /// Serialize a complete JSONL snapshot before touching the destination.
    /// Atomic replacement provides whole-batch durability, including after an
    /// abrupt process stop. It deliberately trades O(history size) disk IO per
    /// persisted append for compatibility with the existing one-message-per-
    /// line JSONL format; installations with very large histories may benefit
    /// from a future write-ahead log.
    fn flush_snapshot(&self, new_messages: &[Message]) -> io::Result<()> {
        let Some(path) = &self.persistence_path else {
            return Ok(());
        };
        let mut bytes = Vec::new();
        for message in self.history.iter().chain(new_messages) {
            serde_json::to_writer(&mut bytes, message)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            bytes.push(b'\n');
        }
        sophon_core::atomic_file::replace(path, &bytes)
    }

    pub fn compress(&self, messages: &[Message]) -> CompressedMemory {
        compress_history(messages, &self.config)
    }

    /// Compress with per-call overrides on top of the stored config.
    /// Unset overrides fall back to `self.config`.
    pub fn compress_with_overrides(
        &self,
        messages: &[Message],
        max_tokens: Option<usize>,
        recent_window: Option<usize>,
    ) -> CompressedMemory {
        let mut cfg = self.config.clone();
        if let Some(max) = max_tokens {
            cfg.max_tokens = max;
        }
        if let Some(win) = recent_window {
            cfg.recent_window = win;
        }
        compress_history(messages, &cfg)
    }

    /// Append messages to the session history. If a persistence path is
    /// attached, the messages are also appended to the JSONL file.
    ///
    /// When rolling-summary is enabled (`SOPHON_ROLLING_SUMMARY=1`)
    /// this also calls `refresh_rolling_summary` after the append. If
    /// the un-summarised tail crosses the threshold, a new summary is
    /// computed (one LLM call when `SOPHON_LLM_CMD` is set, otherwise
    /// pure-Rust heuristic) and persisted to the sidecar — moving the
    /// summary cost from query time to ingest time. Subsequent
    /// `snapshot()` calls become near-instant: they just stitch the
    /// stored summary to the live recent window.
    /// Durable append with explicit errors. The in-memory history is updated
    /// only after the full JSONL snapshot has atomically replaced the previous
    /// file. No caller can observe uncommitted state after an IO failure.
    pub fn try_append(&mut self, messages: Vec<Message>) -> io::Result<()> {
        if !messages.is_empty() {
            self.flush_snapshot(&messages)?;
        }
        self.history.extend(messages);

        if self.rolling_enabled {
            if let Some(new_rolling) = refresh_rolling_summary(
                &self.history,
                self.rolling.as_ref(),
                &self.config,
                self.rolling_threshold,
            ) {
                if let Some(path) = &self.persistence_path {
                    if let Err(err) = write_rolling_sidecar(path, &new_rolling) {
                        // The history snapshot is committed already. A
                        // derived sidecar failure must not ask the caller to
                        // retry the append and accidentally duplicate it.
                        tracing::warn!(error = %err, "rolling summary sidecar could not be persisted");
                    }
                }
                self.rolling = Some(new_rolling);
            }
        }
        Ok(())
    }

    /// Backwards-compatible convenience API. Prefer try_append() when a
    /// caller can surface storage errors to the end user.
    pub fn append(&mut self, messages: Vec<Message>) {
        if let Err(err) = self.try_append(messages) {
            tracing::error!(error = %err, "memory append failed; RAM state was not advanced");
        }
    }

    /// Compress the accumulated session history.
    ///
    /// When a rolling summary is present this serves it directly (the
    /// fast path); otherwise falls back to the full-history summariser.
    /// Output is identical to the pre-2B path when the feature is off.
    pub fn snapshot(&self) -> CompressedMemory {
        compress_history_with_rolling(&self.history, self.rolling.as_ref(), &self.config)
    }

    /// Clear the accumulated session history. If a persistence path is
    /// attached, the JSONL file is truncated and the rolling-summary
    /// sidecar is deleted.
    /// Atomic reset: do not discard the in-memory session unless the empty
    /// persistent snapshot has been committed successfully.
    pub fn try_reset(&mut self) -> io::Result<()> {
        if let Some(path) = &self.persistence_path {
            sophon_core::atomic_file::replace(path, b"")?;
        }
        self.history.clear();
        self.rolling = None;
        if let Some(path) = &self.persistence_path {
            delete_rolling_sidecar(path);
        }
        Ok(())
    }

    /// Backwards-compatible convenience API. Prefer try_reset() in RPCs.
    pub fn reset(&mut self) {
        if let Err(err) = self.try_reset() {
            tracing::error!(error = %err, "memory reset failed; previous history retained");
        }
    }

    /// Direct accessor for the current rolling summary state. Used by
    /// the `sophon doctor` diagnostic and by the bench harness to
    /// assert that the feature actually fires under expected workloads.
    pub fn rolling_summary(&self) -> Option<&RollingSummary> {
        self.rolling.as_ref()
    }

    /// Whether the feature was active when this manager was constructed.
    /// Reflects `SOPHON_ROLLING_SUMMARY` at instantiation time, not
    /// the current env var (deliberate — keeps behaviour
    /// reproducible mid-session).
    pub fn rolling_enabled(&self) -> bool {
        self.rolling_enabled
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn expand(
        &self,
        compressed: &CompressedMemory,
        query: &str,
        original_messages: &[Message],
    ) -> Vec<Message> {
        expand_memory(compressed, query, original_messages)
    }
}
