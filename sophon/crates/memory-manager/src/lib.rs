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

use std::{fs, io, io::{Read, Seek, SeekFrom, Write}, path::PathBuf};

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

/// Transaction metadata rides alongside normal Message fields so that
/// existing JSONL readers still see one Message per line. An incomplete batch
/// is discarded on recovery rather than exposing a valid-looking prefix.
#[derive(serde::Deserialize)]
struct BatchMarker {
    id: String,
    seq: usize,
    count: usize,
}

struct PendingBatch {
    id: String,
    count: usize,
    messages: Vec<Message>,
}

/// Replay complete legacy messages and committed append batches. Crash-torn
/// tails are truncated to the last committed byte boundary. A valid legacy
/// final line without a trailing newline is accepted for compatibility.
fn load_jsonl_transactions(path: &std::path::Path) -> io::Result<Vec<Message>> {
    let raw = fs::read(path)?;
    let mut history = Vec::new();
    let mut committed_end = 0usize;
    let mut offset = 0usize;
    let mut pending: Option<PendingBatch> = None;

    for line in raw.split_inclusive(|b| *b == b'\n') {
        let end = offset + line.len();
        let complete_line = line.ends_with(b"\n");
        let value: serde_json::Value = match serde_json::from_slice(line) {
            Ok(value) => value,
            Err(err) if pending.is_none() && complete_line => {
                // Keep the old permissive behavior for malformed *legacy*
                // complete rows, while still replaying subsequent good rows.
                tracing::warn!(error = %err, "skipping invalid legacy memory JSONL row");
                committed_end = end;
                offset = end;
                continue;
            }
            Err(_) => break,
        };
        let message: Message = match serde_json::from_value(value.clone()) {
            Ok(message) => message,
            Err(err) if pending.is_none() && complete_line => {
                tracing::warn!(error = %err, "skipping invalid legacy memory row");
                committed_end = end;
                offset = end;
                continue;
            }
            Err(_) => break,
        };

        if let Some(marker_value) = value.get("_sophon_txn") {
            let Ok(marker) = serde_json::from_value::<BatchMarker>(marker_value.clone()) else {
                break;
            };
            // Bound corrupted metadata to avoid unbounded allocations from
            // disk-controlled counts. No batch is allocated by count alone.
            if marker.count == 0 || marker.count > 100_000 || marker.seq >= marker.count {
                break;
            }
            if pending.is_none() {
                if marker.seq != 0 {
                    break;
                }
                pending = Some(PendingBatch {
                    id: marker.id.clone(),
                    count: marker.count,
                    messages: Vec::new(),
                });
            }
            let Some(batch) = pending.as_mut() else {
                break;
            };
            if batch.id != marker.id
                || batch.count != marker.count
                || marker.seq != batch.messages.len()
            {
                break;
            }
            batch.messages.push(message);
            if batch.messages.len() == batch.count {
                history.extend(pending.take().expect("checked pending").messages);
                committed_end = end;
            }
        } else {
            // A legacy message cannot arrive in the middle of an unfinished
            // transaction. Stop before the batch to avoid partial replay.
            if pending.is_some() {
                break;
            }
            history.push(message);
            committed_end = end;
        }

        if !complete_line && pending.is_some() {
            // An incomplete final batch is uncommitted even if its last
            // individual row happened to be parseable.
            break;
        }
        offset = end;
    }

    if committed_end < raw.len() {
        // Truncate only the uncommitted suffix (partial JSON, orphaned
        // batch rows), not a previously committed legacy row.
        fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(committed_end as u64)?;
        tracing::warn!(
            discarded_bytes = raw.len() - committed_end,
            "discarded incomplete persistent-memory transaction"
        );
    }
    Ok(history)
}

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
            self.history.extend(load_jsonl_transactions(&path)?);
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

    /// Append transaction records in O(new batch size) rather than rewriting
    /// the entire session. For multi-message updates, each row retains its
    /// legacy Message fields plus private per-batch sequence metadata.
    /// If a process dies mid-batch, the loader drops its incomplete suffix.
    fn flush_append(&self, messages: &[Message]) -> io::Result<()> {
        let Some(path) = &self.persistence_path else {
            return Ok(());
        };
        if messages.is_empty() {
            return Ok(());
        }

        let mut bytes = Vec::new();
        if messages.len() == 1 {
            serde_json::to_writer(&mut bytes, &messages[0])
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            bytes.push(b'\n');
        } else {
            let batch_id = uuid::Uuid::new_v4().to_string();
            for (seq, message) in messages.iter().enumerate() {
                let mut record = serde_json::to_value(message)
                    .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
                let Some(object) = record.as_object_mut() else {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "message must be an object"));
                };
                object.insert(
                    "_sophon_txn".to_string(),
                    serde_json::json!({"id": batch_id, "seq": seq, "count": messages.len()}),
                );
                serde_json::to_writer(&mut bytes, &record)
                    .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
                bytes.push(b'\n');
            }
        }

        let mut file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)?;
        let old_len = file.metadata()?.len();
        if old_len > 0 {
            // Legacy files may end in a complete JSON message without a
            // newline. Preserve it by inserting the missing delimiter.
            file.seek(SeekFrom::End(-1))?;
            let mut last = [0u8; 1];
            file.read_exact(&mut last)?;
            if last[0] != b'\n' {
                bytes.insert(0, b'\n');
            }
        }

        if let Err(error) = file.write_all(&bytes).and_then(|_| file.flush()) {
            // Failed writes must not expose a new RAM state; attempt to
            // restore the old file length too. On a process crash the
            // loader independently handles a torn batch suffix.
            if let Err(rollback) = file.set_len(old_len) {
                return Err(io::Error::other(format!(
                    "append failed: {error}; rollback also failed: {rollback}"
                )));
            }
            return Err(error);
        }
        // The bytes are in the OS page cache, durable against a process
        // crash. This fast path deliberately does not fsync per turn:
        // power-loss durability requires a separate opt-in WAL/fsync policy.
        Ok(())
    }

    /// Serialize a complete JSONL snapshot before touching the destination.
    /// Atomic replacement provides whole-batch durability, including after an
    /// abrupt process stop. It deliberately trades O(history size) disk IO per
    /// persisted append for compatibility with the existing one-message-per-
    /// line JSONL format; installations with very large histories may benefit
    /// from a future write-ahead log.
    fn flush_snapshot(&self, existing: &[Message], new_messages: &[Message]) -> io::Result<()> {
        let Some(path) = &self.persistence_path else {
            return Ok(());
        };
        let mut bytes = Vec::new();
        for message in existing.iter().chain(new_messages) {
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
            self.flush_append(&messages)?;
        }
        self.history.extend(messages);

        self.refresh_rolling_after_commit();
        Ok(())
    }

    /// Replace the entire session in a single atomic operation. Used for
    /// update_memory(reset=true, messages=[...]) so reset + new history
    /// cannot be split into two separately committed writes.
    pub fn try_replace(&mut self, messages: Vec<Message>) -> io::Result<()> {
        self.flush_snapshot(&[], &messages)?;
        self.history = messages;
        self.rolling = None;
        if let Some(path) = &self.persistence_path {
            delete_rolling_sidecar(path);
        }
        self.refresh_rolling_after_commit();
        Ok(())
    }

    fn refresh_rolling_after_commit(&mut self) {
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
        self.try_replace(Vec::new())
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

#[cfg(test)]
mod atomic_persistence_tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn multi_message_append_roundtrips_as_legacy_jsonl() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut memory = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        memory
            .try_append(vec![Message::user("first"), Message::assistant("second")])
            .unwrap();
        assert_eq!(memory.history_len(), 2);
        let lines = fs::read_to_string(&path).unwrap();
        assert_eq!(lines.lines().count(), 2);
        assert!(lines
            .lines()
            .all(|line| serde_json::from_str::<Message>(line).is_ok()));

        let resumed = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        assert_eq!(resumed.history_len(), 2);
    }

    #[test]
    fn failed_append_does_not_publish_uncommitted_ram_state() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut memory = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        fs::create_dir(&path).unwrap();
        assert!(memory
            .try_append(vec![Message::user("must not commit")])
            .is_err());
        assert_eq!(memory.history_len(), 0);
        fs::remove_dir(&path).unwrap();
        memory.try_append(vec![Message::user("committed")]).unwrap();
        assert_eq!(memory.history_len(), 1);
    }

    #[test]
    fn failed_reset_preserves_both_ram_history_and_previous_disk_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let backup = dir.path().join("saved.jsonl");
        let mut memory = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        memory
            .try_append(vec![Message::user("do not lose")])
            .unwrap();
        let old = fs::read(&path).unwrap();
        fs::rename(&path, &backup).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(memory.try_reset().is_err());
        assert_eq!(memory.history_len(), 1);
        fs::remove_dir(&path).unwrap();
        fs::rename(&backup, &path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), old);
    }

    #[test]
    fn reset_with_messages_commits_one_replacement_and_cleans_corrupt_tail() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut memory = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        memory.try_append(vec![Message::user("old")]).unwrap();
        memory
            .try_replace(vec![Message::assistant("new"), Message::user("another")])
            .unwrap();
        assert_eq!(memory.history_len(), 2);
        let resumed = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        assert_eq!(resumed.history_len(), 2);
        assert_eq!(resumed.history[0].content, "new");

        let mut broken = fs::read(&path).unwrap();
        broken.extend_from_slice(b"{\\\"incomplete");
        fs::write(&path, broken).unwrap();
        let mut recovered = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&path)
            .unwrap();
        assert_eq!(recovered.history_len(), 2);
        recovered
            .try_append(vec![Message::user("repaired")])
            .unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 3);
        assert!(raw
            .lines()
            .all(|line| serde_json::from_str::<Message>(line).is_ok()));
    }
}
