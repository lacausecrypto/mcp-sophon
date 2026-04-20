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

use std::{fs, path::PathBuf};

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
pub use summarizer::{compress_history, MemoryConfig};
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
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            persistence_path: None,
        }
    }

    /// Attach a JSONL persistence path. If the file exists, its contents
    /// are loaded into the in-memory history. Subsequent `append()` and
    /// `reset()` calls sync to disk.
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
        self.persistence_path = Some(path);
        Ok(self)
    }

    pub fn persistence_path(&self) -> Option<&PathBuf> {
        self.persistence_path.as_ref()
    }

    fn flush_append(&self, msgs: &[Message]) -> std::io::Result<()> {
        use std::io::Write;
        let Some(path) = &self.persistence_path else {
            return Ok(());
        };
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        for m in msgs {
            let line = serde_json::to_string(m)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(f, "{}", line)?;
        }
        f.flush()
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
    pub fn append(&mut self, messages: Vec<Message>) {
        if self.persistence_path.is_some() {
            let _ = self.flush_append(&messages);
        }
        self.history.extend(messages);
    }

    /// Compress the accumulated session history.
    pub fn snapshot(&self) -> CompressedMemory {
        compress_history(&self.history, &self.config)
    }

    /// Clear the accumulated session history. If a persistence path is
    /// attached, the JSONL file is truncated.
    pub fn reset(&mut self) {
        self.history.clear();
        if let Some(path) = &self.persistence_path {
            let _ = fs::write(path, b"");
        }
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
