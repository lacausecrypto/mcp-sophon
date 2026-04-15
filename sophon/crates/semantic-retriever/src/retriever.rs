//! Retriever facade — ties chunker + embedder + index + store together.
//!
//! Lifecycle:
//! 1. `Retriever::open(config)` loads any existing JSONL store from disk and
//!    rebuilds the in-memory vector index by re-embedding stored chunks.
//!    No model is downloaded; embeddings are deterministic so the rebuild
//!    is reproducible.
//! 2. `index_messages(messages)` chunks new messages, embeds the new ones
//!    only (existing ids are skipped via store dedup), and adds them to the
//!    index. Idempotent.
//! 3. `retrieve(query)` embeds the query, runs cosine k-NN, applies token
//!    budget + min-score filters, and returns scored chunks in descending
//!    order.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::chunker::{chunk_messages, Chunk, ChunkConfig, ChunkInputMessage, ChunkType};
use crate::embedder::{Embedder, EmbedderError, HashEmbedder};
use crate::index::VectorIndex;
use crate::store::{ChunkStore, StoreError};

#[derive(Debug, thiserror::Error)]
pub enum RetrieverError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Embedder(#[from] EmbedderError),
    #[error("index error: {0}")]
    Index(String),
}

#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Number of chunks to return.
    pub top_k: usize,
    /// Minimum cosine similarity to consider a chunk relevant. Cosine over
    /// L2-normalized vectors lives in `[-1, 1]`; with the HashEmbedder
    /// keyword-overlap baseline a score of 0.15-0.3 already indicates real
    /// shared vocabulary.
    pub min_score: f32,
    /// Hard cap on retrieved chunk tokens. Prevents the retrieval payload
    /// from blowing up the prompt the caller is trying to compress.
    pub max_retrieved_tokens: usize,
    /// Drop chunks whose content matches a chunk already in the result.
    pub deduplicate: bool,
    /// Where to persist the JSONL chunk store.
    pub storage_path: PathBuf,
    /// Chunking parameters.
    pub chunk_config: ChunkConfig,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.15,
            max_retrieved_tokens: 1000,
            deduplicate: true,
            storage_path: PathBuf::from(".sophon-retriever/chunks.jsonl"),
            chunk_config: ChunkConfig::default(),
        }
    }
}

/// Result of a retrieval query — what callers see.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub chunks: Vec<ScoredChunk>,
    pub total_searched: usize,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk: RetrievedChunk,
    pub score: f32,
}

/// Public-facing chunk shape (drops the embedding to keep MCP responses small).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub id: String,
    pub content: String,
    pub token_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub session_id: Option<String>,
    pub chunk_type: ChunkType,
}

impl From<&Chunk> for RetrievedChunk {
    fn from(c: &Chunk) -> Self {
        Self {
            id: c.id.clone(),
            content: c.content.clone(),
            token_count: c.token_count,
            timestamp: c.timestamp,
            session_id: c.session_id.clone(),
            chunk_type: c.chunk_type,
        }
    }
}

pub struct Retriever {
    embedder: Arc<dyn Embedder>,
    index: VectorIndex,
    store: ChunkStore,
    config: RetrieverConfig,
}

impl std::fmt::Debug for Retriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("embedder", &self.embedder.name())
            .field("index_len", &self.index.len())
            .field("store_len", &self.store.len())
            .field("config", &self.config)
            .finish()
    }
}

impl Retriever {
    /// Open a retriever using the deterministic HashEmbedder. Loads any
    /// existing store and re-embeds it into the in-memory index.
    pub fn open(config: RetrieverConfig) -> Result<Self, RetrieverError> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default());
        Self::with_embedder(config, embedder)
    }

    /// Open a retriever with a caller-provided embedder. Useful for tests
    /// and for the future BERT backend.
    pub fn with_embedder(
        config: RetrieverConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, RetrieverError> {
        let store = ChunkStore::open(&config.storage_path)?;
        let mut index = VectorIndex::new(embedder.dimension());

        // Re-embed existing chunks. Embeddings are deterministic so this
        // produces the same vectors as when they were first indexed.
        for chunk in store.iter() {
            let v = embedder.embed(&chunk.content)?;
            index
                .upsert(&chunk.id, &v)
                .map_err(|e| RetrieverError::Index(e.to_string()))?;
        }

        Ok(Self {
            embedder,
            index,
            store,
            config,
        })
    }

    pub fn config(&self) -> &RetrieverConfig {
        &self.config
    }

    pub fn store_len(&self) -> usize {
        self.store.len()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    pub fn embedder_name(&self) -> &'static str {
        self.embedder.name()
    }

    /// Index a batch of messages. Returns the number of *new* chunks
    /// actually written (existing chunks are skipped).
    pub fn index_messages(
        &mut self,
        messages: &[ChunkInputMessage<'_>],
    ) -> Result<usize, RetrieverError> {
        let chunks = chunk_messages(messages, &self.config.chunk_config);
        let mut added = 0;
        for mut chunk in chunks {
            if self.store.contains(&chunk.id) {
                continue;
            }
            let v = self.embedder.embed(&chunk.content)?;
            chunk.embedding = Some(v.clone());
            self.index
                .upsert(&chunk.id, &v)
                .map_err(|e| RetrieverError::Index(e.to_string()))?;
            self.store.insert(chunk)?;
            added += 1;
        }
        Ok(added)
    }

    /// Run a query against the indexed chunks.
    pub fn retrieve(&self, query: &str) -> Result<RetrievalResult, RetrieverError> {
        let start = std::time::Instant::now();

        if self.index.is_empty() {
            return Ok(RetrievalResult {
                chunks: Vec::new(),
                total_searched: 0,
                latency_ms: start.elapsed().as_millis() as u64,
            });
        }

        let q_vec = self.embedder.embed(query)?;
        // Pull more than top_k so dedup + token budget filtering have room.
        let raw = self.index.search(&q_vec, self.config.top_k * 4);

        let mut out: Vec<ScoredChunk> = Vec::new();
        let mut seen_hashes: HashSet<[u8; 8]> = HashSet::new();
        let mut total_tokens = 0usize;

        for (id, score) in raw {
            if score < self.config.min_score {
                continue;
            }
            let Some(chunk) = self.store.get(&id) else {
                continue;
            };

            if self.config.deduplicate {
                let h = content_hash8(&chunk.content);
                if !seen_hashes.insert(h) {
                    continue;
                }
            }
            if total_tokens + chunk.token_count > self.config.max_retrieved_tokens
                && !out.is_empty()
            {
                break;
            }
            total_tokens += chunk.token_count;
            out.push(ScoredChunk {
                chunk: RetrievedChunk::from(chunk),
                score,
            });
            if out.len() >= self.config.top_k {
                break;
            }
        }

        Ok(RetrievalResult {
            chunks: out,
            total_searched: self.index.len(),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Drop everything (index + store). Mirrors `update_memory(reset=true)`.
    pub fn reset(&mut self) -> Result<(), RetrieverError> {
        self.store.clear()?;
        self.index.clear();
        Ok(())
    }
}

fn content_hash8(s: &str) -> [u8; 8] {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let d = h.finalize();
    let mut out = [0u8; 8];
    out.copy_from_slice(&d[..8]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkInputRole;
    use chrono::Utc;
    use tempfile::tempdir;

    fn cfg_in(dir: &std::path::Path) -> RetrieverConfig {
        RetrieverConfig {
            top_k: 3,
            min_score: 0.0,
            max_retrieved_tokens: 2000,
            deduplicate: true,
            storage_path: dir.join("chunks.jsonl"),
            chunk_config: ChunkConfig::default(),
        }
    }

    fn msg<'a>(idx: usize, role: ChunkInputRole, content: &'a str) -> ChunkInputMessage<'a> {
        ChunkInputMessage {
            index: idx,
            role,
            content,
            timestamp: Utc::now(),
            session_id: None,
        }
    }

    #[test]
    fn empty_retriever_returns_empty() {
        let dir = tempdir().unwrap();
        let r = Retriever::open(cfg_in(dir.path())).unwrap();
        let result = r.retrieve("anything").unwrap();
        assert!(result.chunks.is_empty());
        assert_eq!(result.total_searched, 0);
    }

    #[test]
    fn index_then_retrieve_finds_relevant_chunk() {
        let dir = tempdir().unwrap();
        let mut r = Retriever::open(cfg_in(dir.path())).unwrap();
        let msgs = vec![
            msg(0, ChunkInputRole::User, "What's a good Italian restaurant?"),
            msg(
                1,
                ChunkInputRole::Assistant,
                "Try Chez Luigi on Rue de Passy. Best carbonara in Paris.",
            ),
            msg(2, ChunkInputRole::User, "What about Japanese food?"),
            msg(
                3,
                ChunkInputRole::Assistant,
                "Kinugawa is excellent for sushi near Place Vendome.",
            ),
        ];
        let added = r.index_messages(&msgs).unwrap();
        assert!(added > 0, "expected to index something, got {}", added);

        // Query containing words from the *answer* — what the LOCOMO open-ended
        // failure mode looks like in practice. The user asks "what was that
        // restaurant on Rue de Passy?", they don't ask "Italian restaurant"
        // again. HashEmbedder is a keyword retriever and finds the right turn
        // when the query vocabulary overlaps the source.
        let result = r.retrieve("Luigi carbonara Rue de Passy").unwrap();
        assert!(!result.chunks.is_empty(), "no results for keyword query");
        let top = &result.chunks[0];
        assert!(
            top.chunk.content.to_lowercase().contains("luigi"),
            "expected top chunk to mention Luigi, got: {}",
            top.chunk.content
        );
        // Also: the Japanese chunk must NOT be in the top-1.
        assert!(
            !top.chunk.content.to_lowercase().contains("kinugawa"),
            "wrong topic in top-1: {}",
            top.chunk.content
        );
    }

    #[test]
    fn keyword_retriever_limitation_is_documented() {
        // This test pins the *known limitation* of the deterministic
        // HashEmbedder: a query that uses different vocabulary than the
        // answer will not retrieve the answer first. To fix this, build
        // with `--features bert` (or substitute another `Embedder` impl).
        let dir = tempdir().unwrap();
        let mut r = Retriever::open(cfg_in(dir.path())).unwrap();
        let msgs = vec![
            msg(0, ChunkInputRole::User, "What's a good Italian restaurant?"),
            msg(
                1,
                ChunkInputRole::Assistant,
                "Try Chez Luigi. Best carbonara in town.",
            ),
        ];
        r.index_messages(&msgs).unwrap();

        // Pure-vocabulary mismatch: the query talks about "Italian" but the
        // answer doesn't mention that word at all. HashEmbedder ranks the
        // question above the answer because the question itself contains
        // "italian restaurant". This is keyword retrieval working as
        // designed — semantic retrieval would do better here.
        let result = r.retrieve("Italian food recommendation").unwrap();
        let top_is_question = result
            .chunks
            .first()
            .map(|c| c.chunk.content.to_lowercase().contains("italian"))
            .unwrap_or(false);
        assert!(
            top_is_question,
            "expected question (with shared keyword 'italian') as top match"
        );
    }

    #[test]
    fn reindexing_same_messages_is_idempotent() {
        let dir = tempdir().unwrap();
        let mut r = Retriever::open(cfg_in(dir.path())).unwrap();
        let msgs = vec![msg(0, ChunkInputRole::User, "Same content twice please.")];
        let first = r.index_messages(&msgs).unwrap();
        let second = r.index_messages(&msgs).unwrap();
        assert_eq!(first, 1);
        assert_eq!(second, 0);
        assert_eq!(r.store_len(), 1);
        assert_eq!(r.index_len(), 1);
    }

    #[test]
    fn store_persists_across_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            let mut r = Retriever::open(cfg_in(&path)).unwrap();
            r.index_messages(&[msg(
                0,
                ChunkInputRole::User,
                "Persistent chunk content for round-trip test.",
            )])
            .unwrap();
        }
        let r2 = Retriever::open(cfg_in(&path)).unwrap();
        assert_eq!(r2.store_len(), 1);
        assert_eq!(r2.index_len(), 1);

        let result = r2.retrieve("persistent chunk content").unwrap();
        assert!(!result.chunks.is_empty());
    }

    #[test]
    fn token_budget_caps_retrieval() {
        let dir = tempdir().unwrap();
        let mut cfg = cfg_in(dir.path());
        cfg.max_retrieved_tokens = 30;
        cfg.top_k = 10;
        let mut r = Retriever::with_embedder(cfg, Arc::new(HashEmbedder::default())).unwrap();

        // Index several substantial chunks
        let messages: Vec<ChunkInputMessage> = (0..6)
            .map(|i| {
                let s: &'static str = Box::leak(
                    format!(
                        "This is the long sample content number {} with several distinct words to embed.",
                        i
                    )
                    .into_boxed_str(),
                );
                msg(i, ChunkInputRole::User, s)
            })
            .collect();
        r.index_messages(&messages).unwrap();

        let result = r.retrieve("long sample content").unwrap();
        let total: usize = result.chunks.iter().map(|c| c.chunk.token_count).sum();
        // Allow one chunk over budget (we only stop after exceeding).
        assert!(total <= 60, "total = {}", total);
    }
}
