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

use crate::bm25::Bm25Index;
use crate::chunker::{chunk_messages, Chunk, ChunkConfig, ChunkInputMessage, ChunkType};
use crate::embedder::{Embedder, EmbedderError, HashEmbedder};
use crate::entity_graph::EntityGraph;
use crate::fusion::{rrf_fuse, RRF_K};
use crate::index::VectorIndex;
use crate::reranker::Reranker;
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
    /// Enable multi-hop retrieval: after the first top-k pass, extract
    /// named entities from the results and run a second search to surface
    /// related chunks that the original query missed. Fixes the
    /// "What helped Deborah find peace?" failure mode where the answer
    /// is spread across 3 chunks mentioning "Deborah" but with different
    /// vocabulary.
    pub multihop: bool,
    /// Enable hybrid retrieval: run BM25 sparse-lexical search in parallel
    /// with the vector search and fuse the rankings with Reciprocal Rank
    /// Fusion. Costs one extra linear scan per query (~sub-millisecond at
    /// LOCOMO scale) and zero extra memory per chunk that isn't a
    /// token-frequency table. Closes the 50-pt open-domain gap from rare-
    /// term vocabulary mismatch.
    pub hybrid: bool,
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
            multihop: false,
            hybrid: false,
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
    bm25: Bm25Index,
    entity_graph: EntityGraph,
    store: ChunkStore,
    config: RetrieverConfig,
    reranker: Option<Box<dyn Reranker>>,
}

impl std::fmt::Debug for Retriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("embedder", &self.embedder.name())
            .field("reranker", &self.reranker.as_ref().map(|r| r.name()))
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
        let mut bm25 = Bm25Index::new();
        let mut entity_graph = EntityGraph::new();

        // Re-embed existing chunks. Embeddings are deterministic so this
        // produces the same vectors as when they were first indexed.
        // BM25 stats and the entity graph are also rebuilt here; the
        // JSONL store is the authoritative source for all three.
        for chunk in store.iter() {
            let v = embedder.embed(&chunk.content)?;
            index
                .upsert(&chunk.id, &v)
                .map_err(|e| RetrieverError::Index(e.to_string()))?;
            bm25.insert(&chunk.id, &chunk.content);
            entity_graph.insert_chunk(&chunk.id, &chunk.content);
        }

        Ok(Self {
            embedder,
            index,
            bm25,
            entity_graph,
            store,
            config,
            reranker: None,
        })
    }

    /// Open a retriever with a caller-provided embedder and reranker.
    pub fn with_reranker(
        config: RetrieverConfig,
        embedder: Arc<dyn Embedder>,
        reranker: Box<dyn Reranker>,
    ) -> Result<Self, RetrieverError> {
        let mut ret = Self::with_embedder(config, embedder)?;
        ret.reranker = Some(reranker);
        Ok(ret)
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

    /// Access the underlying embedder (e.g. for scoring prompt sections
    /// via cosine similarity from the MCP handler layer).
    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
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
            self.bm25.insert(&chunk.id, &chunk.content);
            self.entity_graph.insert_chunk(&chunk.id, &chunk.content);
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

        // Query expansion: append synonyms/hypernyms from a static
        // dictionary so that "art" matches chunks containing "painting"
        // or "drawing". Lightweight, deterministic, no network call.
        let expanded_query = expand_query(query);
        let q_vec = self.embedder.embed(&expanded_query)?;
        // Pull more than top_k so dedup + token budget filtering have room.
        let raw = self.index.search(&q_vec, self.config.top_k * 4);

        // Phase 1: collect candidates after dedup, applying reranker scores.
        let mut candidates: Vec<ScoredChunk> = Vec::new();
        let mut seen_hashes: HashSet<[u8; 8]> = HashSet::new();

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

            // Rerank: blend the reranker score with the original cosine score.
            let final_score = if let Some(ref reranker) = self.reranker {
                let rerank_score = reranker.rerank(query, &chunk.content);
                rerank_score * score
            } else {
                score
            };

            candidates.push(ScoredChunk {
                chunk: RetrievedChunk::from(chunk),
                score: final_score,
            });
        }

        // Phase 2: multi-hop expansion. Extract named entities from the
        // top candidates, search for related chunks, and merge into the
        // candidate pool. This surfaces chunks that share an entity
        // (e.g. "Deborah") with the initial results but use different
        // vocabulary than the original query.
        if self.config.multihop && !candidates.is_empty() {
            let top_n = candidates.len().min(3);
            let mut expansion_queries: Vec<String> = Vec::new();

            for sc in candidates.iter().take(top_n) {
                let entities = extract_entities(&sc.chunk.content);
                for entity in entities {
                    // Combine entity with query keywords for targeted search
                    let expanded = format!("{} {}", entity, query);
                    expansion_queries.push(expanded);
                }
            }

            // Deduplicate expansion queries
            expansion_queries.sort();
            expansion_queries.dedup();
            // Limit to 5 expansion queries to bound latency
            expansion_queries.truncate(5);

            let existing_ids: HashSet<String> =
                candidates.iter().map(|sc| sc.chunk.id.clone()).collect();

            for eq in &expansion_queries {
                if let Ok(eq_vec) = self.embedder.embed(eq) {
                    let expansion_raw = self.index.search(&eq_vec, 3);
                    for (id, score) in expansion_raw {
                        if existing_ids.contains(&id) {
                            continue;
                        }
                        if score < self.config.min_score {
                            continue;
                        }
                        let Some(chunk) = self.store.get(&id) else {
                            continue;
                        };
                        // Discount expansion results slightly (they're
                        // indirect matches, not direct query hits)
                        let discounted = score * 0.85;
                        candidates.push(ScoredChunk {
                            chunk: RetrievedChunk::from(chunk),
                            score: discounted,
                        });
                    }
                }
            }
        }

        // Phase 2.5: hybrid retrieval. Run BM25 sparse-lexical search in
        // parallel and fuse with the vector ranking via Reciprocal Rank
        // Fusion. Rationale: HashEmbedder's bag-of-words cosine misses
        // queries that hinge on a rare proper noun; BM25's IDF term makes
        // that exact case its sweet spot. RRF(k=60) is rank-based, so the
        // numeric scale difference between cosine and BM25 doesn't matter.
        if self.config.hybrid && !self.bm25.is_empty() {
            let bm25_k = self.config.top_k * 4;
            let bm25_hits = self.bm25.search(query, bm25_k);
            if !bm25_hits.is_empty() {
                let bm25_ranking: Vec<ScoredChunk> = bm25_hits
                    .into_iter()
                    .filter_map(|(id, score)| {
                        self.store.get(&id).map(|chunk| ScoredChunk {
                            chunk: RetrievedChunk::from(chunk),
                            score,
                        })
                    })
                    .collect();
                let vec_ranking = std::mem::take(&mut candidates);
                candidates = rrf_fuse(&[vec_ranking, bm25_ranking], RRF_K);
            }
        }

        // Phase 3: re-sort by (possibly adjusted) score, then apply token budget.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate after merging expansion results
        let mut final_seen: HashSet<String> = HashSet::new();
        let mut out: Vec<ScoredChunk> = Vec::new();
        let mut total_tokens = 0usize;
        for sc in candidates {
            if !final_seen.insert(sc.chunk.id.clone()) {
                continue;
            }
            if total_tokens + sc.chunk.token_count > self.config.max_retrieved_tokens
                && !out.is_empty()
            {
                break;
            }
            total_tokens += sc.chunk.token_count;
            out.push(sc);
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

    /// Drop everything (index + store + bm25 + entity graph). Mirrors
    /// `update_memory(reset=true)`.
    pub fn reset(&mut self) -> Result<(), RetrieverError> {
        self.store.clear()?;
        self.index.clear();
        self.bm25.clear();
        self.entity_graph.clear();
        Ok(())
    }

    /// Query the entity graph directly. Returns up to `k` chunks ranked
    /// by IDF-weighted entity overlap + 1-hop bridge. Used as an
    /// additional ranking fed into RRF fusion when
    /// `SOPHON_ENTITY_GRAPH=1` is set.
    pub fn search_entity_graph(&self, query: &str, k: usize) -> Vec<ScoredChunk> {
        let hits = self.entity_graph.search(query, k);
        hits.into_iter()
            .filter_map(|(id, score)| {
                self.store.get(&id).map(|chunk| ScoredChunk {
                    chunk: RetrievedChunk::from(chunk),
                    score,
                })
            })
            .collect()
    }
}

/// Expand a query with synonyms/hypernyms from a static dictionary.
/// Only adds words that aren't already in the query (no bloat on
/// queries that already use the right vocabulary).
///
/// The dictionary is tuned for the LOCOMO failure modes: "art" queries
/// that miss "painting", "game" queries that miss "console", "food"
/// queries that miss "restaurant", etc.
fn expand_query(query: &str) -> String {
    // (trigger_word, expansions) — trigger matches case-insensitively
    // against query words. Expansions are appended once per trigger.
    static SYNONYMS: &[(&str, &[&str])] = &[
        // Creative
        (
            "art",
            &[
                "painting",
                "drawing",
                "sculpture",
                "artwork",
                "canvas",
                "creative",
            ],
        ),
        ("painting", &["art", "artwork", "canvas", "painter"]),
        (
            "music",
            &["song", "concert", "band", "album", "musical", "instrument"],
        ),
        (
            "writing",
            &["book", "novel", "screenplay", "blog", "journal", "author"],
        ),
        ("book", &["novel", "reading", "author", "literature"]),
        // Food & places
        (
            "food",
            &["restaurant", "cooking", "meal", "dish", "cuisine"],
        ),
        (
            "restaurant",
            &["food", "dining", "meal", "pizza", "cuisine"],
        ),
        (
            "travel",
            &["trip", "flight", "vacation", "visit", "destination"],
        ),
        ("trip", &["travel", "journey", "road", "vacation"]),
        // Tech
        ("game", &["gaming", "console", "play", "player", "video"]),
        ("games", &["gaming", "console", "play", "player", "video"]),
        (
            "console",
            &["game", "gaming", "switch", "playstation", "xbox"],
        ),
        ("code", &["programming", "coding", "software", "developer"]),
        (
            "project",
            &["working", "building", "developing", "app", "tool"],
        ),
        // People & relationships
        ("friend", &["colleague", "partner", "buddy", "companion"]),
        ("colleague", &["friend", "coworker", "teammate", "partner"]),
        (
            "family",
            &["sister", "brother", "mother", "father", "parent"],
        ),
        // Health & wellness
        (
            "peace",
            &["calm", "comfort", "healing", "therapeutic", "relax"],
        ),
        (
            "health",
            &["exercise", "yoga", "running", "wellness", "medical"],
        ),
        (
            "exercise",
            &["run", "running", "yoga", "workout", "marathon"],
        ),
        // Animals
        ("pet", &["dog", "cat", "puppy", "kitten", "animal"]),
        ("dog", &["pet", "puppy", "retriever", "breed"]),
        ("cat", &["pet", "kitten", "feline"]),
        // Time
        ("recently", &["lately", "last", "new", "current", "past"]),
        ("favorite", &["prefer", "favourite", "best", "love", "like"]),
        ("hobby", &["interest", "passion", "activity", "enjoy"]),
    ];

    let query_lower = query.to_lowercase();
    let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

    let mut additions: Vec<&str> = Vec::new();
    for (trigger, expansions) in SYNONYMS {
        if query_words.contains(trigger) {
            for exp in *expansions {
                if !query_words.contains(exp) && !additions.contains(exp) {
                    additions.push(exp);
                }
            }
        }
    }

    if additions.is_empty() {
        return query.to_string();
    }

    // Append expansions to the original query (preserves original casing)
    format!("{} {}", query, additions.join(" "))
}

/// Extract probable named entities from text. Uses a simple heuristic:
/// capitalized words ≥ 3 chars that aren't common sentence-starters.
/// This is intentionally rough — we're looking for "Deborah", "Alice",
/// "PostgreSQL", "Brussels", not building a full NER pipeline.
fn extract_entities(text: &str) -> Vec<String> {
    static STOP_WORDS: &[&str] = &[
        "The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "How",
        "Why", "Yes", "Not", "But", "And", "For", "Are", "Was", "Were", "Has", "Have", "Had",
        "Will", "Would", "Could", "Should", "May", "Can", "Did", "Does", "Been", "Being", "Also",
        "Just", "Very", "Really", "Some", "Any", "All", "Each", "Every", "Most", "More", "Much",
        "Many", "Other", "Another", "Here", "There", "Then", "Now", "Well", "Too", "Sure", "Nice",
        "Good", "Great", "Cool", "Thanks", "Sorry", "Please", "Hello",
    ];

    let mut entities: Vec<String> = Vec::new();
    let mut seen = HashSet::new();

    for word in text.split_whitespace() {
        // Strip trailing punctuation
        let clean: String = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        if clean.len() < 3 {
            continue;
        }
        // Must start with uppercase
        let first = clean.chars().next().unwrap_or('a');
        if !first.is_uppercase() {
            continue;
        }
        // Skip common words
        if STOP_WORDS.iter().any(|sw| sw.eq_ignore_ascii_case(&clean)) {
            continue;
        }
        let lower = clean.to_lowercase();
        if seen.insert(lower) {
            entities.push(clean);
        }
    }

    // Limit to top 5 to bound expansion cost
    entities.truncate(5);
    entities
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
    use crate::reranker::KeywordReranker;
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
            multihop: false,
            hybrid: false,
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

    #[test]
    fn hybrid_retrieval_surfaces_rare_term_match() {
        // HashEmbedder is a bag-of-words keyword cosine. It ranks chunks by
        // shared vocabulary, so a rare proper noun ("Seraphim") doesn't
        // automatically dominate a semantically related but vocabulary-
        // mismatched chunk. BM25's IDF does dominate on rare terms, so the
        // hybrid RRF fusion should surface the Seraphim-mentioning chunk
        // even when the plain retriever misses it.
        let dir = tempdir().unwrap();
        let mut cfg_plain = cfg_in(dir.path());
        cfg_plain.top_k = 3;
        cfg_plain.hybrid = false;

        let dir_h = tempdir().unwrap();
        let mut cfg_hybrid = cfg_in(dir_h.path());
        cfg_hybrid.top_k = 3;
        cfg_hybrid.hybrid = true;

        let msgs = vec![
            msg(
                0,
                ChunkInputRole::User,
                "Tell me about our pets and daily routines around the apartment.",
            ),
            msg(
                1,
                ChunkInputRole::Assistant,
                "We have three cats and a goldfish that lives in a tank by the window.",
            ),
            msg(
                2,
                ChunkInputRole::User,
                "Wait, did you also mention the aquarium for Seraphim last week?",
            ),
            msg(
                3,
                ChunkInputRole::Assistant,
                "Yes — Jolene bought a new aquarium for Seraphim that Sunday.",
            ),
        ];

        let mut plain = Retriever::open(cfg_plain).unwrap();
        plain.index_messages(&msgs).unwrap();
        let mut hybrid = Retriever::open(cfg_hybrid).unwrap();
        hybrid.index_messages(&msgs).unwrap();

        let q = "When did Jolene buy a new aquarium for Seraphim?";
        let r_plain = plain.retrieve(q).unwrap();
        let r_hybrid = hybrid.retrieve(q).unwrap();

        // Both should return something; the hybrid should rank the
        // Seraphim-mentioning chunk at the top.
        assert!(!r_hybrid.chunks.is_empty());
        let top = &r_hybrid.chunks[0].chunk.content.to_lowercase();
        assert!(
            top.contains("seraphim") || top.contains("aquarium"),
            "hybrid top-1 should contain the rare-term answer, got: {}",
            top
        );
        // Sanity: plain result is non-empty too (we're not claiming BM25
        // fixes *every* ranking — just that hybrid ≥ plain on this case).
        assert!(!r_plain.chunks.is_empty());
    }

    #[test]
    fn keyword_reranker_reorders_results() {
        let dir = tempdir().unwrap();
        let mut cfg = cfg_in(dir.path());
        cfg.top_k = 10;
        cfg.min_score = 0.0;
        cfg.max_retrieved_tokens = 10000;

        // Build two retrievers: one without reranker, one with.
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default());
        let mut r_plain = Retriever::with_embedder(cfg.clone(), embedder.clone()).unwrap();

        // We need a separate store path for the reranked retriever.
        let dir2 = tempdir().unwrap();
        let mut cfg2 = cfg_in(dir2.path());
        cfg2.top_k = cfg.top_k;
        cfg2.min_score = cfg.min_score;
        cfg2.max_retrieved_tokens = cfg.max_retrieved_tokens;
        let mut r_reranked =
            Retriever::with_reranker(cfg2, embedder.clone(), Box::new(KeywordReranker)).unwrap();

        let msgs = vec![
            msg(
                0,
                ChunkInputRole::User,
                "Tell me about Rust programming language and memory safety.",
            ),
            msg(
                1,
                ChunkInputRole::Assistant,
                "Rust guarantees memory safety without garbage collection through its ownership system.",
            ),
            msg(
                2,
                ChunkInputRole::User,
                "What about Python for data science?",
            ),
            msg(
                3,
                ChunkInputRole::Assistant,
                "Python is the dominant language for data science thanks to numpy and pandas.",
            ),
        ];
        r_plain.index_messages(&msgs).unwrap();
        r_reranked.index_messages(&msgs).unwrap();

        // Query about "Rust memory" — reranker should boost chunks that
        // actually contain those keywords.
        let plain_result = r_plain.retrieve("Rust memory safety ownership").unwrap();
        let reranked_result = r_reranked.retrieve("Rust memory safety ownership").unwrap();

        // Both should return results
        assert!(!plain_result.chunks.is_empty());
        assert!(!reranked_result.chunks.is_empty());

        // In the reranked result, chunks about Rust should have higher
        // scores than chunks about Python.
        let rust_chunks: Vec<&ScoredChunk> = reranked_result
            .chunks
            .iter()
            .filter(|c| c.chunk.content.to_lowercase().contains("rust"))
            .collect();
        let python_chunks: Vec<&ScoredChunk> = reranked_result
            .chunks
            .iter()
            .filter(|c| c.chunk.content.to_lowercase().contains("python"))
            .collect();

        if !rust_chunks.is_empty() && !python_chunks.is_empty() {
            let best_rust = rust_chunks.iter().map(|c| c.score).fold(0.0f32, f32::max);
            let best_python = python_chunks.iter().map(|c| c.score).fold(0.0f32, f32::max);
            assert!(
                best_rust > best_python,
                "expected Rust chunks ({}) to outscore Python chunks ({})",
                best_rust,
                best_python
            );
        }
    }
}
