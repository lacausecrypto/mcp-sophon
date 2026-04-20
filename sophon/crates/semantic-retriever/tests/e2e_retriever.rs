//! End-to-end integration tests for the semantic retriever pipeline.
//!
//! These exercise the full chunker → embedder → index → store → query
//! path that the MCP `compress_history` handler wires together, using
//! the default deterministic `HashEmbedder` so they are hermetic and
//! require no network / model download. Unit tests in individual
//! modules cover branch-level behaviour; this file covers the
//! cross-module contracts:
//!
//! - Persistence survives a retriever close/reopen.
//! - Keyword overlap is enough to surface the relevant chunk among
//!   distractors.
//! - Token-budget + min-score filters trim results as configured.
//! - Re-indexing the same messages is idempotent.
//!
//! If any of these invariants break, a working `compress_history`
//! call can silently return the wrong chunks — unit tests on one
//! module can't catch that.

use chrono::Utc;
use semantic_retriever::chunker::{ChunkInputMessage, ChunkInputRole};
use semantic_retriever::{Retriever, RetrieverConfig};
use tempfile::tempdir;

fn msg<'a>(index: usize, role: ChunkInputRole, content: &'a str) -> ChunkInputMessage<'a> {
    ChunkInputMessage {
        index,
        role,
        content,
        timestamp: Utc::now(),
        session_id: None,
    }
}

fn cfg_with_path(path: std::path::PathBuf) -> RetrieverConfig {
    let mut cfg = RetrieverConfig::default();
    cfg.storage_path = path;
    cfg.min_score = 0.0; // deterministic tests — don't filter by score
    cfg
}

#[test]
fn e2e_index_then_retrieve_returns_matching_chunk() {
    let dir = tempdir().unwrap();
    let cfg = cfg_with_path(dir.path().join("chunks.jsonl"));
    let mut retriever = Retriever::open(cfg).expect("open should succeed");

    let msgs = vec![
        msg(
            0,
            ChunkInputRole::User,
            "I wrote a Rust library that compresses prompts for LLM agents.",
        ),
        msg(
            1,
            ChunkInputRole::Assistant,
            "Great — how do you handle cache eviction?",
        ),
        msg(
            2,
            ChunkInputRole::User,
            "Croissants from the bakery near my apartment are legendary.",
        ),
    ];
    let added = retriever
        .index_messages(&msgs)
        .expect("index should succeed");
    assert!(
        added >= 1,
        "at least one chunk should be indexed, got {added}"
    );

    let result = retriever
        .retrieve("how do I compress prompts in Rust?")
        .expect("retrieve should succeed");

    assert!(
        !result.chunks.is_empty(),
        "expected at least one chunk, got zero"
    );
    let top = &result.chunks[0].chunk.content;
    assert!(
        top.contains("Rust") || top.contains("compress") || top.contains("prompts"),
        "top chunk should match query keywords, got: {top:?}"
    );
    // The unrelated "croissants" message should never be rank 1 for this query.
    assert!(
        !result.chunks[0].chunk.content.contains("Croissants"),
        "unrelated chunk ranked #1: {:?}",
        result.chunks[0].chunk.content
    );
}

#[test]
fn e2e_store_persists_across_retriever_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunks.jsonl");

    let added = {
        let mut r = Retriever::open(cfg_with_path(path.clone())).expect("open should succeed");
        let msgs = vec![msg(
            0,
            ChunkInputRole::User,
            "Persistence test: this chunk MUST survive a retriever restart.",
        )];
        r.index_messages(&msgs).expect("index should succeed")
    };
    assert!(added >= 1);

    // Drop + reopen. The JSONL store is the source of truth; the
    // vector index gets rebuilt from it on open.
    let r2 = Retriever::open(cfg_with_path(path)).expect("reopen should succeed");
    assert!(
        r2.store_len() >= 1,
        "store should preserve chunks across reopen"
    );
    assert_eq!(
        r2.store_len(),
        r2.index_len(),
        "index rebuild must cover every stored chunk"
    );

    let result = r2
        .retrieve("persistence test survive restart")
        .expect("retrieve on reopened retriever should succeed");
    assert!(
        !result.chunks.is_empty(),
        "reopened retriever should still surface the seeded chunk"
    );
}

#[test]
fn e2e_reindexing_same_messages_is_idempotent() {
    let dir = tempdir().unwrap();
    let cfg = cfg_with_path(dir.path().join("chunks.jsonl"));
    let mut r = Retriever::open(cfg).expect("open should succeed");

    let msgs = vec![
        msg(0, ChunkInputRole::User, "Idempotent indexing contract."),
        msg(
            1,
            ChunkInputRole::Assistant,
            "Exactly — retries must not double the store.",
        ),
    ];

    let first = r.index_messages(&msgs).expect("first index should succeed");
    let second = r
        .index_messages(&msgs)
        .expect("second index should succeed");
    assert!(first >= 1);
    assert_eq!(
        second, 0,
        "re-indexing identical messages must add zero chunks, got {second}"
    );
    assert_eq!(r.store_len(), r.index_len());
}

#[test]
fn e2e_empty_retriever_returns_empty_result() {
    let dir = tempdir().unwrap();
    let cfg = cfg_with_path(dir.path().join("chunks.jsonl"));
    let r = Retriever::open(cfg).expect("open should succeed");
    let result = r
        .retrieve("anything")
        .expect("retrieve on empty store should succeed");
    assert_eq!(result.chunks.len(), 0);
    assert_eq!(result.total_searched, 0);
}

#[test]
fn e2e_token_budget_caps_retrieved_payload() {
    // Seed many chunks, set a tight `max_retrieved_tokens` budget, and
    // verify the retriever never exceeds it. The exact top-k may be
    // smaller than requested when the budget is binding.
    let dir = tempdir().unwrap();
    let mut cfg = cfg_with_path(dir.path().join("chunks.jsonl"));
    cfg.max_retrieved_tokens = 50;
    cfg.top_k = 10;
    cfg.min_score = 0.0;

    let mut r = Retriever::open(cfg).expect("open should succeed");

    // 20 short messages all containing the query keyword so min_score
    // doesn't trim them; token budget is the only filter under test.
    let bodies: Vec<String> = (0..20)
        .map(|i| format!("Rust compression is a topic — note number {i}."))
        .collect();
    let msgs: Vec<ChunkInputMessage> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| msg(i, ChunkInputRole::User, body.as_str()))
        .collect();
    r.index_messages(&msgs).expect("index should succeed");

    let result = r
        .retrieve("Rust compression topic")
        .expect("retrieve should succeed");
    let total_tokens: usize = result.chunks.iter().map(|c| c.chunk.token_count).sum();
    assert!(
        total_tokens <= 50,
        "retrieved payload exceeded budget: {total_tokens} tokens over cap of 50"
    );
    // Should still produce at least one chunk; the budget is generous
    // enough for that on this corpus.
    assert!(!result.chunks.is_empty(), "at least one chunk expected");
}

#[test]
fn e2e_hash_embedder_is_deterministic_across_reopens() {
    // The deterministic HashEmbedder is a core positioning claim — two
    // Retrievers opened on the same store at different times must
    // rebuild bit-identical vector indexes, otherwise retrieval
    // would drift silently across `sophon serve` restarts.
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunks.jsonl");

    {
        let mut r = Retriever::open(cfg_with_path(path.clone())).expect("open should succeed");
        let msgs = vec![
            msg(
                0,
                ChunkInputRole::User,
                "deterministic embedding anchor one",
            ),
            msg(
                1,
                ChunkInputRole::User,
                "deterministic embedding anchor two",
            ),
            msg(
                2,
                ChunkInputRole::Assistant,
                "totally unrelated pizza topping",
            ),
        ];
        r.index_messages(&msgs).expect("index should succeed");
    }

    // Compare two independent reopens — same query must return same
    // top chunk with the same score.
    let run = |query: &str| -> (String, f32) {
        let r = Retriever::open(cfg_with_path(path.clone())).expect("reopen should succeed");
        let result = r.retrieve(query).expect("retrieve should succeed");
        let top = result.chunks.first().expect("at least one chunk");
        (top.chunk.content.clone(), top.score)
    };

    let q = "deterministic anchor";
    let (c1, s1) = run(q);
    let (c2, s2) = run(q);
    assert_eq!(c1, c2, "deterministic top chunk content drift");
    assert!(
        (s1 - s2).abs() < 1e-6,
        "deterministic top chunk score drift: {s1} vs {s2}"
    );
}
