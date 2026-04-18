//! Ingestion-time graph memory (Path A) — deterministic, query-time
//! zero-LLM alternative to block-based summarisation.
//!
//! Submodules:
//!   - `types`  — Entity, Fact, EntityId, Predicate, FactObject, FactId
//!   - `store`  — (next step) in-memory + JSON-persistable graph storage
//!   - `extract` — (next step) LLM triple extraction
//!   - `query`  — (next step) pure-Rust entity disambiguation + traversal
//!
//! See the root CLAUDE.md / BENCHMARK.md for the design rationale.

pub mod extract;
pub mod ingest;
pub mod query;
pub mod store;
pub mod types;

pub use extract::extract_triples;
pub use ingest::{
    apply_triples, ingest_messages, ingest_messages_batched, IngestReport, DEFAULT_BATCH_SIZE,
};
pub use query::{
    extract_query_entities, query as query_graph, render_facts, resolve_query_entities, ScoredFact,
};
pub use store::{GraphStore, GraphStoreError};
pub use types::{Entity, EntityId, EntityType, Fact, FactId, FactObject, Predicate};
