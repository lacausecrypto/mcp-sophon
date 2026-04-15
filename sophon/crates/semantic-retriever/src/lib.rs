//! Sophon semantic retriever — adds query-time chunk retrieval to the
//! conversation memory pipeline so that recall-heavy questions can find the
//! exact turns where a fact was mentioned.
//!
//! The default build ships a deterministic [`HashEmbedder`](embedder::HashEmbedder)
//! (TF over hashed n-grams, no ML, no model download, no network) and a
//! linear-scan k-NN index. This keeps Sophon's "no runtime ML, ~10 MB binary"
//! positioning intact while still giving meaningful retrieval quality on
//! lexical queries.
//!
//! For semantic embeddings (Sentence-BERT family), build with
//! `--features bert` to pull `candle-transformers` and download
//! `all-MiniLM-L6-v2` on first use. The BERT backend is opt-in by design.
//!
//! See the workspace `BENCHMARK.md` for the LOCOMO open-ended results that
//! motivated this module.

pub mod chunker;
pub mod embedder;
pub mod index;
pub mod retriever;
pub mod store;

pub use chunker::{chunk_messages, Chunk, ChunkConfig, ChunkType};
pub use embedder::{Embedder, HashEmbedder};
pub use index::VectorIndex;
pub use retriever::{
    RetrievalResult, RetrievedChunk, Retriever, RetrieverConfig, RetrieverError, ScoredChunk,
};
pub use store::ChunkStore;
