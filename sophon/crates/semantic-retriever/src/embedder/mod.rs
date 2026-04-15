//! Embedder abstraction.
//!
//! All retrievers go through this trait so that swapping the deterministic
//! default for an ML-backed implementation is a one-line change.

pub mod hash;
pub use hash::HashEmbedder;

#[cfg(feature = "bert")]
pub mod bert;
#[cfg(feature = "bert")]
pub use bert::BertEmbedder;

/// Errors emitted by [`Embedder`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("embedding model error: {0}")]
    Model(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("input was empty")]
    EmptyInput,
}

/// Generates dense vector representations of text.
///
/// Implementations must be **deterministic for a given input** within a
/// single process, and must produce vectors of the same `dimension()` for
/// every call. Vectors are L2-normalized so that cosine similarity reduces
/// to a dot product.
pub trait Embedder: Send + Sync {
    /// Output vector dimension.
    fn dimension(&self) -> usize;

    /// Embed a single text into a normalized vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;

    /// Embed multiple texts. Default impl just calls `embed` in a loop;
    /// ML backends should override for batch efficiency.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Short identifier used in logs and the chunk store metadata.
    fn name(&self) -> &'static str;
}
