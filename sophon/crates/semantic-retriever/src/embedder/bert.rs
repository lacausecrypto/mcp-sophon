//! BERT embedder — opt-in semantic backend, gated by `--features bert`.
//!
//! This is currently a stub that documents the integration surface. To
//! finish wiring it up you need to:
//!
//! 1. Add the runtime crates (`candle-core`, `candle-transformers`,
//!    `tokenizers`, `hf-hub`) which the workspace `Cargo.toml` already
//!    declares as optional behind the `bert` feature.
//! 2. Implement the `embed` method below using
//!    `candle_transformers::models::bert::BertModel` and
//!    `tokenizers::Tokenizer`, mean-pooling over the last hidden states
//!    and L2-normalizing the result.
//! 3. The default model is `sentence-transformers/all-MiniLM-L6-v2`
//!    (384 dimensions, ~80 MB), downloaded on first use via `hf-hub`.
//!
//! Until that work is done, instantiating `BertEmbedder::new` returns an
//! error so the failure is loud and obvious — we will *not* silently fall
//! back to the hash embedder, because the user asked for semantic
//! retrieval and deserves to know they got something else.

use std::path::PathBuf;

use super::{Embedder, EmbedderError};

#[derive(Debug)]
pub struct BertEmbedder {
    _model_path: PathBuf,
}

impl BertEmbedder {
    pub fn new(_model_path: impl Into<PathBuf>) -> Result<Self, EmbedderError> {
        Err(EmbedderError::Model(
            "BertEmbedder is a feature-gated stub. \
             Build with `--features bert` AND finish the candle wiring in \
             crates/semantic-retriever/src/embedder/bert.rs before instantiating it. \
             For deterministic retrieval today, use HashEmbedder."
                .to_string(),
        ))
    }
}

impl Embedder for BertEmbedder {
    fn dimension(&self) -> usize {
        384
    }

    fn name(&self) -> &'static str {
        "bert-all-MiniLM-L6-v2"
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        Err(EmbedderError::Model(
            "BertEmbedder::embed is not implemented yet — see file header".into(),
        ))
    }
}
