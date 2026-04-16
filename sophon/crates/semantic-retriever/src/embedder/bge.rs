//! BGE-small-en-v1.5 embedder backed by `fastembed` (ONNX runtime).
//!
//! Produces 384-dimensional L2-normalized vectors with genuine
//! semantic understanding. The ONNX model (~33 MB) is downloaded on
//! first use to `~/.cache/fastembed/` and reused afterwards.
//!
//! Gated behind the `bge` Cargo feature.

#![cfg(feature = "bge")]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::RwLock;

use super::{Embedder, EmbedderError};

pub struct BgeEmbedder {
    model: std::sync::Mutex<TextEmbedding>,
    cache: RwLock<HashMap<u64, Vec<f32>>>,
}

impl BgeEmbedder {
    pub fn new() -> Result<Self, EmbedderError> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )
        .map_err(|e| EmbedderError::Model(format!("failed to load BGE-small: {e}")))?;

        Ok(Self {
            model: std::sync::Mutex::new(model),
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn hash_text(text: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }
}

impl Embedder for BgeEmbedder {
    fn dimension(&self) -> usize {
        384
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        if text.is_empty() {
            return Err(EmbedderError::EmptyInput);
        }

        let h = Self::hash_text(text);
        if let Some(cached) = self.cache.read().get(&h) {
            return Ok(cached.clone());
        }

        let mut model = self
            .model
            .lock()
            .map_err(|e| EmbedderError::Model(format!("lock poisoned: {e}")))?;
        let embeddings = model
            .embed(vec![text.to_string()], None)
            .map_err(|e| EmbedderError::Model(format!("embed failed: {e}")))?;

        let mut vec = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Model("no embedding returned".into()))?;

        Self::l2_normalize(&mut vec);
        self.cache.write().insert(h, vec.clone());
        Ok(vec)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let mut model = self
            .model
            .lock()
            .map_err(|e| EmbedderError::Model(format!("lock poisoned: {e}")))?;
        let mut results = model
            .embed(owned, None)
            .map_err(|e| EmbedderError::Model(format!("batch embed failed: {e}")))?;

        for v in &mut results {
            Self::l2_normalize(v);
        }
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "bge-small-en-v1.5"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_bge() -> Option<BgeEmbedder> {
        BgeEmbedder::new().ok()
    }

    #[test]
    #[ignore = "requires ONNX model download (~33 MB). Run with: cargo test -p semantic-retriever --features bge -- --ignored"]
    fn bge_produces_384_dim() {
        let emb = try_bge().expect("BGE model not available");
        let v = emb.embed("hello world").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    #[ignore = "requires ONNX model download"]
    fn bge_is_normalized() {
        let emb = try_bge().expect("BGE model not available");
        let v = emb.embed("test normalization").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}, expected ~1.0");
    }

    #[test]
    #[ignore = "requires ONNX model download"]
    fn bge_semantic_similarity() {
        let emb = try_bge().expect("BGE model not available");
        let restaurant = emb.embed("italian restaurant in downtown").unwrap();
        let pizzeria = emb.embed("neapolitan pizzeria nearby").unwrap();
        let garage = emb.embed("automobile repair shop").unwrap();

        let sim_related: f32 = restaurant.iter().zip(&pizzeria).map(|(a, b)| a * b).sum();
        let sim_unrelated: f32 = restaurant.iter().zip(&garage).map(|(a, b)| a * b).sum();

        assert!(
            sim_related > sim_unrelated,
            "related={sim_related:.3} should be > unrelated={sim_unrelated:.3}"
        );
        assert!(sim_related > 0.5, "related={sim_related:.3} should be > 0.5");
    }

    #[test]
    #[ignore = "requires ONNX model download"]
    fn bge_caches() {
        let emb = try_bge().expect("BGE model not available");
        let v1 = emb.embed("cache test").unwrap();
        let v2 = emb.embed("cache test").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    #[ignore = "requires ONNX model download"]
    fn bge_empty_is_error() {
        let emb = try_bge().expect("BGE model not available");
        assert!(emb.embed("").is_err());
    }
}
