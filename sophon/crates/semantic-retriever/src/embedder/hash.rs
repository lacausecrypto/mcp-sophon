//! Deterministic hash-based embedder — Sophon's default backend.
//!
//! Approach: feature hashing (the "hashing trick" of Weinberger et al.) over
//! a mix of word unigrams and character 3-grams. For each token we compute a
//! 64-bit hash, project it into a fixed bucket, and accumulate a TF count.
//! The resulting vector is L2-normalized so cosine similarity = dot product.
//!
//! This is **not semantic** — it cannot match synonyms. But it is:
//! - deterministic across runs and platforms
//! - ML-free, model-free, network-free
//! - single-digit microseconds per query, ~zero memory overhead
//! - good enough to recover named entities, dates, code identifiers, and
//!   any query that shares vocabulary with the source — which covers the
//!   bulk of LOCOMO-style "what did X say about Y?" questions.
//!
//! For true semantic retrieval (synonyms, paraphrase, multilingual), build
//! with `--features bert` to use [`super::BertEmbedder`].

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{Embedder, EmbedderError};

/// Default vector dimension. 256 buckets is a good trade-off: dense enough
/// to give meaningful cosine values on conversational chunks, small enough
/// that linear-scan k-NN over 10k chunks stays under 10 ms.
pub const DEFAULT_DIMENSION: usize = 256;

/// Deterministic hashing-trick embedder.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
    /// Whether to include character 3-grams in addition to word unigrams.
    /// Adds robustness to typos and morphological variants at the cost of
    /// a ~4× increase in token count per text.
    use_char_ngrams: bool,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self {
            dim: DEFAULT_DIMENSION,
            use_char_ngrams: true,
        }
    }
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim, use_char_ngrams: true }
    }

    pub fn with_char_ngrams(mut self, enabled: bool) -> Self {
        self.use_char_ngrams = enabled;
        self
    }

    /// Tokenize text into the units we hash.
    fn tokens(&self, text: &str) -> Vec<String> {
        let lowered = text.to_lowercase();
        let mut out: Vec<String> = lowered
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect();

        if self.use_char_ngrams {
            // Character 3-grams, computed over the un-tokenized text so that
            // entity boundaries (e.g. "Chez Luigi") still produce useful
            // overlap features.
            let chars: Vec<char> = lowered.chars().filter(|c| !c.is_whitespace()).collect();
            for window in chars.windows(3) {
                out.push(window.iter().collect::<String>());
            }
        }

        out
    }

    fn bucket(&self, token: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        (hasher.finish() as usize) % self.dim
    }
}

impl Embedder for HashEmbedder {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "hash-256"
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        if text.trim().is_empty() {
            return Err(EmbedderError::EmptyInput);
        }

        let mut vec = vec![0.0f32; self.dim];
        for token in self.tokens(text) {
            let idx = self.bucket(&token);
            // Sub-linear TF damping mirrors what BM25 does: a word that appears
            // 10 times is more important than one that appears once, but not
            // 10 times more important.
            vec[idx] += 1.0;
        }

        // Apply log dampening + L2 normalize.
        for v in vec.iter_mut() {
            if *v > 0.0 {
                *v = (*v + 1.0).ln();
            }
        }
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }

        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        let e = HashEmbedder::default();
        let a = e.embed("Alice recommended Chez Luigi").unwrap();
        let b = e.embed("Alice recommended Chez Luigi").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn dimension_consistency() {
        let e = HashEmbedder::default();
        let a = e.embed("hi").unwrap();
        let b = e.embed("a much longer piece of text with many tokens").unwrap();
        assert_eq!(a.len(), e.dimension());
        assert_eq!(b.len(), e.dimension());
    }

    #[test]
    fn normalized() {
        let e = HashEmbedder::default();
        let v = e.embed("some text here").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[test]
    fn shared_vocabulary_higher_similarity() {
        let e = HashEmbedder::default();
        let q = e.embed("what restaurant did Alice recommend").unwrap();
        let close = e.embed("Alice recommended a restaurant").unwrap();
        let far = e.embed("the weather forecast for tomorrow").unwrap();

        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let s_close = dot(&q, &close);
        let s_far = dot(&q, &far);
        assert!(s_close > s_far, "close={} far={}", s_close, s_far);
    }

    #[test]
    fn empty_input_rejected() {
        let e = HashEmbedder::default();
        assert!(matches!(e.embed("   "), Err(EmbedderError::EmptyInput)));
    }
}
