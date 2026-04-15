//! Linear-scan cosine k-NN index.
//!
//! Why linear scan instead of HNSW: for the chunk counts Sophon realistically
//! sees (single-digit thousands per session), a linear scan over 256-dim
//! L2-normalized vectors takes single-digit milliseconds on CPU. HNSW would
//! save a few ms at the cost of ~50 KB per crate of dependency, lossy recall,
//! and a serialization format we'd have to maintain. Not worth it.
//!
//! When a deployment grows past ~50k chunks the right move is not "add HNSW
//! to this file" but "shard the index by session" — see retriever.rs notes.

use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct VectorIndex {
    /// Parallel arrays: `ids[i]` is the chunk id for the embedding at
    /// `vectors[i * dim .. (i+1) * dim]`. Flat storage avoids per-chunk
    /// allocation overhead for the hot loop.
    ids: Vec<String>,
    vectors: Vec<f32>,
    id_to_pos: HashMap<String, usize>,
    dim: usize,
}

impl VectorIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            ids: Vec::new(),
            vectors: Vec::new(),
            id_to_pos: HashMap::new(),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_to_pos.contains_key(id)
    }

    /// Add or replace a vector for `id`. Replacing is in-place so existing
    /// positions stay stable.
    pub fn upsert(&mut self, id: &str, embedding: &[f32]) -> Result<(), &'static str> {
        if embedding.len() != self.dim {
            return Err("embedding dimension mismatch");
        }
        if let Some(&pos) = self.id_to_pos.get(id) {
            let start = pos * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(embedding);
            return Ok(());
        }
        let pos = self.ids.len();
        self.ids.push(id.to_string());
        self.vectors.extend_from_slice(embedding);
        self.id_to_pos.insert(id.to_string(), pos);
        Ok(())
    }

    /// Search for the top-k chunk ids by cosine similarity. Vectors and
    /// query are assumed L2-normalized — cosine reduces to a dot product,
    /// no square-root in the hot loop.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if self.is_empty() || k == 0 || query.len() != self.dim {
            return Vec::new();
        }

        // Compute all dot products in one pass.
        let n = self.ids.len();
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * self.dim;
            let mut s = 0.0f32;
            // Manual loop unrolling helps a tiny bit on CPU; stick to the
            // straightforward version for clarity.
            for d in 0..self.dim {
                s += query[d] * self.vectors[start + d];
            }
            scored.push((i, s));
        }

        // Partial sort: select the top-k by descending score.
        let k = k.min(n);
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(i, s)| (self.ids[i].clone(), s))
            .collect()
    }

    /// Drop every vector. Used when the underlying store is reset.
    pub fn clear(&mut self) {
        self.ids.clear();
        self.vectors.clear();
        self.id_to_pos.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n == 0.0 {
            v
        } else {
            v.into_iter().map(|x| x / n).collect()
        }
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = VectorIndex::new(4);
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn upsert_and_search() {
        let mut idx = VectorIndex::new(4);
        idx.upsert("a", &unit(vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        idx.upsert("b", &unit(vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        idx.upsert("c", &unit(vec![0.7, 0.7, 0.0, 0.0])).unwrap();

        let q = unit(vec![1.0, 0.1, 0.0, 0.0]);
        let results = idx.search(&q, 2);
        assert_eq!(results.len(), 2);
        // 'a' should be the closest, 'c' second
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn upsert_replaces_in_place() {
        let mut idx = VectorIndex::new(4);
        idx.upsert("x", &unit(vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        let initial_len = idx.len();
        idx.upsert("x", &unit(vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        assert_eq!(idx.len(), initial_len);
        let r = idx.search(&unit(vec![0.0, 1.0, 0.0, 0.0]), 1);
        assert_eq!(r[0].0, "x");
        assert!(r[0].1 > 0.99);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut idx = VectorIndex::new(4);
        assert!(idx.upsert("z", &[1.0, 0.0]).is_err());
    }
}
