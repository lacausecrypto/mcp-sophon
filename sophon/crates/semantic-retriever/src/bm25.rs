//! Okapi BM25 sparse-lexical scorer — a zero-ML complement to the vector
//! retriever.
//!
//! BM25 is the de facto baseline in IR: term frequency × inverse document
//! frequency, with document-length normalisation. It complements dense/hash
//! retrieval in three specific ways relevant to Sophon's LOCOMO gap:
//!
//! - **Rare-term sensitivity**: a query mentioning a specific proper noun
//!   (e.g. "Seraphim") is dominated by that term's IDF, not by vocabulary
//!   overlap — which is where HashEmbedder's keyword bag-of-words underperforms.
//! - **Length normalisation**: short chunks that happen to share one query
//!   token don't over-rank; longer chunks with more hits get credit
//!   proportional to effective information density.
//! - **Sparse-complement to dense**: SOTA hybrid retrieval (BM25 + vector,
//!   fused via RRF) consistently outperforms either alone on
//!   out-of-vocabulary queries.
//!
//! Implementation:
//! - Tokenisation: lowercase, strip non-alphanumeric boundaries, split on
//!   whitespace. Same tokeniser used for index and query. Stop words are NOT
//!   filtered — BM25's IDF already down-weights them.
//! - `k1 = 1.2`, `b = 0.75` (standard Robertson/Jones defaults).
//! - Incremental indexing: `insert(id, text)` keeps stats in sync without
//!   requiring a rebuild. Idempotent on duplicate ids (first wins).

use std::collections::HashMap;

/// Robertson et al. default — controls term-frequency saturation. Higher k1
/// means repeated terms keep adding weight longer before plateauing.
pub const BM25_K1: f32 = 1.2;

/// Document length normalisation strength. 0 = no length correction,
/// 1 = fully normalised by ratio to average length.
pub const BM25_B: f32 = 0.75;

/// Lowercase + alphanumeric split. Used for both index and query.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[derive(Debug)]
struct DocEntry {
    id: String,
    length: usize,
    tf: HashMap<String, u32>,
}

/// Incremental BM25 index over a set of documents keyed by opaque string id.
#[derive(Debug, Default)]
pub struct Bm25Index {
    docs: Vec<DocEntry>,
    id_to_pos: HashMap<String, usize>,
    df: HashMap<String, u32>,
    total_length: usize,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_to_pos.contains_key(id)
    }

    /// Insert a document. If an id is already present, the call is a no-op —
    /// BM25 stats stay consistent (the first content wins, matching how the
    /// chunk store dedups).
    pub fn insert(&mut self, id: &str, text: &str) {
        if self.id_to_pos.contains_key(id) {
            return;
        }
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }
        for term in tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }
        let length = tokens.len();
        self.total_length += length;
        let pos = self.docs.len();
        self.docs.push(DocEntry {
            id: id.to_string(),
            length,
            tf,
        });
        self.id_to_pos.insert(id.to_string(), pos);
    }

    pub fn clear(&mut self) {
        self.docs.clear();
        self.id_to_pos.clear();
        self.df.clear();
        self.total_length = 0;
    }

    /// Average document length — denominator in the length-normalisation
    /// term. Returns 1.0 when the index is empty to keep the formula safe.
    fn avg_length(&self) -> f32 {
        if self.docs.is_empty() {
            1.0
        } else {
            self.total_length as f32 / self.docs.len() as f32
        }
    }

    /// Top-k docs by BM25 score for `query`, descending. Docs with score ≤ 0
    /// are filtered out (they share no query tokens).
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if self.docs.is_empty() || k == 0 {
            return Vec::new();
        }
        let q_tokens = tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let avgdl = self.avg_length();
        let n = self.docs.len() as f32;

        // Dedupe query tokens — repeats don't add information.
        let mut q_set: Vec<String> = q_tokens.into_iter().collect();
        q_set.sort();
        q_set.dedup();

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (doc_pos, doc) in self.docs.iter().enumerate() {
            let mut s = 0.0f32;
            let dl_norm = doc.length as f32 / avgdl;
            for term in &q_set {
                let Some(&tf) = doc.tf.get(term) else {
                    continue;
                };
                let df = *self.df.get(term).unwrap_or(&0) as f32;
                if df == 0.0 {
                    continue;
                }
                // Robertson-Spärck Jones IDF with +1 smoothing to avoid
                // negative scores on terms present in > half the corpus.
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                let tfn = tf as f32;
                let num = tfn * (BM25_K1 + 1.0);
                let denom = tfn + BM25_K1 * (1.0 - BM25_B + BM25_B * dl_norm);
                s += idf * (num / denom);
            }
            if s > 0.0 {
                scored.push((doc_pos, s));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(pos, s)| (self.docs[pos].id.clone(), s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_strips_punctuation_and_lowers() {
        let t = tokenize("Hello, World! Rust-lang rocks.");
        assert_eq!(t, vec!["hello", "world", "rust", "lang", "rocks"]);
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = Bm25Index::new();
        assert!(idx.search("anything", 5).is_empty());
    }

    #[test]
    fn duplicate_id_is_ignored() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "the quick brown fox");
        idx.insert("a", "totally different text");
        assert_eq!(idx.len(), 1);
        let hits = idx.search("quick", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");
    }

    #[test]
    fn rare_term_outscores_common_term() {
        // "the" appears in every doc — IDF ≈ 0.
        // "seraphim" appears in one doc — high IDF.
        let mut idx = Bm25Index::new();
        idx.insert("d1", "the quick brown fox jumps");
        idx.insert("d2", "the lazy dog sleeps");
        idx.insert("d3", "the cat watches the seraphim");
        idx.insert("d4", "the pen is on the table");

        let hits = idx.search("seraphim the", 4);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, "d3", "d3 should win via rare-term IDF");
    }

    #[test]
    fn length_normalisation_penalises_long_docs_with_one_hit() {
        let mut idx = Bm25Index::new();
        idx.insert("short", "Jolene");
        idx.insert(
            "long",
            "once upon a time there was a very long rambling \
             text that just happens to mention Jolene somewhere in \
             the middle of all this other filler content nobody \
             really cares about but the tokenizer still sees it",
        );
        let hits = idx.search("Jolene", 2);
        assert_eq!(hits[0].0, "short", "short exact-match should beat long diluted");
    }

    #[test]
    fn zero_query_token_overlap_returns_nothing() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "apples bananas cherries");
        idx.insert("b", "dogs elephants frogs");
        let hits = idx.search("zebra", 3);
        assert!(hits.is_empty());
    }

    #[test]
    fn clear_resets_all_stats() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "hello world");
        idx.clear();
        assert!(idx.is_empty());
        assert!(!idx.contains("a"));
        assert!(idx.search("hello", 5).is_empty());
    }

    #[test]
    fn top_k_respects_limit() {
        let mut idx = Bm25Index::new();
        for i in 0..10 {
            idx.insert(&format!("d{i}"), "shared common word");
        }
        let hits = idx.search("word", 3);
        assert_eq!(hits.len(), 3);
    }
}
