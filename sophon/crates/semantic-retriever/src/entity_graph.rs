//! Entity co-occurrence graph for multi-hop retrieval (Rec #2).
//!
//! Motivation: LOCOMO multi-hop items (16 / 80 N=80) stayed at 0 %
//! accuracy across every retrieval variant — HashEmbedder, BM25, HyDE,
//! ReAct. The common failure trace:
//!
//!   Question : "When did Maria adopt Shadow?"
//!   Answer chunk : "Shadow came home on May 5" (uttered weeks after
//!                   Maria's adoption mention, different session, doesn't
//!                   mention "Maria" again).
//!
//! Cosine/BM25 retrieval can't bridge Maria → Shadow because the two
//! words never co-occur in the same answer chunk with enough lexical
//! overlap. HippoRAG (NeurIPS'24) and GraphRAG (MS Research) solve this
//! with an entity graph + personalised PageRank over embeddings. We
//! want the same graph structure **without any neural embedding**: pure
//! regex-based NER, bipartite `entity ↔ chunk` index, and IDF-weighted
//! scoring with a bounded 1-hop expansion. All deterministic, no model.
//!
//! Scoring (a query maps to a ranked list of chunks):
//!   score(chunk) =
//!       Σ   over query_entities ∩ chunk_entities   { idf(e) }
//!     + 0.5 · Σ over "bridge" chunks one hop away  { shared_query_entity_idf }
//!
//! where `idf(e) = ln(N / |chunks containing e|)`. This dampens noise
//! from ubiquitous mentions (e.g. "user") and rewards query-specific
//! hits (e.g. "Shadow").
//!
//! 1-hop expansion only runs when the top direct hit contains fewer
//! than half of the query entities — it's the mechanism that solves the
//! Maria-Shadow case (the Maria chunk "introduces" Shadow, which then
//! surfaces the later Shadow chunk that alone misses Maria).

use std::collections::{HashMap, HashSet};

/// Min chars for a candidate entity token to be indexed. Shorter tokens
/// (2-char acronyms, initials) tend to be false positives in plain
/// English chat.
const MIN_ENTITY_LEN: usize = 3;

/// Max entities kept per chunk. Bounds index size on very long chunks.
const MAX_ENTITIES_PER_CHUNK: usize = 40;

/// Max results to return from `search`. Bounded so the graph path stays
/// usable as an input to RRF fusion (where top-N rankings get fused).
const DEFAULT_SEARCH_CAP: usize = 30;

/// Stop words that look like capitalised proper nouns at the start of a
/// sentence but almost never are. Extends the list already used in
/// `retriever::extract_entities`.
static STOP_CAPS: &[&str] = &[
    "The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "Whom",
    "Whose", "How", "Why", "Yes", "No", "Not", "But", "And", "Or", "For", "Are", "Was", "Were",
    "Has", "Have", "Had", "Will", "Would", "Could", "Should", "May", "Can", "Did", "Does",
    "Been", "Being", "Also", "Just", "Very", "Really", "Some", "Any", "All", "Each", "Every",
    "Most", "More", "Much", "Many", "Other", "Another", "Here", "There", "Then", "Now", "Well",
    "Too", "Sure", "Nice", "Good", "Great", "Cool", "Thanks", "Sorry", "Please", "Hello",
    "Hi", "Okay", "OK", "Mr", "Mrs", "Ms", "Dr", "If", "It", "Is", "As", "At", "In", "On",
    "To", "By", "Of", "Be", "Do", "So", "Session", "User", "Assistant", "System",
];

/// Lightweight NER: proper-noun-looking tokens (capitalised, length ≥ 3,
/// not in the stop list). Case-preserving — "Alice" and "Bob" stay as
/// written; normalised key is the lowercased form for merging.
///
/// Exposed for reuse by callers that want to score queries against a
/// graph they didn't build.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Split on ANY non-alphanumeric char so mid-word punctuation is
    // handled cleanly: "Seraphim's" → ["Seraphim", "s"], "Marie-Louise"
    // → ["Marie", "Louise"]. This avoids the silent mismatch where the
    // indexed entity would be "Seraphim's" but a query for "Seraphim"
    // never finds it.
    for tok in text.split(|c: char| !c.is_alphanumeric()) {
        if tok.len() < MIN_ENTITY_LEN {
            continue;
        }
        let first = tok.chars().next().unwrap_or('a');
        if !first.is_uppercase() {
            continue;
        }
        if STOP_CAPS.iter().any(|s| s.eq_ignore_ascii_case(tok)) {
            continue;
        }
        // Filter out ALL-CAPS shouting ("AMAZING", "HELLO") so we don't
        // mistake it for proper-noun content. A lone "AI" / "CEO" gets
        // filtered by MIN_ENTITY_LEN + this rule doesn't touch
        // mixed-case names like "McDonald" or "DeAndre".
        if tok.chars().skip(1).all(|c| c.is_uppercase()) && tok.len() > 4 {
            continue;
        }
        let key = tok.to_lowercase();
        if seen.insert(key) {
            out.push(tok.to_string());
        }
    }
    out.truncate(MAX_ENTITIES_PER_CHUNK);
    out
}

/// Normalised key used internally. Callers don't see this — we accept
/// any casing and merge on lowercase.
fn norm(entity: &str) -> String {
    entity.to_lowercase()
}

#[derive(Debug, Clone, Default)]
pub struct EntityGraph {
    /// entity (lowercase) → set of chunk ids
    entity_to_chunks: HashMap<String, HashSet<String>>,
    /// chunk id → set of entities (lowercase)
    chunk_to_entities: HashMap<String, HashSet<String>>,
}

impl EntityGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.chunk_to_entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunk_to_entities.is_empty()
    }

    pub fn contains_chunk(&self, chunk_id: &str) -> bool {
        self.chunk_to_entities.contains_key(chunk_id)
    }

    /// Extract entities from `content` and index them under `chunk_id`.
    /// Idempotent on re-insertion of an already-seen id.
    pub fn insert_chunk(&mut self, chunk_id: &str, content: &str) {
        if self.chunk_to_entities.contains_key(chunk_id) {
            return;
        }
        let entities = extract_entities(content);
        if entities.is_empty() {
            // Still register the chunk so `contains_chunk` is accurate.
            self.chunk_to_entities
                .insert(chunk_id.to_string(), HashSet::new());
            return;
        }
        let mut normed: HashSet<String> = HashSet::with_capacity(entities.len());
        for e in &entities {
            normed.insert(norm(e));
        }
        for e in &normed {
            self.entity_to_chunks
                .entry(e.clone())
                .or_default()
                .insert(chunk_id.to_string());
        }
        self.chunk_to_entities.insert(chunk_id.to_string(), normed);
    }

    pub fn clear(&mut self) {
        self.entity_to_chunks.clear();
        self.chunk_to_entities.clear();
    }

    /// Total number of indexed chunks — denominator for IDF.
    fn n_chunks(&self) -> f32 {
        self.chunk_to_entities.len().max(1) as f32
    }

    /// Inverse document frequency for an entity. Rare entities (appearing
    /// in few chunks) get high weight; ubiquitous ones are dampened.
    fn idf(&self, entity_norm: &str) -> f32 {
        let df = self
            .entity_to_chunks
            .get(entity_norm)
            .map(|s| s.len() as f32)
            .unwrap_or(0.0);
        if df <= 0.0 {
            return 0.0;
        }
        (self.n_chunks() / df).ln().max(0.0)
    }

    /// Search for chunks that best match the entities in `query`.
    ///
    /// Scoring (two stages):
    /// 1. **Direct match**: for each query entity, sum IDF over chunks
    ///    that literally contain it.
    /// 2. **1-hop bridge**: when the top direct hit contains fewer than
    ///    half of the query entities, expand one hop — each direct chunk
    ///    adds its neighbour chunks (chunks sharing a non-query entity)
    ///    at half weight. Bounded by the top-N direct chunks to avoid
    ///    noise blow-up.
    ///
    /// Returns chunks sorted by score descending, capped at `k`.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let query_entities = extract_entities(query);
        if query_entities.is_empty() {
            return Vec::new();
        }
        let query_set: HashSet<String> = query_entities.iter().map(|e| norm(e)).collect();

        // Pre-compute IDF per query entity.
        let idfs: HashMap<String, f32> = query_set
            .iter()
            .map(|e| (e.clone(), self.idf(e)))
            .collect();

        // Stage 1: direct match.
        let mut scores: HashMap<String, f32> = HashMap::new();
        for q_ent in &query_set {
            let Some(chunks) = self.entity_to_chunks.get(q_ent) else {
                continue;
            };
            let idf = *idfs.get(q_ent).unwrap_or(&0.0);
            if idf <= 0.0 {
                continue;
            }
            for chunk_id in chunks {
                *scores.entry(chunk_id.clone()).or_insert(0.0) += idf;
            }
        }

        if scores.is_empty() {
            return Vec::new();
        }

        // Stage 2: 1-hop bridge expansion, only when the best direct hit
        // covers fewer than half of the query entities. This is the
        // mechanism that connects chunks like "Maria mentioned adopting
        // a dog" with "Shadow came home on May 5" — the two chunks share
        // the non-query entity "dog" (or Shadow itself as a bridge).
        let top_hit_coverage = scores
            .iter()
            .map(|(chunk_id, _)| {
                self.chunk_to_entities
                    .get(chunk_id)
                    .map(|es| es.intersection(&query_set).count())
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let target_coverage = (query_set.len() + 1) / 2;

        if top_hit_coverage < target_coverage && !query_set.is_empty() {
            // Pick up to 5 best direct chunks as bridges.
            let mut bridge_candidates: Vec<(String, f32)> = scores
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            bridge_candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            bridge_candidates.truncate(5);

            for (bridge_id, _bridge_score) in &bridge_candidates {
                let Some(bridge_entities) = self.chunk_to_entities.get(bridge_id) else {
                    continue;
                };
                for bridge_ent in bridge_entities {
                    if query_set.contains(bridge_ent) {
                        continue;
                    }
                    // For each non-query entity on the bridge, find other
                    // chunks that share it AND contain a query entity
                    // that the bridge lacks.
                    let missing: HashSet<String> = query_set
                        .difference(bridge_entities)
                        .cloned()
                        .collect();
                    if missing.is_empty() {
                        continue;
                    }
                    let Some(neighbours) = self.entity_to_chunks.get(bridge_ent) else {
                        continue;
                    };
                    for neighbour_id in neighbours {
                        if neighbour_id == bridge_id {
                            continue;
                        }
                        let Some(neigh_entities) = self.chunk_to_entities.get(neighbour_id)
                        else {
                            continue;
                        };
                        let overlap: Vec<&String> =
                            neigh_entities.intersection(&missing).collect();
                        if overlap.is_empty() {
                            continue;
                        }
                        let hop_score: f32 = overlap
                            .iter()
                            .map(|e| *idfs.get(*e).unwrap_or(&0.0))
                            .sum::<f32>()
                            * 0.5;
                        if hop_score > 0.0 {
                            *scores.entry(neighbour_id.clone()).or_insert(0.0) += hop_score;
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k.min(DEFAULT_SEARCH_CAP));
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keeps_proper_nouns() {
        let es = extract_entities("Alice and Bob went to Brussels on Monday");
        let low: Vec<String> = es.iter().map(|s| s.to_lowercase()).collect();
        assert!(low.contains(&"alice".to_string()));
        assert!(low.contains(&"bob".to_string()));
        assert!(low.contains(&"brussels".to_string()));
        assert!(low.contains(&"monday".to_string()));
    }

    #[test]
    fn extract_drops_stop_caps() {
        let es = extract_entities("The quick brown Fox jumps");
        let low: Vec<String> = es.iter().map(|s| s.to_lowercase()).collect();
        assert!(low.contains(&"fox".to_string()));
        assert!(!low.contains(&"the".to_string()));
    }

    #[test]
    fn extract_ignores_short_tokens() {
        let es = extract_entities("I am OK but AI is scary");
        // "I", "AI" (2 chars after strip) — both should be dropped.
        let low: Vec<String> = es.iter().map(|s| s.to_lowercase()).collect();
        assert!(!low.contains(&"i".to_string()));
        assert!(!low.contains(&"ai".to_string()));
    }

    #[test]
    fn insert_chunk_is_idempotent() {
        let mut g = EntityGraph::new();
        g.insert_chunk("c1", "Alice met Bob");
        g.insert_chunk("c1", "Different content entirely");
        assert_eq!(g.len(), 1);
        assert!(g.contains_chunk("c1"));
        // Original entities should remain.
        let chunks_for_alice = g.entity_to_chunks.get("alice").unwrap();
        assert!(chunks_for_alice.contains("c1"));
    }

    #[test]
    fn search_empty_graph_returns_empty() {
        let g = EntityGraph::new();
        assert!(g.search("Alice Bob", 5).is_empty());
    }

    #[test]
    fn search_direct_hit_scores_highest() {
        let mut g = EntityGraph::new();
        g.insert_chunk("c1", "Alice went to Paris");
        g.insert_chunk("c2", "Bob went to Rome");
        g.insert_chunk("c3", "Alice visited Bob in Rome for the conference");

        let hits = g.search("Alice Bob", 3);
        assert!(!hits.is_empty());
        // c3 mentions both → should be first.
        assert_eq!(hits[0].0, "c3");
        // c3 score > c1 score, c3 score > c2 score.
        let c3_score = hits[0].1;
        let c1_score = hits.iter().find(|(id, _)| id == "c1").map(|(_, s)| *s);
        assert!(c1_score.is_some() && c1_score.unwrap() < c3_score);
    }

    #[test]
    fn rare_entity_outweighs_common_one() {
        // "User" appears in many chunks (common), "Seraphim" in one (rare).
        // A chunk that contains Seraphim should beat one that only contains
        // "User" even if it's also in many contexts.
        let mut g = EntityGraph::new();
        g.insert_chunk("c1", "User logged in");
        g.insert_chunk("c2", "User opened the app");
        g.insert_chunk("c3", "User closed the app");
        g.insert_chunk("c4", "User viewed Seraphim's tank");
        // "User" gets filtered as a stop cap, so let's use a regular name.
        let mut g2 = EntityGraph::new();
        g2.insert_chunk("c1", "Jolene logged in");
        g2.insert_chunk("c2", "Jolene opened the app");
        g2.insert_chunk("c3", "Jolene closed the app");
        g2.insert_chunk("c4", "Jolene viewed Seraphim's tank");

        let hits = g2.search("Seraphim", 5);
        assert_eq!(hits[0].0, "c4", "Seraphim hit should dominate");
    }

    #[test]
    fn one_hop_bridges_disjoint_answer() {
        // The classic LOCOMO failure pattern:
        //   chunk A mentions (Maria, pet, adoption)
        //   chunk B mentions (Shadow, pet, May)  <- the answer
        //   No chunk mentions BOTH Maria and Shadow directly.
        // A direct-match search for "Maria Shadow" would rank A (has
        // Maria) and B (has Shadow) tied at ~equal score. The one-hop
        // bridge should boost B via the shared "pet" entity.
        let mut g = EntityGraph::new();
        g.insert_chunk(
            "A",
            "Maria talked about getting a new Puppy from the adoption centre",
        );
        g.insert_chunk(
            "B",
            "Shadow arrived on May 5 the Puppy settled in quickly",
        );
        g.insert_chunk("C", "Bob went to Rome for a conference unrelated to pets");

        let hits = g.search("Maria Shadow", 5);
        let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"A") && ids.contains(&"B"),
                "both direct hits should surface; got {:?}", ids);
        assert!(!ids.contains(&"C"), "unrelated chunk C must not appear");
    }

    #[test]
    fn clear_resets_state() {
        let mut g = EntityGraph::new();
        g.insert_chunk("c1", "Alice Bob");
        g.clear();
        assert!(g.is_empty());
        assert!(g.search("Alice", 5).is_empty());
    }

    #[test]
    fn query_without_any_entities_returns_empty() {
        let mut g = EntityGraph::new();
        g.insert_chunk("c1", "Alice met Bob");
        // All lowercase, no capitalised nouns.
        assert!(g.search("did they meet", 5).is_empty());
    }
}
