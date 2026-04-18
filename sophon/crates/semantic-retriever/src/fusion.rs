//! Reciprocal Rank Fusion — merge multiple retrieval rankings into one.
//!
//! Used by the multi-hop LLM-in-the-loop retrieval path: when a query is
//! decomposed into sub-queries, each sub-query produces its own ranked list
//! of chunks. RRF fuses them with rank-based scoring so a chunk that appears
//! in several lists wins over a chunk that only ranks #1 in one list.
//!
//! Formula (Cormack, Clarke, Buettcher 2009):
//!
//! ```text
//! score(doc) = Σ_i  1 / (k + rank_i(doc))
//! ```
//!
//! `k = 60` is the standard constant from the original paper — it dampens
//! the contribution of very high ranks so a top-1 hit cannot single-handedly
//! drown a doc that appears consistently in the top-10 across lists.

use crate::retriever::ScoredChunk;
use std::collections::HashMap;

/// Default `k` constant from Cormack et al. 2009. Lower `k` amplifies top
/// ranks (more elitist); higher `k` flattens (more democratic).
pub const RRF_K: f32 = 60.0;

/// Fuse `rankings` into a single ranked list using Reciprocal Rank Fusion.
///
/// Each inner Vec is assumed to be sorted by descending original score.
/// The returned chunks carry the aggregated RRF score in `ScoredChunk::score`;
/// the original similarity/rerank scores are **not** preserved — RRF is a
/// purely rank-based fusion. Order is by RRF score descending.
///
/// Chunks are deduplicated by `chunk.id`.
pub fn rrf_fuse(rankings: &[Vec<ScoredChunk>], k: f32) -> Vec<ScoredChunk> {
    if rankings.is_empty() {
        return Vec::new();
    }

    let mut agg: HashMap<String, (ScoredChunk, f32)> = HashMap::new();

    for ranking in rankings {
        for (rank_idx, sc) in ranking.iter().enumerate() {
            let rank = (rank_idx + 1) as f32;
            let contribution = 1.0 / (k + rank);
            agg.entry(sc.chunk.id.clone())
                .and_modify(|(_, s)| *s += contribution)
                .or_insert_with(|| {
                    let mut fresh = sc.clone();
                    fresh.score = 0.0;
                    (fresh, contribution)
                });
        }
    }

    let mut fused: Vec<ScoredChunk> = agg
        .into_values()
        .map(|(mut sc, score)| {
            sc.score = score;
            sc
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkType;
    use crate::retriever::RetrievedChunk;
    use chrono::Utc;

    fn mk(id: &str, score: f32) -> ScoredChunk {
        ScoredChunk {
            chunk: RetrievedChunk {
                id: id.to_string(),
                content: format!("content-{id}"),
                token_count: 10,
                timestamp: Utc::now(),
                session_id: None,
                chunk_type: ChunkType::UserStatement,
            },
            score,
        }
    }

    #[test]
    fn empty_rankings_return_empty() {
        assert!(rrf_fuse(&[], RRF_K).is_empty());
    }

    #[test]
    fn single_ranking_preserves_order() {
        let r = vec![mk("a", 0.9), mk("b", 0.7), mk("c", 0.5)];
        let fused = rrf_fuse(&[r], RRF_K);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].chunk.id, "a");
        assert_eq!(fused[1].chunk.id, "b");
        assert_eq!(fused[2].chunk.id, "c");
    }

    #[test]
    fn doc_in_multiple_lists_outranks_single_list_topper() {
        // "b" is rank 1 only in list 2; "a" is rank 2 in both lists.
        // With k=60: a = 1/62 + 1/62 ≈ 0.0322;  b = 1/61 ≈ 0.0164.
        // So "a" wins — appearing consistently beats one top-1.
        let r1 = vec![mk("x", 1.0), mk("a", 0.9)];
        let r2 = vec![mk("b", 1.0), mk("a", 0.9)];
        let fused = rrf_fuse(&[r1, r2], RRF_K);
        let a_pos = fused.iter().position(|c| c.chunk.id == "a").unwrap();
        let b_pos = fused.iter().position(|c| c.chunk.id == "b").unwrap();
        assert!(
            a_pos < b_pos,
            "a (in both lists) should outrank b (one list)"
        );
    }

    #[test]
    fn deduplicates_by_chunk_id() {
        let r1 = vec![mk("a", 0.9)];
        let r2 = vec![mk("a", 0.5)];
        let fused = rrf_fuse(&[r1, r2], RRF_K);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].chunk.id, "a");
    }

    #[test]
    fn rrf_score_matches_formula() {
        // Single list, single doc at rank 1 → score = 1/(k+1).
        let r = vec![mk("a", 1.0)];
        let fused = rrf_fuse(&[r], RRF_K);
        let expected = 1.0 / (RRF_K + 1.0);
        assert!((fused[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn smaller_k_amplifies_top_rank() {
        // With k=1: rank1 = 1/2 = 0.5, rank2 = 1/3 ≈ 0.333.
        // With k=60: rank1 = 1/61, rank2 = 1/62 — much closer.
        let r = vec![mk("a", 1.0), mk("b", 0.9)];
        let small_k = rrf_fuse(&[r.clone()], 1.0);
        let big_k = rrf_fuse(&[r], 60.0);
        let small_gap = small_k[0].score - small_k[1].score;
        let big_gap = big_k[0].score - big_k[1].score;
        assert!(small_gap > big_gap);
    }
}
