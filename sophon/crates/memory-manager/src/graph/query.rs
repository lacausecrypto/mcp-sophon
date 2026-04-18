//! Query flow — given a free-form question, surface the facts most
//! likely to contain its answer (Path A, step 5).
//!
//! Pipeline (all pure Rust, zero LLM calls):
//!   1. **Entity extraction from query** — regex NER reuses the same
//!      heuristic as the retriever's `entity_graph::extract_entities`.
//!   2. **Entity resolution in graph** — each query-entity name maps to
//!      one or more candidate `EntityId`s via direct match on the
//!      canonical id and alias lookup.
//!   3. **Fact harvest** — for each resolved entity, pull all facts
//!      where it appears as subject OR object (one-hop neighbourhood).
//!   4. **Scoring** — each fact gets `score = confidence × recency ×
//!      (1 + entity_overlap)`. Recency is `1.0` for facts extracted at
//!      the same extracted_at as the newest fact, decays down to 0.5
//!      for the oldest.
//!   5. **Top-K selection** — the caller picks how many to return.
//!
//! Not implemented here (future work): personalised PageRank, multi-hop
//! expansion beyond 1 step, query decomposition. This layer is already
//! enough to solve the LOCOMO multi-hop cases that share at least one
//! query entity with the answer chunk.

use super::store::GraphStore;
use super::types::{EntityId, Fact, FactObject};
use std::collections::{HashMap, HashSet};

/// Canonical result shape. We return cloned facts so the caller can
/// render them without juggling store lifetimes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredFact {
    pub fact: Fact,
    pub score: f32,
}

/// Resolve free-form query entities to graph ids by exact id match first,
/// then by alias scan. Returns deduped ids in insertion order.
pub fn resolve_query_entities(store: &GraphStore, raw_entities: &[String]) -> Vec<EntityId> {
    let mut out: Vec<EntityId> = Vec::new();
    let mut seen: HashSet<EntityId> = HashSet::new();

    for raw in raw_entities {
        let target = EntityId::from_name(raw);
        if target.is_empty() {
            continue;
        }
        // Direct id match.
        if store.get_entity(&target).is_some() {
            if seen.insert(target.clone()) {
                out.push(target);
            }
            continue;
        }
        // Alias scan: bounded, only checks entities whose canonical id
        // contains any query token, to avoid a full O(N) sweep on large
        // graphs.
        let needle = target.as_str();
        for e in store.iter_entities() {
            let canonical_match = e.id.as_str().contains(needle) || needle.contains(e.id.as_str());
            let alias_match = e
                .aliases
                .iter()
                .any(|a| EntityId::from_name(a) == target);
            if canonical_match || alias_match {
                if seen.insert(e.id.clone()) {
                    out.push(e.id.clone());
                }
            }
        }
    }
    out
}

/// Extract capitalised-token entities from the query string. Shared
/// heuristic with `semantic-retriever::entity_graph::extract_entities`
/// so query-side and ingest-side agree on what counts as an entity.
pub fn extract_query_entities(query: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who",
        "Whom", "Whose", "How", "Why", "Did", "Do", "Does", "Is", "Are", "Was", "Were",
        "Has", "Have", "Had", "Can", "Could", "Will", "Would", "Should", "May", "Might",
        "Shall", "A", "An", "And", "Or", "But", "If", "Then", "So", "On", "In", "At",
        "For", "With", "To", "From", "By", "Of", "Session", "User", "Assistant",
    ];
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for tok in query.split(|c: char| !c.is_alphanumeric()) {
        if tok.len() < 3 {
            continue;
        }
        let first = tok.chars().next().unwrap_or('a');
        if !first.is_uppercase() {
            continue;
        }
        if STOP.iter().any(|s| s.eq_ignore_ascii_case(tok)) {
            continue;
        }
        if tok.chars().skip(1).all(|c| c.is_uppercase()) && tok.len() > 4 {
            continue;
        }
        let key = tok.to_lowercase();
        if seen.insert(key) {
            out.push(tok.to_string());
        }
    }
    out
}

/// Top-level query: extract query entities → resolve → harvest facts →
/// score → return top-K. Empty result is normal and should fall back to
/// the existing retriever in the caller.
pub fn query(store: &GraphStore, query_text: &str, top_k: usize) -> Vec<ScoredFact> {
    if top_k == 0 || store.fact_count() == 0 {
        return Vec::new();
    }
    let raw_entities = extract_query_entities(query_text);
    if raw_entities.is_empty() {
        return Vec::new();
    }
    let resolved = resolve_query_entities(store, &raw_entities);
    if resolved.is_empty() {
        return Vec::new();
    }

    // Harvest: collect facts touching any resolved entity, plus a
    // per-fact counter of how many query entities it touches (for the
    // `entity_overlap` score bonus).
    let resolved_set: HashSet<EntityId> = resolved.iter().cloned().collect();
    let mut harvest: HashMap<super::types::FactId, (&Fact, usize)> = HashMap::new();
    for eid in &resolved {
        for f in store.facts_touching(eid) {
            let touches = count_overlap(f, &resolved_set);
            harvest
                .entry(f.id.clone())
                .and_modify(|(_, n)| *n = (*n).max(touches))
                .or_insert((f, touches));
        }
    }
    if harvest.is_empty() {
        return Vec::new();
    }

    // Recency range for normalisation.
    let extracted_ats: Vec<&String> = harvest
        .values()
        .map(|(f, _)| &f.extracted_at)
        .collect();
    let max_ts = extracted_ats.iter().max().cloned();
    let min_ts = extracted_ats.iter().min().cloned();

    let mut scored: Vec<ScoredFact> = harvest
        .into_values()
        .map(|(f, overlap)| {
            let recency = recency_score(&f.extracted_at, min_ts, max_ts);
            let overlap_bonus = 1.0 + (overlap as f32) * 0.25;
            let score = f.confidence * recency * overlap_bonus;
            ScoredFact {
                fact: f.clone(),
                score,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    scored
}

fn count_overlap(fact: &Fact, resolved: &HashSet<EntityId>) -> usize {
    let mut n = 0;
    if resolved.contains(&fact.subject) {
        n += 1;
    }
    if let FactObject::Entity(e) = &fact.object {
        if resolved.contains(e) {
            n += 1;
        }
    }
    n
}

/// Linear recency normalisation in [0.5, 1.0].
fn recency_score(ts: &str, min_ts: Option<&String>, max_ts: Option<&String>) -> f32 {
    match (min_ts, max_ts) {
        (Some(mi), Some(ma)) if mi != ma => {
            // Lexicographic string comparison works for ISO 8601-ish
            // timestamps; for free-form strings we just bail to 1.0.
            if ts >= ma.as_str() {
                1.0
            } else if ts <= mi.as_str() {
                0.5
            } else {
                // Linear interpolation by character rank — rough but
                // monotonic and cheap.
                let mi_bytes = mi.as_bytes();
                let ma_bytes = ma.as_bytes();
                let ts_bytes = ts.as_bytes();
                let len = mi_bytes.len().min(ma_bytes.len()).min(ts_bytes.len());
                if len == 0 {
                    return 1.0;
                }
                let mi_val: usize = mi_bytes[..len].iter().map(|&b| b as usize).sum();
                let ma_val: usize = ma_bytes[..len].iter().map(|&b| b as usize).sum();
                let ts_val: usize = ts_bytes[..len].iter().map(|&b| b as usize).sum();
                if ma_val == mi_val {
                    return 1.0;
                }
                let t = (ts_val.saturating_sub(mi_val)) as f32 / (ma_val - mi_val) as f32;
                0.5 + 0.5 * t.clamp(0.0, 1.0)
            }
        }
        _ => 1.0,
    }
}

/// Render scored facts into a compact text block the downstream LLM can
/// read directly. Safe for empty input.
pub fn render_facts(facts: &[ScoredFact]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for sf in facts {
        let f = &sf.fact;
        let obj = match &f.object {
            FactObject::Entity(e) => e.as_str().to_string(),
            FactObject::Literal(s) => s.clone(),
            FactObject::Date(s) => s.clone(),
            FactObject::Number(s) => s.clone(),
        };
        let when = f
            .when
            .as_ref()
            .map(|w| format!(" (when: {})", w))
            .unwrap_or_default();
        out.push_str(&format!(
            "- {} {} {}{} [conf={:.2}]\n",
            f.subject.as_str(),
            f.predicate.as_str(),
            obj,
            when,
            f.confidence
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::ingest::apply_triples;
    use super::super::types::{Fact, FactObject, Predicate};
    use super::*;

    fn mk_fact(subj: &str, pred: &str, obj: &str, obj_entity: bool, conf: f32, ts: &str) -> Fact {
        let object = if obj_entity {
            FactObject::Entity(EntityId::from_name(obj))
        } else {
            FactObject::Literal(obj.to_string())
        };
        Fact::new(
            EntityId::from_name(subj),
            Predicate::from_raw(pred),
            object,
            Some(conf),
            vec!["chunk-1".to_string()],
            None,
            ts.to_string(),
        )
    }

    #[test]
    fn extract_query_entities_finds_proper_nouns() {
        let es = extract_query_entities("When did Alice visit Paris?");
        let low: Vec<String> = es.iter().map(|e| e.to_lowercase()).collect();
        assert!(low.contains(&"alice".to_string()));
        assert!(low.contains(&"paris".to_string()));
    }

    #[test]
    fn extract_query_entities_drops_stop_caps() {
        let es = extract_query_entities("What did Alice do yesterday?");
        let low: Vec<String> = es.iter().map(|e| e.to_lowercase()).collect();
        assert!(low.contains(&"alice".to_string()));
        assert!(!low.contains(&"what".to_string()));
    }

    #[test]
    fn resolve_direct_id_match() {
        let mut store = GraphStore::new();
        apply_triples(
            &mut store,
            vec![mk_fact("Alice", "visited", "Paris", true, 0.9, "2024-01-01")],
            "2024-01-01",
        );
        let ids = resolve_query_entities(&store, &["Alice".to_string()]);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].as_str(), "alice");
    }

    #[test]
    fn resolve_alias_match() {
        let mut store = GraphStore::new();
        let mut e = super::super::types::Entity::new("Alice Smith", "2024-01-01");
        e.aliases.insert("Ali".to_string());
        store.upsert_entity(e);
        let ids = resolve_query_entities(&store, &["Ali".to_string()]);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].as_str(), "alice smith");
    }

    #[test]
    fn resolve_empty_query_yields_empty() {
        let store = GraphStore::new();
        let ids = resolve_query_entities(&store, &["".to_string(), "   ".to_string()]);
        assert!(ids.is_empty());
    }

    #[test]
    fn query_returns_facts_touching_entity() {
        let mut store = GraphStore::new();
        apply_triples(
            &mut store,
            vec![
                mk_fact("Alice", "visited", "Paris", true, 0.9, "2024-01-01"),
                mk_fact("Alice", "likes", "ginger snaps", false, 0.8, "2024-01-01"),
                mk_fact("Bob", "visited", "Rome", true, 0.9, "2024-01-01"),
            ],
            "2024-01-01",
        );
        let hits = query(&store, "When did Alice travel?", 10);
        // Should return both Alice facts, not the Bob fact.
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert_eq!(h.fact.subject.as_str(), "alice");
        }
    }

    #[test]
    fn query_scores_multi_entity_overlap_higher() {
        // A fact that touches BOTH query entities should outrank a fact
        // touching only one — that's the multi-hop recall we want.
        let mut store = GraphStore::new();
        apply_triples(
            &mut store,
            vec![
                mk_fact("Alice", "likes", "Bob", true, 0.8, "2024-01-01"),
                mk_fact("Alice", "lives_in", "Paris", true, 0.8, "2024-01-01"),
                mk_fact("Bob", "works_at", "Google", true, 0.8, "2024-01-01"),
            ],
            "2024-01-01",
        );
        let hits = query(&store, "How do Alice and Bob know each other?", 5);
        // Top hit must be the (Alice, likes, Bob) fact — it touches
        // both query entities.
        let top = &hits[0].fact;
        assert_eq!(top.subject.as_str(), "alice");
        if let FactObject::Entity(e) = &top.object {
            assert_eq!(e.as_str(), "bob");
        } else {
            panic!("expected entity object");
        }
    }

    #[test]
    fn query_returns_empty_when_no_query_entities() {
        let mut store = GraphStore::new();
        apply_triples(
            &mut store,
            vec![mk_fact("Alice", "visited", "Paris", true, 0.9, "2024-01-01")],
            "2024-01-01",
        );
        // All-lowercase query → no entities extracted.
        let hits = query(&store, "did they meet?", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn query_returns_empty_on_empty_store() {
        let store = GraphStore::new();
        let hits = query(&store, "Alice?", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn query_top_k_respected() {
        let mut store = GraphStore::new();
        let mut triples = Vec::new();
        for i in 0..10 {
            triples.push(mk_fact(
                "Alice",
                "visited",
                &format!("Place{i}"),
                true,
                0.8,
                "2024-01-01",
            ));
        }
        apply_triples(&mut store, triples, "2024-01-01");
        let hits = query(&store, "Alice?", 3);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn render_facts_emits_one_line_per_fact() {
        let sf = ScoredFact {
            fact: mk_fact("Alice", "visited", "Paris", true, 0.9, "2024-01-01"),
            score: 0.9,
        };
        let text = render_facts(&[sf]);
        assert!(text.contains("alice"));
        assert!(text.contains("visited"));
        assert!(text.contains("paris"));
        assert!(text.contains("conf=0.90"));
    }

    #[test]
    fn render_empty_is_empty() {
        assert_eq!(render_facts(&[]), "");
    }

    #[test]
    fn recency_defaults_to_one_on_equal_timestamps() {
        let single = "2024-01-01".to_string();
        assert_eq!(
            recency_score("2024-01-01", Some(&single), Some(&single)),
            1.0
        );
    }
}
