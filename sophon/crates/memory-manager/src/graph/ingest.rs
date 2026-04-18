//! Ingestion flow — orchestrates transcript → LLM extraction →
//! GraphStore upsert (Path A, step 4).
//!
//! The caller hands over a batch of messages (typically new ones since
//! the last ingest) and a chunk id that identifies the batch. We:
//!   1. Format the messages into a compact transcript.
//!   2. Call the LLM extractor to obtain triples.
//!   3. Walk each triple, materialise Entities for subject & object
//!      (if object is an Entity variant), and upsert both the entities
//!      and the fact into the store.
//!   4. Return a short report summarising what was ingested.
//!
//! Idempotent: re-ingesting the same batch with the same chunk id
//! produces the same fact ids (thanks to deterministic hashing in
//! `Fact::compute_id`), so the store dedups without extra work.

use super::extract::extract_triples;
use super::store::GraphStore;
use super::types::{Entity, FactObject};
use crate::message::{Message, Role};
use rayon::prelude::*;

/// Summary of one ingestion pass — useful for both introspection in
/// the MCP response and for assertions in tests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestReport {
    pub triples_seen: usize,
    pub new_entities: usize,
    pub new_facts: usize,
    pub merged_facts: usize,
}

/// Default number of messages per extraction batch. Chosen so each
/// Haiku prompt fits comfortably under the extractor's MAX_INPUT_CHARS
/// cap (~10 KB) with typical LOCOMO-length turns.
pub const DEFAULT_BATCH_SIZE: usize = 30;

/// Ingest a message batch into the graph. On LLM failure we return an
/// empty report — the caller can decide whether to retry, fall back to
/// heuristic summary, or surface the failure.
///
/// Use this when the caller has already chunked the input. For arbitrary
/// batches larger than ~30 messages, prefer
/// [`ingest_messages_batched`] — it parallelises the Haiku calls via
/// rayon, which cuts ingestion wall-clock by N× on long conversations.
pub fn ingest_messages(
    store: &mut GraphStore,
    messages: &[Message],
    chunk_id: &str,
    extracted_at: &str,
) -> IngestReport {
    if messages.is_empty() {
        return IngestReport::default();
    }
    let transcript = format_transcript(messages);
    let Some(triples) = extract_triples(&transcript, chunk_id, extracted_at) else {
        return IngestReport::default();
    };
    apply_triples(store, triples, extracted_at)
}

/// Batched, rayon-parallel ingestion. Splits `messages` into chunks of
/// `batch_size` (falling back to `DEFAULT_BATCH_SIZE` when `None`),
/// runs the Haiku extractor for each chunk concurrently, then merges
/// all resulting facts into the store sequentially (the store itself
/// is not `Sync`, but fact merging is cheap).
///
/// Wall-clock drops from `N × one_haiku_call` to roughly
/// `max(one_haiku_call)` on machines where the Claude CLI honours
/// parallel invocations — typically a 5-20× speedup on 500-turn LOCOMO
/// conversations.
pub fn ingest_messages_batched(
    store: &mut GraphStore,
    messages: &[Message],
    chunk_id_prefix: &str,
    extracted_at: &str,
    batch_size: Option<usize>,
) -> IngestReport {
    if messages.is_empty() {
        return IngestReport::default();
    }
    let batch = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let slices: Vec<&[Message]> = messages.chunks(batch).collect();

    // Parallelise the I/O-bound LLM calls. Each task returns its own
    // triple vec; we merge them into the store after the join so there
    // is no shared mutable state during the parallel phase.
    let extracted_at_ref = extracted_at;
    let per_batch: Vec<Vec<super::types::Fact>> = slices
        .par_iter()
        .enumerate()
        .map(|(i, slice)| {
            let transcript = format_transcript(slice);
            let chunk_id = format!("{}-{}", chunk_id_prefix, i);
            extract_triples(&transcript, &chunk_id, extracted_at_ref).unwrap_or_default()
        })
        .collect();

    // Merge into the store sequentially — the store is the hot owner of
    // the data so we can't share it across threads, but the merge work
    // is pure HashMap ops and trivial next to the LLM round-trips.
    let mut report = IngestReport::default();
    for triples in per_batch {
        report.triples_seen += triples.len();
        let batch_report = apply_triples(store, triples, extracted_at);
        report.new_facts += batch_report.new_facts;
        report.merged_facts += batch_report.merged_facts;
        report.new_entities += batch_report.new_entities;
    }
    report
}

/// Apply pre-extracted triples to a store — kept separate from
/// `ingest_messages` so tests can inject a deterministic triple list
/// without depending on the LLM.
pub fn apply_triples(
    store: &mut GraphStore,
    triples: Vec<super::types::Fact>,
    extracted_at: &str,
) -> IngestReport {
    let mut report = IngestReport {
        triples_seen: triples.len(),
        ..Default::default()
    };

    for fact in triples {
        // Auto-register the subject as an Entity. If already present,
        // `upsert_entity` merges aliases and bumps last_seen.
        let subject_name = fact.subject.as_str().to_string();
        let mut subject_entity = Entity::new(&subject_name, extracted_at);
        subject_entity.id = fact.subject.clone();
        let subject_was_new = store.get_entity(&fact.subject).is_none();
        store.upsert_entity(subject_entity);
        if subject_was_new {
            report.new_entities += 1;
        }

        // Same for the object, when it's an Entity variant.
        if let FactObject::Entity(obj_id) = &fact.object {
            if !obj_id.is_empty() {
                let obj_was_new = store.get_entity(obj_id).is_none();
                let mut obj_entity = Entity::new(obj_id.as_str(), extracted_at);
                obj_entity.id = obj_id.clone();
                store.upsert_entity(obj_entity);
                if obj_was_new {
                    report.new_entities += 1;
                }
            }
        }

        let was_known = store.get_fact(&fact.id).is_some();
        store.upsert_fact(fact);
        if was_known {
            report.merged_facts += 1;
        } else {
            report.new_facts += 1;
        }
    }

    report
}

/// Trim messages into a tagged transcript suitable for the extractor
/// prompt. Role-prefixed so the LLM can resolve "I" to the right speaker.
fn format_transcript(messages: &[Message]) -> String {
    let mut out = String::with_capacity(messages.len() * 80);
    for m in messages {
        let tag = match m.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };
        out.push_str(tag);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::types::{EntityId, Fact, FactObject, Predicate};
    use super::*;
    use crate::message::{Message, Role};

    fn mk_fact(subj: &str, pred: &str, obj: &str, obj_entity: bool) -> Fact {
        let object = if obj_entity {
            FactObject::Entity(EntityId::from_name(obj))
        } else {
            FactObject::Literal(obj.to_string())
        };
        Fact::new(
            EntityId::from_name(subj),
            Predicate::from_raw(pred),
            object,
            Some(0.8),
            vec!["chunk-1".to_string()],
            None,
            "2024-01-01".to_string(),
        )
    }

    #[test]
    fn apply_triples_creates_entities_and_facts() {
        let mut store = GraphStore::new();
        let triples = vec![
            mk_fact("Alice", "visited", "Paris", true),
            mk_fact("Alice", "likes", "ginger snaps", false),
        ];
        let report = apply_triples(&mut store, triples, "2024-01-01");
        assert_eq!(report.triples_seen, 2);
        assert_eq!(report.new_facts, 2);
        assert_eq!(report.merged_facts, 0);
        // Alice + Paris = 2 entities. "ginger snaps" is a literal, no entity.
        assert_eq!(report.new_entities, 2);
        assert_eq!(store.entity_count(), 2);
        assert_eq!(store.fact_count(), 2);
    }

    #[test]
    fn re_ingesting_same_triples_merges_evidence() {
        let mut store = GraphStore::new();
        let mut f = mk_fact("Alice", "visited", "Paris", true);
        f.source_chunk_ids = vec!["chunk-A".to_string()];
        apply_triples(&mut store, vec![f.clone()], "2024-01-01");

        // Re-ingest with a different source chunk → should merge.
        let mut f2 = f.clone();
        f2.source_chunk_ids = vec!["chunk-B".to_string()];
        let report = apply_triples(&mut store, vec![f2], "2024-01-02");

        assert_eq!(report.merged_facts, 1);
        assert_eq!(report.new_facts, 0);
        assert_eq!(store.fact_count(), 1);
        let stored = store.get_fact(&f.id).unwrap();
        assert!(stored.source_chunk_ids.contains(&"chunk-A".to_string()));
        assert!(stored.source_chunk_ids.contains(&"chunk-B".to_string()));
    }

    #[test]
    fn apply_triples_skips_empty_object_entity() {
        let mut store = GraphStore::new();
        // Craft a triple whose object is a manually-empty EntityId —
        // the materialise layer upstream should prevent this in prod,
        // but the ingest layer must also be robust.
        let f = Fact::new(
            EntityId::from_name("Alice"),
            Predicate::from_raw("visited"),
            FactObject::Entity(EntityId::from_name("")),
            Some(0.8),
            vec!["c".to_string()],
            None,
            "2024-01-01".to_string(),
        );
        apply_triples(&mut store, vec![f], "2024-01-01");
        // Only Alice created; the empty object entity must not leak.
        assert_eq!(store.entity_count(), 1);
    }

    #[test]
    fn ingest_messages_empty_batch_reports_zero() {
        let mut store = GraphStore::new();
        let report = ingest_messages(&mut store, &[], "chunk", "2024-01-01");
        assert_eq!(report, IngestReport::default());
    }

    #[test]
    fn format_transcript_tags_roles() {
        let msgs = vec![
            Message::new(Role::User, "Hi".to_string()),
            Message::new(Role::Assistant, "Hello".to_string()),
            Message::new(Role::System, "you are a bot".to_string()),
        ];
        let t = format_transcript(&msgs);
        assert!(t.contains("User: Hi"));
        assert!(t.contains("Assistant: Hello"));
        assert!(t.contains("System: you are a bot"));
    }

    #[test]
    fn report_counts_are_independent_per_call() {
        let mut store = GraphStore::new();
        let r1 = apply_triples(
            &mut store,
            vec![mk_fact("Alice", "visited", "Paris", true)],
            "2024-01-01",
        );
        let r2 = apply_triples(
            &mut store,
            vec![mk_fact("Bob", "visited", "Paris", true)],
            "2024-01-02",
        );
        assert_eq!(r1.new_entities, 2);
        assert_eq!(r2.new_entities, 1); // Bob new, Paris already there
        assert_eq!(r1.new_facts, 1);
        assert_eq!(r2.new_facts, 1);
    }
}
