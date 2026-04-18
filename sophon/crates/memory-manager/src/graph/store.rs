//! In-memory graph store with JSON-file persistence (Path A, step 2).
//!
//! Design:
//!   - All indexes are `HashMap`-backed for O(1) lookup on the hot paths
//!     (entity-by-id, fact-by-subject).
//!   - Persistence is "save snapshot" — the whole graph is serialised
//!     to a single JSON file. Fine for the LOCOMO scale we target
//!     (a few thousand entities, tens of thousands of facts). A
//!     SQLite backend can slot in later behind the same public API.
//!   - Mutations are idempotent. Inserting the same entity / fact
//!     twice merges rather than duplicating, which matches how the
//!     LLM extractor will behave across repeated ingestions.

use super::types::{Entity, EntityId, Fact, FactId, Predicate};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GraphStoreError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Snapshot shape used both for in-memory state and for on-disk JSON.
/// Keeping one struct keeps load/save symmetric and the schema obvious.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Snapshot {
    entities: HashMap<EntityId, Entity>,
    facts: HashMap<FactId, Fact>,
}

/// Graph store with three derived indexes kept in sync:
///   - `facts_by_subject`: subject → fact ids (for entity-centric query)
///   - `facts_by_object_entity`: entity-as-object → fact ids
///     (for reverse traversal)
///   - `facts_by_predicate`: predicate → fact ids (for relation
///     queries like "what did anyone visit?")
#[derive(Debug, Default)]
pub struct GraphStore {
    snapshot: Snapshot,
    facts_by_subject: HashMap<EntityId, HashSet<FactId>>,
    facts_by_object_entity: HashMap<EntityId, HashSet<FactId>>,
    facts_by_predicate: HashMap<Predicate, HashSet<FactId>>,
    path: Option<PathBuf>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open (or create) a JSON-backed store at `path`. If the file
    /// exists it is loaded; otherwise the store starts empty and the
    /// path is remembered for later `save()` calls.
    pub fn open<P: Into<PathBuf>>(path: P) -> Result<Self, GraphStoreError> {
        let path = path.into();
        let mut store = Self::default();
        store.path = Some(path.clone());
        if path.exists() {
            let raw = fs::read_to_string(&path)?;
            if !raw.trim().is_empty() {
                let snap: Snapshot = serde_json::from_str(&raw)?;
                store.snapshot = snap;
                store.rebuild_indexes();
            }
        }
        Ok(store)
    }

    /// Save the current snapshot to the attached path. No-op if the
    /// store was built via `new()` without a path.
    pub fn save(&self) -> Result<(), GraphStoreError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.snapshot)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn rebuild_indexes(&mut self) {
        self.facts_by_subject.clear();
        self.facts_by_object_entity.clear();
        self.facts_by_predicate.clear();
        for (fid, f) in &self.snapshot.facts {
            self.facts_by_subject
                .entry(f.subject.clone())
                .or_default()
                .insert(fid.clone());
            self.facts_by_predicate
                .entry(f.predicate.clone())
                .or_default()
                .insert(fid.clone());
            if let super::types::FactObject::Entity(e) = &f.object {
                self.facts_by_object_entity
                    .entry(e.clone())
                    .or_default()
                    .insert(fid.clone());
            }
        }
    }

    pub fn entity_count(&self) -> usize {
        self.snapshot.entities.len()
    }

    pub fn fact_count(&self) -> usize {
        self.snapshot.facts.len()
    }

    /// Insert or merge an entity. If the id already exists, aliases are
    /// unioned and `last_seen` is bumped. The display `name` is kept
    /// from the first insert (callers can edit via `update_name()`).
    pub fn upsert_entity(&mut self, entity: Entity) {
        let id = entity.id.clone();
        match self.snapshot.entities.get_mut(&id) {
            Some(existing) => {
                for alias in &entity.aliases {
                    existing.aliases.insert(alias.clone());
                }
                // also remember the incoming surface form as an alias
                if existing.name != entity.name
                    && super::types::EntityId::from_name(&entity.name) == existing.id
                {
                    existing.aliases.insert(entity.name.clone());
                }
                if entity.type_hint.is_some() && existing.type_hint.is_none() {
                    existing.type_hint = entity.type_hint;
                }
                if entity.last_seen > existing.last_seen {
                    existing.last_seen = entity.last_seen;
                }
            }
            None => {
                self.snapshot.entities.insert(id, entity);
            }
        }
    }

    /// Insert or merge a fact. If the fact id exists (same subject /
    /// predicate / object / when) the source-chunk evidence is unioned
    /// and the highest confidence wins.
    pub fn upsert_fact(&mut self, fact: Fact) {
        let fid = fact.id.clone();
        match self.snapshot.facts.get_mut(&fid) {
            Some(existing) => {
                for c in &fact.source_chunk_ids {
                    if !existing.source_chunk_ids.contains(c) {
                        existing.source_chunk_ids.push(c.clone());
                    }
                }
                if fact.confidence > existing.confidence {
                    existing.confidence = fact.confidence;
                }
                if fact.extracted_at > existing.extracted_at {
                    existing.extracted_at = fact.extracted_at;
                }
            }
            None => {
                self.facts_by_subject
                    .entry(fact.subject.clone())
                    .or_default()
                    .insert(fid.clone());
                self.facts_by_predicate
                    .entry(fact.predicate.clone())
                    .or_default()
                    .insert(fid.clone());
                if let super::types::FactObject::Entity(e) = &fact.object {
                    self.facts_by_object_entity
                        .entry(e.clone())
                        .or_default()
                        .insert(fid.clone());
                }
                self.snapshot.facts.insert(fid, fact);
            }
        }
    }

    pub fn get_entity(&self, id: &EntityId) -> Option<&Entity> {
        self.snapshot.entities.get(id)
    }

    pub fn get_fact(&self, id: &FactId) -> Option<&Fact> {
        self.snapshot.facts.get(id)
    }

    /// All facts whose subject matches the given entity id.
    pub fn facts_with_subject(&self, subject: &EntityId) -> Vec<&Fact> {
        self.facts_by_subject
            .get(subject)
            .map(|ids| {
                ids.iter()
                    .filter_map(|fid| self.snapshot.facts.get(fid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All facts whose object is an entity matching `object`.
    pub fn facts_with_object_entity(&self, object: &EntityId) -> Vec<&Fact> {
        self.facts_by_object_entity
            .get(object)
            .map(|ids| {
                ids.iter()
                    .filter_map(|fid| self.snapshot.facts.get(fid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All facts touching `entity` in either subject or object position.
    pub fn facts_touching(&self, entity: &EntityId) -> Vec<&Fact> {
        let mut out: Vec<&Fact> = Vec::new();
        let mut seen: HashSet<FactId> = HashSet::new();
        for f in self.facts_with_subject(entity) {
            if seen.insert(f.id.clone()) {
                out.push(f);
            }
        }
        for f in self.facts_with_object_entity(entity) {
            if seen.insert(f.id.clone()) {
                out.push(f);
            }
        }
        out
    }

    /// Iterate every entity for bulk-scan use cases (e.g. alias-based
    /// query disambiguation that needs to look across all aliases).
    pub fn iter_entities(&self) -> impl Iterator<Item = &Entity> {
        self.snapshot.entities.values()
    }

    pub fn iter_facts(&self) -> impl Iterator<Item = &Fact> {
        self.snapshot.facts.values()
    }

    pub fn clear(&mut self) {
        self.snapshot = Snapshot::default();
        self.facts_by_subject.clear();
        self.facts_by_object_entity.clear();
        self.facts_by_predicate.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Entity, EntityId, Fact, FactObject, Predicate};
    use super::*;
    use tempfile::tempdir;

    fn mk_entity(name: &str) -> Entity {
        Entity::new(name, "2024-01-01T00:00:00Z")
    }

    fn mk_fact(subj: &str, pred: &str, obj_entity: &str) -> Fact {
        Fact::new(
            EntityId::from_name(subj),
            Predicate::from_raw(pred),
            FactObject::Entity(EntityId::from_name(obj_entity)),
            Some(0.8),
            vec!["chunk-1".to_string()],
            None,
            "2024-01-01T00:00:00Z".to_string(),
        )
    }

    #[test]
    fn empty_store_is_empty() {
        let s = GraphStore::new();
        assert_eq!(s.entity_count(), 0);
        assert_eq!(s.fact_count(), 0);
    }

    #[test]
    fn upsert_entity_adds_and_merges() {
        let mut s = GraphStore::new();
        s.upsert_entity(mk_entity("Alice"));
        assert_eq!(s.entity_count(), 1);

        // Same id, alternate casing — should merge, not duplicate.
        let mut e = mk_entity("ALICE");
        e.aliases.insert("al".to_string());
        s.upsert_entity(e);
        assert_eq!(s.entity_count(), 1);

        let stored = s.get_entity(&EntityId::from_name("Alice")).unwrap();
        // Original name preserved
        assert_eq!(stored.name, "Alice");
        // Alias "al" carried over
        assert!(stored.aliases.contains("al"));
        // Surface form "ALICE" added as alias
        assert!(stored.aliases.contains("ALICE"));
    }

    #[test]
    fn upsert_fact_adds_and_merges_evidence() {
        let mut s = GraphStore::new();
        let f = mk_fact("Alice", "visited", "Paris");
        s.upsert_fact(f.clone());
        assert_eq!(s.fact_count(), 1);

        // Same fact, different chunk → evidence merges, count stays at 1.
        let mut f2 = f.clone();
        f2.source_chunk_ids = vec!["chunk-2".to_string()];
        f2.confidence = 0.95;
        s.upsert_fact(f2);
        assert_eq!(s.fact_count(), 1);

        let got = s.get_fact(&f.id).unwrap();
        assert!(got.source_chunk_ids.contains(&"chunk-1".to_string()));
        assert!(got.source_chunk_ids.contains(&"chunk-2".to_string()));
        assert!((got.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn facts_with_subject_finds_all() {
        let mut s = GraphStore::new();
        s.upsert_fact(mk_fact("Alice", "visited", "Paris"));
        s.upsert_fact(mk_fact("Alice", "likes", "Bob"));
        s.upsert_fact(mk_fact("Bob", "visited", "Rome"));
        let alice = EntityId::from_name("Alice");
        let hits = s.facts_with_subject(&alice);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn facts_with_object_entity_finds_reverse_edges() {
        let mut s = GraphStore::new();
        s.upsert_fact(mk_fact("Alice", "visited", "Paris"));
        s.upsert_fact(mk_fact("Bob", "visited", "Paris"));
        let paris = EntityId::from_name("Paris");
        let hits = s.facts_with_object_entity(&paris);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn facts_touching_union_subject_and_object() {
        let mut s = GraphStore::new();
        s.upsert_fact(mk_fact("Alice", "visited", "Paris"));
        s.upsert_fact(mk_fact("Bob", "met", "Alice"));
        s.upsert_fact(mk_fact("Alice", "likes", "cats"));
        let alice = EntityId::from_name("Alice");
        let hits = s.facts_touching(&alice);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("graph.json");

        {
            let mut s = GraphStore::open(&path).unwrap();
            s.upsert_entity(mk_entity("Alice"));
            s.upsert_fact(mk_fact("Alice", "visited", "Paris"));
            s.save().unwrap();
        }

        // Reopen — data should come back identical.
        let s2 = GraphStore::open(&path).unwrap();
        assert_eq!(s2.entity_count(), 1);
        assert_eq!(s2.fact_count(), 1);
        assert!(s2.get_entity(&EntityId::from_name("Alice")).is_some());
        // Indexes must also be rebuilt on load.
        let alice = EntityId::from_name("Alice");
        let hits = s2.facts_with_subject(&alice);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn open_missing_file_yields_empty_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");
        let s = GraphStore::open(&path).unwrap();
        assert_eq!(s.entity_count(), 0);
        assert_eq!(s.fact_count(), 0);
    }

    #[test]
    fn open_empty_file_yields_empty_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.json");
        std::fs::write(&path, "").unwrap();
        let s = GraphStore::open(&path).unwrap();
        assert_eq!(s.entity_count(), 0);
    }

    #[test]
    fn clear_resets_all_indexes() {
        let mut s = GraphStore::new();
        s.upsert_entity(mk_entity("Alice"));
        s.upsert_fact(mk_fact("Alice", "visited", "Paris"));
        s.clear();
        assert_eq!(s.entity_count(), 0);
        assert_eq!(s.fact_count(), 0);
        let alice = EntityId::from_name("Alice");
        assert!(s.facts_with_subject(&alice).is_empty());
        assert!(s.facts_with_object_entity(&EntityId::from_name("Paris")).is_empty());
    }

    #[test]
    fn higher_confidence_wins_on_merge() {
        let mut s = GraphStore::new();
        let mut f = mk_fact("Alice", "visited", "Paris");
        f.confidence = 0.5;
        s.upsert_fact(f.clone());
        let mut f2 = mk_fact("Alice", "visited", "Paris");
        f2.confidence = 0.9;
        s.upsert_fact(f2.clone());
        assert!((s.get_fact(&f.id).unwrap().confidence - 0.9).abs() < 1e-6);
        // Reverse order must yield the same result.
        let mut s2 = GraphStore::new();
        s2.upsert_fact(f2);
        s2.upsert_fact(f);
        assert!((s2.get_fact(&mk_fact("Alice", "visited", "Paris").id).unwrap().confidence - 0.9).abs() < 1e-6);
    }
}
