//! Data model for the ingestion-time graph memory (Path A, step 1).
//!
//! The graph stores (subject, predicate, object) triples extracted from
//! conversation messages by an LLM pass, plus the entity + predicate
//! vocabularies they reference. It is the deterministic, query-time
//! zero-LLM alternative to block-based summarisation: at ingestion we
//! pay one Haiku call to extract facts; at query time we traverse the
//! graph in pure Rust.
//!
//! Design principles:
//!   - **Normalised keys**: `EntityId` and `Predicate` are lowercased,
//!     whitespace-collapsed forms so the same concept ("Alice", "alice",
//!     "ALICE") maps to one node. The original surface form is kept in
//!     `Entity::name` for display.
//!   - **Aliases as first-class**: "Alice", "my wife", "her" may all map
//!     to the same entity. We store aliases so the query path can match
//!     both the canonical name and any alias.
//!   - **Facts are immutable**: conflicts go in as separate facts with
//!     different confidence scores. A merge/dedup pass decides winners
//!     later — we do not rewrite history.
//!   - **Deterministic IDs**: fact IDs hash (subject, predicate, object,
//!     timestamp) so re-ingesting the same message does not duplicate.
//!     This makes `ingest()` idempotent.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Normalised identifier for an entity. Always lowercase, single-spaced.
/// Used as the key in the graph's adjacency maps so retrieval can hit
/// the right entity regardless of surface casing / punctuation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityId(String);

impl EntityId {
    /// Canonicalise a free-form entity string into a stable id.
    /// Lowercases, strips non-alphanumerics (keeps `-` and `_`), and
    /// collapses internal whitespace to a single space. Empty input
    /// returns the zero-length id — callers should avoid inserting it.
    pub fn from_name(raw: &str) -> Self {
        let mut out = String::with_capacity(raw.len());
        let mut last_was_space = false;
        for c in raw.chars() {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                for lc in c.to_lowercase() {
                    out.push(lc);
                }
                last_was_space = false;
            } else if c.is_whitespace() {
                if !last_was_space && !out.is_empty() {
                    out.push(' ');
                }
                last_was_space = true;
            }
        }
        // trim trailing space
        while out.ends_with(' ') {
            out.pop();
        }
        EntityId(out)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// An entity node in the graph. The canonical `name` is the display
/// form; `aliases` are alternate surface forms the LLM extractor saw for
/// the same entity (all normalise to the same `EntityId`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub aliases: BTreeSet<String>,
    pub type_hint: Option<EntityType>,
    /// First time this entity was mentioned (ISO 8601 string, kept
    /// opaque to avoid pulling in a time crate in this module).
    pub first_seen: String,
    pub last_seen: String,
}

impl Entity {
    pub fn new(name: &str, first_seen: &str) -> Self {
        let id = EntityId::from_name(name);
        Self {
            id,
            name: name.to_string(),
            aliases: BTreeSet::new(),
            type_hint: None,
            first_seen: first_seen.to_string(),
            last_seen: first_seen.to_string(),
        }
    }

    /// Add an alias iff it normalises to the same `EntityId` as `name`.
    /// Returns true on insert.
    pub fn add_alias(&mut self, alias: &str) -> bool {
        if EntityId::from_name(alias) != self.id {
            return false;
        }
        if alias == self.name {
            return false;
        }
        self.aliases.insert(alias.to_string())
    }
}

/// Coarse entity type hint the LLM extractor may emit. Left optional so
/// missing hints don't break the flow; used by the query path to bias
/// scoring (e.g. `Date` entities matter more for temporal questions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Event,
    Object,
    Concept,
    Date,
}

/// Normalised predicate — the relation connecting subject to object.
/// Same casing rules as `EntityId`. Predicates are free-form strings
/// (we don't impose a fixed ontology) but they canonicalise so that
/// "likes", "Likes", "LIKES " all collapse to one key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Predicate(String);

impl Predicate {
    pub fn from_raw(raw: &str) -> Self {
        let mut out = String::with_capacity(raw.len());
        let mut last_space = false;
        for c in raw.chars() {
            if c.is_alphanumeric() || c == '_' {
                for lc in c.to_lowercase() {
                    out.push(lc);
                }
                last_space = false;
            } else if c.is_whitespace() || c == '-' {
                if !last_space && !out.is_empty() {
                    out.push('_');
                }
                last_space = true;
            }
        }
        while out.ends_with('_') {
            out.pop();
        }
        Predicate(out)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The right-hand side of a triple. Either another entity in the graph
/// (creating an edge) or a literal value (date, number, free string).
///
/// Serialised as `{"kind": "...", "value": "..."}` so the payload is
/// compact and still round-trips cleanly for every variant (the
/// internally-tagged form can't hold a non-object newtype).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FactObject {
    Entity(EntityId),
    Literal(String),
    Date(String),
    Number(String), // kept as string so we don't lose precision on parse
}

impl FactObject {
    pub fn as_display(&self) -> &str {
        match self {
            FactObject::Entity(e) => e.as_str(),
            FactObject::Literal(s) | FactObject::Date(s) | FactObject::Number(s) => s,
        }
    }
}

/// A single (subject, predicate, object) triple extracted from a chunk,
/// with provenance metadata so the query path can rank by recency and
/// confidence, and cite the source when the downstream LLM needs it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    pub subject: EntityId,
    pub predicate: Predicate,
    pub object: FactObject,
    /// LLM-reported confidence in [0.0, 1.0]. Missing → assumed 0.7.
    pub confidence: f32,
    /// Chunks this fact was extracted from. Multiple entries possible if
    /// the same fact was seen in several chunks (evidence accumulates).
    pub source_chunk_ids: Vec<String>,
    /// When the fact is about (event date) — not when extracted.
    pub when: Option<String>,
    /// When the fact was extracted (for recency scoring).
    pub extracted_at: String,
}

impl Fact {
    /// Deterministic id based on (subject, predicate, object, when).
    /// Re-extracting the same triple in the same temporal context yields
    /// the same id, making ingestion idempotent and evidence merging
    /// trivial: the store sees a known id and merges source lists.
    pub fn compute_id(
        subject: &EntityId,
        predicate: &Predicate,
        object: &FactObject,
        when: Option<&str>,
    ) -> FactId {
        let mut h = Sha256::new();
        h.update(subject.as_str().as_bytes());
        h.update(b"|");
        h.update(predicate.as_str().as_bytes());
        h.update(b"|");
        match object {
            FactObject::Entity(e) => {
                h.update(b"e:");
                h.update(e.as_str().as_bytes());
            }
            FactObject::Literal(s) => {
                h.update(b"l:");
                h.update(s.as_bytes());
            }
            FactObject::Date(s) => {
                h.update(b"d:");
                h.update(s.as_bytes());
            }
            FactObject::Number(s) => {
                h.update(b"n:");
                h.update(s.as_bytes());
            }
        }
        h.update(b"|");
        h.update(when.unwrap_or("").as_bytes());
        let digest = h.finalize();
        let mut hex = String::with_capacity(16);
        for b in &digest[..8] {
            hex.push_str(&format!("{:02x}", b));
        }
        FactId(hex)
    }

    /// Build a fact with a computed id and default confidence if none is
    /// supplied. The caller should still prefer to pass the LLM's actual
    /// confidence — 0.7 is the neutral fallback.
    pub fn new(
        subject: EntityId,
        predicate: Predicate,
        object: FactObject,
        confidence: Option<f32>,
        source_chunk_ids: Vec<String>,
        when: Option<String>,
        extracted_at: String,
    ) -> Self {
        let id = Self::compute_id(&subject, &predicate, &object, when.as_deref());
        Self {
            id,
            subject,
            predicate,
            object,
            confidence: confidence.unwrap_or(0.7).clamp(0.0, 1.0),
            source_chunk_ids,
            when,
            extracted_at,
        }
    }
}

/// Deterministic short hash id used to dedup facts at ingestion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FactId(String);

impl FactId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_normalises_casing_and_spaces() {
        assert_eq!(EntityId::from_name("Alice").as_str(), "alice");
        assert_eq!(EntityId::from_name("  ALICE  ").as_str(), "alice");
        assert_eq!(EntityId::from_name("Alice Smith").as_str(), "alice smith");
        assert_eq!(EntityId::from_name("Alice   Smith").as_str(), "alice smith");
    }

    #[test]
    fn entity_id_strips_punctuation() {
        assert_eq!(EntityId::from_name("Alice!").as_str(), "alice");
        assert_eq!(EntityId::from_name("Alice's dog").as_str(), "alices dog");
        assert_eq!(EntityId::from_name("New York, NY.").as_str(), "new york ny");
    }

    #[test]
    fn entity_id_preserves_hyphens_and_underscores() {
        assert_eq!(EntityId::from_name("Marie-Louise").as_str(), "marie-louise");
        assert_eq!(EntityId::from_name("foo_bar").as_str(), "foo_bar");
    }

    #[test]
    fn entity_id_empty_input_is_empty() {
        assert!(EntityId::from_name("").is_empty());
        assert!(EntityId::from_name("   ").is_empty());
        assert!(EntityId::from_name("!!!").is_empty());
    }

    #[test]
    fn predicate_normalises_spaces_to_underscore() {
        assert_eq!(Predicate::from_raw("likes").as_str(), "likes");
        assert_eq!(Predicate::from_raw("moved to").as_str(), "moved_to");
        assert_eq!(Predicate::from_raw("works-for").as_str(), "works_for");
        assert_eq!(Predicate::from_raw("Is A").as_str(), "is_a");
    }

    #[test]
    fn predicate_collapses_multiple_spaces() {
        assert_eq!(Predicate::from_raw("moved   to").as_str(), "moved_to");
    }

    #[test]
    fn entity_add_alias_accepts_normalised_match() {
        let mut e = Entity::new("Alice", "2024-01-01T00:00:00Z");
        assert!(e.add_alias("alice"));
        assert!(e.add_alias("ALICE"));
        assert!(!e.add_alias("Bob"));
        assert_eq!(e.aliases.len(), 2);
    }

    #[test]
    fn entity_add_alias_rejects_self_name() {
        let mut e = Entity::new("Alice", "2024-01-01T00:00:00Z");
        assert!(!e.add_alias("Alice"));
    }

    #[test]
    fn fact_id_is_deterministic() {
        let s = EntityId::from_name("Alice");
        let p = Predicate::from_raw("likes");
        let o = FactObject::Literal("ginger snaps".to_string());
        let id1 = Fact::compute_id(&s, &p, &o, Some("2024-01-01"));
        let id2 = Fact::compute_id(&s, &p, &o, Some("2024-01-01"));
        assert_eq!(id1, id2);
    }

    #[test]
    fn fact_id_differs_on_object_change() {
        let s = EntityId::from_name("Alice");
        let p = Predicate::from_raw("likes");
        let id1 = Fact::compute_id(
            &s,
            &p,
            &FactObject::Literal("ginger snaps".to_string()),
            None,
        );
        let id2 = Fact::compute_id(
            &s,
            &p,
            &FactObject::Literal("sugar cookies".to_string()),
            None,
        );
        assert_ne!(id1, id2);
    }

    #[test]
    fn fact_id_differs_on_timestamp_change() {
        let s = EntityId::from_name("Alice");
        let p = Predicate::from_raw("visited");
        let o = FactObject::Entity(EntityId::from_name("Paris"));
        let id1 = Fact::compute_id(&s, &p, &o, Some("2023"));
        let id2 = Fact::compute_id(&s, &p, &o, Some("2024"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn fact_new_clamps_confidence() {
        let f_high = Fact::new(
            EntityId::from_name("X"),
            Predicate::from_raw("is"),
            FactObject::Literal("y".to_string()),
            Some(2.0),
            vec![],
            None,
            "2024".to_string(),
        );
        assert_eq!(f_high.confidence, 1.0);
        let f_low = Fact::new(
            EntityId::from_name("X"),
            Predicate::from_raw("is"),
            FactObject::Literal("y".to_string()),
            Some(-0.5),
            vec![],
            None,
            "2024".to_string(),
        );
        assert_eq!(f_low.confidence, 0.0);
    }

    #[test]
    fn fact_new_default_confidence() {
        let f = Fact::new(
            EntityId::from_name("X"),
            Predicate::from_raw("is"),
            FactObject::Literal("y".to_string()),
            None,
            vec![],
            None,
            "2024".to_string(),
        );
        assert_eq!(f.confidence, 0.7);
    }

    #[test]
    fn fact_object_as_display_round_trip() {
        assert_eq!(
            FactObject::Entity(EntityId::from_name("Alice")).as_display(),
            "alice"
        );
        assert_eq!(
            FactObject::Literal("hello".to_string()).as_display(),
            "hello"
        );
        assert_eq!(
            FactObject::Date("2024-01-01".to_string()).as_display(),
            "2024-01-01"
        );
        assert_eq!(FactObject::Number("42".to_string()).as_display(), "42");
    }

    #[test]
    fn serde_roundtrip_entity() {
        let mut e = Entity::new("Alice", "2024-01-01T00:00:00Z");
        e.add_alias("alice");
        e.type_hint = Some(EntityType::Person);
        let s = serde_json::to_string(&e).unwrap();
        let back: Entity = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn serde_roundtrip_fact() {
        let f = Fact::new(
            EntityId::from_name("Alice"),
            Predicate::from_raw("visited"),
            FactObject::Entity(EntityId::from_name("Paris")),
            Some(0.9),
            vec!["chunk-42".to_string()],
            Some("2023-06".to_string()),
            "2024-01-01T00:00:00Z".to_string(),
        );
        let s = serde_json::to_string(&f).unwrap();
        let back: Fact = serde_json::from_str(&s).unwrap();
        assert_eq!(f, back);
    }
}
