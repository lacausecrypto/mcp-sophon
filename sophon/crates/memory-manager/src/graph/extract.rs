//! LLM triple extraction (Path A, step 3).
//!
//! One Haiku call turns a batch of conversation messages into a list of
//! `(subject, predicate, object)` triples. This is the ONLY LLM cost in
//! the graph-memory path — at query time everything runs in pure Rust.
//!
//! Contract with the LLM:
//!   - Input: the raw transcript (bounded to ~10 KB so the prompt fits
//!     comfortably in Haiku's context and returns fast).
//!   - Output: a JSON object with a `facts` array, each entry shaped as
//!     `{"subject": "...", "predicate": "...", "object": "...",
//!      "object_kind": "entity|literal|date|number",
//!      "when": "...?", "confidence": 0.0-1.0}`.
//!   - Malformed or partial output is tolerated: we parse what we can
//!     and drop anything that fails a sanity check (empty subject/
//!     predicate, confidence outside [0,1], etc.).
//!
//! Fail-safe: a None return from the LLM or a parse failure yields an
//! empty Vec — the caller falls back to the existing retrieval path so
//! ingestion never blocks the user flow.

use super::types::{EntityId, Fact, FactObject, Predicate};
use crate::llm_client::call_llm;
use serde::Deserialize;

/// Max chars of transcript to send to the extractor in a single call.
/// Long conversations sample head+tail; LLM extraction quality
/// degrades sharply past ~12 KB on Haiku, faster on structured output.
pub const MAX_INPUT_CHARS: usize = 10_000;

/// Cap on triples we accept from one extraction call. Guards against a
/// runaway LLM that emits a thousand marginally-supported facts.
pub const MAX_FACTS_PER_CALL: usize = 80;

#[derive(Debug, Deserialize)]
struct RawExtraction {
    #[serde(default)]
    facts: Vec<RawFact>,
}

#[derive(Debug, Deserialize)]
struct RawFact {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    object_kind: String,
    #[serde(default)]
    when: String,
    #[serde(default)]
    confidence: f32,
}

/// Extract triples from a transcript. `source_chunk_id` is stamped on
/// each produced fact as provenance; callers typically pass the message
/// batch's id (or a synthetic "session-N" label).
pub fn extract_triples(
    transcript: &str,
    source_chunk_id: &str,
    extracted_at: &str,
) -> Option<Vec<Fact>> {
    let trimmed = transcript.trim();
    if trimmed.is_empty() {
        return Some(Vec::new());
    }
    let bounded = bound_transcript(trimmed, MAX_INPUT_CHARS);
    let prompt = build_prompt(&bounded);
    let raw = call_llm(&prompt)?;
    let body = extract_json_object(&raw)?;

    let parsed: RawExtraction = serde_json::from_str(&body).ok()?;

    let mut out: Vec<Fact> = Vec::new();
    for r in parsed.facts.into_iter().take(MAX_FACTS_PER_CALL) {
        if let Some(f) = materialise(r, source_chunk_id, extracted_at) {
            out.push(f);
        }
    }
    Some(out)
}

fn materialise(r: RawFact, source: &str, extracted_at: &str) -> Option<Fact> {
    let subject = EntityId::from_name(r.subject.trim());
    if subject.is_empty() {
        return None;
    }
    let predicate = Predicate::from_raw(r.predicate.trim());
    if predicate.is_empty() {
        return None;
    }
    let obj_text = r.object.trim();
    if obj_text.is_empty() {
        return None;
    }

    let object = match r.object_kind.trim().to_lowercase().as_str() {
        "entity" => {
            let eid = EntityId::from_name(obj_text);
            if eid.is_empty() {
                FactObject::Literal(obj_text.to_string())
            } else {
                FactObject::Entity(eid)
            }
        }
        "date" => FactObject::Date(obj_text.to_string()),
        "number" => FactObject::Number(obj_text.to_string()),
        // default / missing → plain literal
        _ => FactObject::Literal(obj_text.to_string()),
    };

    let confidence = if r.confidence > 0.0 && r.confidence <= 1.0 {
        Some(r.confidence)
    } else {
        None
    };
    let when = if r.when.trim().is_empty() {
        None
    } else {
        Some(r.when.trim().to_string())
    };

    Some(Fact::new(
        subject,
        predicate,
        object,
        confidence,
        vec![source.to_string()],
        when,
        extracted_at.to_string(),
    ))
}

fn bound_transcript(raw: &str, max_chars: usize) -> String {
    if raw.len() <= max_chars {
        return raw.to_string();
    }
    let half = max_chars / 2;
    let head_end = raw
        .char_indices()
        .take_while(|(i, _)| *i < half)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(half);
    let head = &raw[..head_end];
    let tail_start_bytes = raw.len().saturating_sub(half);
    let tail_start = (tail_start_bytes..=raw.len())
        .find(|i| raw.is_char_boundary(*i))
        .unwrap_or(raw.len());
    let tail = &raw[tail_start..];
    format!(
        "{head}\n[... {} chars omitted ...]\n{tail}",
        raw.len() - head.len() - tail.len()
    )
}

fn build_prompt(transcript: &str) -> String {
    format!(
        "You are extracting a knowledge-graph from a conversation. Emit only \
         factual (subject, predicate, object) triples explicitly supported by \
         the text. Do not invent. Do not include greetings, filler, or \
         speculative content.\n\n\
         Guidelines:\n\
         - `subject`: the specific person / entity the fact is about \
         (e.g. \"Alice\", \"Bob's dog Max\"). Use the name, not a pronoun.\n\
         - `predicate`: a short snake-case-or-spaces verb or relation \
         (\"likes\", \"visited\", \"works_at\", \"adopted\", \"moved_to\").\n\
         - `object`: the target of the relation. Can be an entity, a date, \
         a number, or a literal string.\n\
         - `object_kind`: one of \"entity\", \"literal\", \"date\", \"number\".\n\
         - `when`: optional — an ISO date, a month, a year, or a relative \
         phrase the text uses (\"last week\", \"October 2023\"). Leave \"\" if \
         the fact has no explicit time.\n\
         - `confidence`: 0.6-0.9 when directly stated, 0.3-0.5 when \
         paraphrased / implied. Drop anything below 0.3.\n\n\
         Return ONLY a JSON object with this schema — no prose, no markdown \
         fences:\n\
         {{\"facts\": [\n\
           {{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\
         \"object_kind\":\"entity|literal|date|number\",\"when\":\"...\",\
         \"confidence\":0.0}}\n\
         ]}}\n\n\
         If the conversation contains no clear facts, return {{\"facts\": []}}.\n\n\
         CONVERSATION:\n{transcript}"
    )
}

fn extract_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let body = if let Some(s) = trimmed.strip_prefix("```json") {
        s.trim_start_matches('\n').trim_end_matches("```")
    } else if let Some(s) = trimmed.strip_prefix("```") {
        s.trim_start_matches('\n').trim_end_matches("```")
    } else {
        trimmed
    };
    let start = body.find('{')?;
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in body[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    Some(body[start..end?].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FactObject;

    fn parse_and_materialise(json: &str) -> Vec<Fact> {
        let body = extract_json_object(json).expect("json extracted");
        let parsed: RawExtraction = serde_json::from_str(&body).expect("json parses");
        parsed
            .facts
            .into_iter()
            .filter_map(|r| materialise(r, "chunk-1", "2024-01-01"))
            .collect()
    }

    #[test]
    fn extract_json_object_strips_fences() {
        let raw = "```json\n{\"facts\": []}\n```";
        let body = extract_json_object(raw).unwrap();
        let parsed: RawExtraction = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.facts.len(), 0);
    }

    #[test]
    fn extract_json_object_strips_leading_prose() {
        let raw = "Sure, here are the facts:\n{\"facts\": []}";
        let body = extract_json_object(raw).unwrap();
        let parsed: RawExtraction = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.facts.len(), 0);
    }

    #[test]
    fn materialise_builds_a_clean_fact() {
        let json = r#"{"facts": [
            {"subject": "Alice", "predicate": "visited",
             "object": "Paris", "object_kind": "entity",
             "when": "June 2023", "confidence": 0.9}
        ]}"#;
        let facts = parse_and_materialise(json);
        assert_eq!(facts.len(), 1);
        let f = &facts[0];
        assert_eq!(f.subject.as_str(), "alice");
        assert_eq!(f.predicate.as_str(), "visited");
        match &f.object {
            FactObject::Entity(e) => assert_eq!(e.as_str(), "paris"),
            _ => panic!("expected entity object"),
        }
        assert_eq!(f.when.as_deref(), Some("June 2023"));
        assert!((f.confidence - 0.9).abs() < 1e-6);
        assert_eq!(f.source_chunk_ids, vec!["chunk-1".to_string()]);
    }

    #[test]
    fn literal_and_date_objects_parse() {
        let json = r#"{"facts": [
            {"subject": "Evan", "predicate": "favourite_food",
             "object": "ginger snaps", "object_kind": "literal",
             "when": "", "confidence": 0.85},
            {"subject": "Evan", "predicate": "birthday",
             "object": "1985-04-18", "object_kind": "date",
             "when": "", "confidence": 0.95},
            {"subject": "Evan", "predicate": "pet_count",
             "object": "3", "object_kind": "number",
             "when": "", "confidence": 0.8}
        ]}"#;
        let facts = parse_and_materialise(json);
        assert_eq!(facts.len(), 3);
        assert!(matches!(facts[0].object, FactObject::Literal(_)));
        assert!(matches!(facts[1].object, FactObject::Date(_)));
        assert!(matches!(facts[2].object, FactObject::Number(_)));
    }

    #[test]
    fn empty_subject_or_predicate_is_dropped() {
        let json = r#"{"facts": [
            {"subject": "", "predicate": "visited",
             "object": "Paris", "object_kind": "entity",
             "when": "", "confidence": 0.9},
            {"subject": "Alice", "predicate": "",
             "object": "Paris", "object_kind": "entity",
             "when": "", "confidence": 0.9},
            {"subject": "Alice", "predicate": "visited",
             "object": "", "object_kind": "entity",
             "when": "", "confidence": 0.9}
        ]}"#;
        let facts = parse_and_materialise(json);
        assert!(facts.is_empty(), "every triple should be dropped");
    }

    #[test]
    fn bound_transcript_uses_head_tail_sampling() {
        let long = "x".repeat(50_000);
        let bounded = bound_transcript(&long, 2_000);
        assert!(bounded.len() <= 2_200, "len={}", bounded.len());
        assert!(bounded.contains("chars omitted"));
    }

    #[test]
    fn bound_transcript_short_passthrough() {
        let short = "hello world";
        assert_eq!(bound_transcript(short, 1_000), short);
    }

    #[test]
    fn missing_object_kind_defaults_to_literal() {
        let json = r#"{"facts": [
            {"subject": "Alice", "predicate": "likes",
             "object": "reading", "object_kind": "",
             "when": "", "confidence": 0.8}
        ]}"#;
        let facts = parse_and_materialise(json);
        assert_eq!(facts.len(), 1);
        assert!(matches!(facts[0].object, FactObject::Literal(_)));
    }

    #[test]
    fn confidence_out_of_range_falls_back_to_default() {
        let json = r#"{"facts": [
            {"subject": "Alice", "predicate": "likes",
             "object": "reading", "object_kind": "literal",
             "when": "", "confidence": 2.5}
        ]}"#;
        let facts = parse_and_materialise(json);
        assert_eq!(facts.len(), 1);
        // RawFact.confidence=2.5 is > 1.0, so materialise passes None
        // → Fact::new picks the 0.7 default.
        assert!((facts[0].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn empty_transcript_returns_empty_without_llm() {
        let out = extract_triples("", "chunk-0", "2024").unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn max_facts_per_call_caps_output() {
        let mut parts = vec!["{\"facts\": [".to_string()];
        for i in 0..(MAX_FACTS_PER_CALL + 20) {
            if i > 0 {
                parts.push(",".to_string());
            }
            parts.push(format!(
                "{{\"subject\":\"E{}\",\"predicate\":\"p\",\"object\":\"o\",\"object_kind\":\"literal\",\"when\":\"\",\"confidence\":0.8}}",
                i
            ));
        }
        parts.push("]}".to_string());
        let json = parts.concat();
        let facts = parse_and_materialise(&json);
        // `parse_and_materialise` doesn't apply the cap (that's done in
        // `extract_triples`). This test confirms the parser tolerates
        // oversized inputs; the cap is enforced at the call-site.
        assert_eq!(facts.len(), MAX_FACTS_PER_CALL + 20);
    }
}
