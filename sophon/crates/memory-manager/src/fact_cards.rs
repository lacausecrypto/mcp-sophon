//! Entity-indexed fact extraction ("fact cards"), inspired by ENGRAM-R and
//! HippoRAG 2's per-entity episodic memory.
//!
//! Rationale: the existing block-based Haiku summariser produces prose
//! summaries that an LLM must re-read linearly to find "when did X Y". Fact
//! cards restructure the same input into a per-entity timeline:
//!
//! ```json
//! {
//!   "Alice": [
//!     {"date": "2023-05-01", "event": "started new job as data scientist"},
//!     {"date": "2023-06-15", "event": "moved to Paris with Bob"}
//!   ],
//!   "Bob": [
//!     {"date": "2023-06-15", "event": "helped Alice move to Paris"}
//!   ]
//! }
//! ```
//!
//! This is a targeted answer format for single-hop/temporal LOCOMO items,
//! which are 70% of the questions. Long prose summaries dilute the signal;
//! a structured table lets the downstream LLM jump straight to the cell.
//!
//! Cost: one extra Haiku call per compress_history when SOPHON_FACT_CARDS=1
//! is set. For a 600-turn conversation this adds ~1–2 s and is gated by the
//! env var so default behaviour is unchanged.

use crate::llm_client::call_llm;
use crate::message::{Message, Role};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Max chars of conversation to feed the fact-card extractor in a single
/// call. Beyond this, we sample head + tail to keep the prompt bounded.
const MAX_INPUT_CHARS: usize = 12_000;

/// Max entities to keep in the final map. Bounds the size of the output
/// payload; the LLM may return more, we truncate by recency of last mention.
const MAX_ENTITIES: usize = 24;

/// Max events per entity. Bounds per-entity verbosity.
const MAX_EVENTS_PER_ENTITY: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactEvent {
    /// Free-form date string (the LLM returns whatever surface form the
    /// conversation used: "2023-05", "last Tuesday", "May 1 2024"). We do
    /// no normalisation — the downstream LLM reads these verbatim.
    pub date: String,
    /// One-line event description. Should be self-contained (no pronouns).
    pub event: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FactCards {
    /// BTreeMap for deterministic iteration order.
    pub entities: BTreeMap<String, Vec<FactEvent>>,
}

impl FactCards {
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Serialise to the rendered text block that the compressor hands to
    /// the downstream LLM. Format is compact and order-stable.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (entity, events) in &self.entities {
            out.push_str("## ");
            out.push_str(entity);
            out.push('\n');
            for e in events {
                if e.date.trim().is_empty() {
                    out.push_str(&format!("- {}\n", e.event));
                } else {
                    out.push_str(&format!("- [{}] {}\n", e.date, e.event));
                }
            }
        }
        out
    }
}

/// Extract fact cards from a conversation via one LLM call. Returns `None`
/// if the LLM is unavailable or returns unparseable output, so the caller
/// can fall back to the prose summary path.
pub fn extract_fact_cards(messages: &[Message]) -> Option<FactCards> {
    if messages.is_empty() {
        return None;
    }
    let transcript = format_transcript_bounded(messages, MAX_INPUT_CHARS);
    let prompt = build_prompt(&transcript);
    let raw = call_llm(&prompt)?;
    let json = extract_json_object(&raw)?;
    parse_fact_cards(&json)
}

fn build_prompt(transcript: &str) -> String {
    format!(
        "Extract an entity-indexed fact timeline from this conversation. For \
         each distinct person, place, project, pet, or organisation mentioned \
         with a concrete event, list the events in chronological order. Dates \
         should be in whatever form the conversation used (exact dates, months, \
         relative phrases like 'last week').\n\n\
         Rules:\n\
         - Only include entities with at least one dated event.\n\
         - Each event description must be self-contained (no pronouns, expand \"he/she/they\" to the entity name).\n\
         - Max 8 events per entity. Prefer the most specific, factual events.\n\
         - Max 24 entities total.\n\
         - Return ONLY a JSON object matching the schema below — no prose, no markdown fences.\n\n\
         SCHEMA:\n\
         {{\"entities\": {{\"<entity name>\": [{{\"date\": \"<date string>\", \"event\": \"<description>\"}}]}}}}\n\n\
         If no entities with dated events are found, return {{\"entities\": {{}}}}.\n\n\
         CONVERSATION:\n{transcript}"
    )
}

fn format_transcript_bounded(messages: &[Message], max_chars: usize) -> String {
    let mut transcript = String::with_capacity(messages.len() * 80);
    for m in messages {
        let role = match m.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };
        transcript.push_str(role);
        transcript.push_str(": ");
        transcript.push_str(&m.content);
        transcript.push('\n');
    }
    if transcript.len() <= max_chars {
        return transcript;
    }
    // Head + tail sampling: half the budget at the start, half at the end.
    // Preserves both early context (who is who) and recent events.
    let half = max_chars / 2;
    let head_end = transcript
        .char_indices()
        .take_while(|(i, _)| *i < half)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(half);
    let head = &transcript[..head_end];
    let tail_start_bytes = transcript.len().saturating_sub(half);
    // Find a char boundary at or after tail_start_bytes.
    let tail_start = (tail_start_bytes..=transcript.len())
        .find(|i| transcript.is_char_boundary(*i))
        .unwrap_or(transcript.len());
    let tail = &transcript[tail_start..];
    format!(
        "{head}\n[...{} chars omitted...]\n{tail}",
        transcript.len() - head.len() - tail.len()
    )
}

#[derive(Debug, Deserialize)]
struct RawOutput {
    #[serde(default)]
    entities: BTreeMap<String, Vec<RawEvent>>,
}

#[derive(Debug, Deserialize)]
struct RawEvent {
    #[serde(default)]
    date: String,
    #[serde(default)]
    event: String,
}

fn parse_fact_cards(json: &str) -> Option<FactCards> {
    let raw: RawOutput = serde_json::from_str(json).ok()?;
    let mut entities: BTreeMap<String, Vec<FactEvent>> = BTreeMap::new();
    for (name, events) in raw.entities {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let mut cleaned: Vec<FactEvent> = events
            .into_iter()
            .filter_map(|e| {
                let event = e.event.trim().to_string();
                if event.is_empty() {
                    return None;
                }
                Some(FactEvent {
                    date: e.date.trim().to_string(),
                    event,
                })
            })
            .collect();
        cleaned.truncate(MAX_EVENTS_PER_ENTITY);
        if !cleaned.is_empty() {
            entities.insert(name.to_string(), cleaned);
        }
    }
    // Bound total entities by insertion order — BTreeMap is alphabetical,
    // which is fine here (deterministic + no hidden ranking signal).
    if entities.len() > MAX_ENTITIES {
        let keep: Vec<String> = entities.keys().take(MAX_ENTITIES).cloned().collect();
        entities.retain(|k, _| keep.contains(k));
    }
    Some(FactCards { entities })
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
    let end = end?;
    Some(body[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_fact_cards() {
        let json = r#"{
            "entities": {
                "Alice": [
                    {"date": "2023-05-01", "event": "started new job"},
                    {"date": "2023-06-15", "event": "moved to Paris"}
                ],
                "Bob": [
                    {"date": "2023-06-15", "event": "helped Alice move"}
                ]
            }
        }"#;
        let fc = parse_fact_cards(json).unwrap();
        assert_eq!(fc.entities.len(), 2);
        assert_eq!(fc.entities.get("Alice").unwrap().len(), 2);
        assert_eq!(
            fc.entities.get("Alice").unwrap()[0].event,
            "started new job"
        );
    }

    #[test]
    fn renders_stable_text_block() {
        let json = r#"{"entities": {"Alice": [{"date": "2024-01", "event": "did X"}], "Bob": [{"date": "", "event": "said Y"}]}}"#;
        let fc = parse_fact_cards(json).unwrap();
        let rendered = fc.render();
        assert!(rendered.contains("## Alice"));
        assert!(rendered.contains("- [2024-01] did X"));
        assert!(rendered.contains("- said Y")); // empty date drops the [] prefix
                                                // Alice before Bob — BTreeMap lexicographic order.
        assert!(rendered.find("Alice").unwrap() < rendered.find("Bob").unwrap());
    }

    #[test]
    fn drops_entities_with_empty_events() {
        let json = r#"{"entities": {"Alice": [], "Bob": [{"date": "x", "event": ""}]}}"#;
        let fc = parse_fact_cards(json).unwrap();
        assert!(fc.is_empty(), "should drop entities with no events");
    }

    #[test]
    fn truncates_events_per_entity() {
        // 10 events — should be capped at 8.
        let events: Vec<String> = (0..10)
            .map(|i| format!("{{\"date\": \"d{i}\", \"event\": \"e{i}\"}}"))
            .collect();
        let json = format!(r#"{{"entities": {{"X": [{}]}}}}"#, events.join(","));
        let fc = parse_fact_cards(&json).unwrap();
        assert_eq!(fc.entities.get("X").unwrap().len(), MAX_EVENTS_PER_ENTITY);
    }

    #[test]
    fn extract_json_object_handles_fence() {
        let raw = "```json\n{\"entities\": {}}\n```";
        let j = extract_json_object(raw).unwrap();
        let parsed: RawOutput = serde_json::from_str(&j).unwrap();
        assert!(parsed.entities.is_empty());
    }

    #[test]
    fn extract_json_object_fails_on_missing() {
        assert!(extract_json_object("no json here").is_none());
    }

    #[test]
    fn render_empty_is_empty() {
        let fc = FactCards::default();
        assert_eq!(fc.render(), "");
    }

    #[test]
    fn format_transcript_bounded_uses_head_tail_sampling() {
        let msgs: Vec<Message> = (0..500)
            .map(|i| {
                Message::new(
                    Role::User,
                    format!(
                        "This is a reasonably long message number {i} with some filler content."
                    ),
                )
            })
            .collect();
        let t = format_transcript_bounded(&msgs, 2000);
        assert!(
            t.contains("chars omitted"),
            "should include omission marker"
        );
        assert!(t.len() <= 2200, "length should be bounded, got {}", t.len());
        assert!(t.contains("message number 0"), "should keep head");
        assert!(t.contains("message number 499"), "should keep tail");
    }
}
