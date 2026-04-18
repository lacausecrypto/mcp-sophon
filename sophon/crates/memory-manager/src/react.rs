//! ReAct-lite iterative retrieval for multi-hop questions.
//!
//! Motivation: LOCOMO multi-hop items stay at 0 % accuracy across every
//! single-pass retrieval variant (HashEmbedder, BM25 hybrid, HyDE,
//! fact cards). The failure trace is consistent: the answer is spread
//! across 2-3 chunks that share entities but use different vocabulary,
//! and no single retrieval query surfaces them all together.
//!
//! Inspired by HippoRAG 2 and the ReAct pattern (Yao et al. 2023), this
//! module implements a two-call LLM loop:
//!
//! 1. Retrieve top-k with the original query.
//! 2. Ask Haiku to inspect the chunks: do they contain the answer, or is
//!    something still missing? If missing, propose a *specific* follow-up
//!    query that would find it.
//! 3. If the LLM asks for more, retrieve with the follow-up, fuse the
//!    rankings via RRF, and repeat — up to `max_rounds` (default 3).
//!
//! Cost bounding:
//! - At most `max_rounds - 1` decide calls per query (2 by default).
//! - At most `max_rounds` retrieval passes per query (3 by default).
//! - Gated by `is_likely_multihop(query)` so single-hop questions take
//!   the fast path with zero overhead.

use crate::llm_client::call_llm;
use serde::Deserialize;

/// Decision emitted by the LLM at each ReAct step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReactDecision {
    /// Current chunks already contain the answer. Stop iterating.
    HasAnswer,
    /// Specific follow-up query to retrieve next, drawn from the
    /// information gap the LLM identified.
    FollowUp(String),
    /// LLM output unparseable or empty — caller should stop.
    Unknown,
}

/// Max characters per chunk shown to the decider, to bound prompt size.
/// Long chunks are truncated head-only. A 30-chunk prompt at 600 char/chunk
/// is ~18 KB — well within Haiku's context budget.
const MAX_CHUNK_CHARS: usize = 600;

/// Ask the LLM whether the provided chunks contain the answer, and if not,
/// what specific sub-query would find the missing piece.
///
/// Returns `Unknown` on any failure so the caller can stop the loop
/// gracefully.
pub fn react_decide(query: &str, chunk_texts: &[&str]) -> ReactDecision {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return ReactDecision::Unknown;
    }

    let prompt = build_prompt(trimmed, chunk_texts);
    let Some(raw) = call_llm(&prompt) else {
        return ReactDecision::Unknown;
    };

    let Some(body) = extract_json_object(&raw) else {
        return ReactDecision::Unknown;
    };

    let parsed: Result<DecisionOutput, _> = serde_json::from_str(&body);
    match parsed {
        Ok(d) => d.to_decision(),
        Err(_) => ReactDecision::Unknown,
    }
}

#[derive(Debug, Deserialize)]
struct DecisionOutput {
    #[serde(default)]
    has_answer: bool,
    #[serde(default)]
    follow_up: String,
}

impl DecisionOutput {
    fn to_decision(self) -> ReactDecision {
        if self.has_answer {
            return ReactDecision::HasAnswer;
        }
        let f = self.follow_up.trim();
        if f.is_empty() {
            ReactDecision::Unknown
        } else {
            ReactDecision::FollowUp(f.to_string())
        }
    }
}

fn build_prompt(query: &str, chunks: &[&str]) -> String {
    let mut listed = String::new();
    for (i, c) in chunks.iter().enumerate() {
        let snippet: String = c.chars().take(MAX_CHUNK_CHARS).collect();
        listed.push_str(&format!("[{}] {}\n", i + 1, snippet));
    }

    format!(
        "You are planning the NEXT retrieval query for a multi-hop question about a \
         long conversation. Inspect the chunks already retrieved and propose a \
         follow-up query that would surface the SPECIFIC missing fact.\n\n\
         QUESTION: {query}\n\n\
         RETRIEVED CHUNKS:\n{listed}\n\
         Analyse: does the question ask for a date, name, place, quantity, or \
         relationship? Do the chunks literally contain that exact fact (not a \
         similar-looking one)? Default to assuming the answer is MISSING unless you \
         can quote the exact gold value verbatim from a chunk.\n\n\
         Reply with one of:\n\
         - If the chunks verbatim contain the exact answer:\n\
           {{\"has_answer\": true}}\n\
         - Otherwise (preferred): name the missing fact and write a 5-15 word \
         retrieval query that would find it. Reuse specific entity names from the \
         QUESTION (not paraphrases), and target a DIFFERENT angle than the original \
         query. For temporal questions, add a date/time lexical target (e.g. \
         \"adopted\", \"arrived\", \"started\", \"in 2023\"). For entity links, add \
         the other entity's name.\n\
           {{\"has_answer\": false, \"follow_up\": \"...\"}}\n\n\
         Return ONLY the JSON object — no prose, no markdown fences."
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

    fn parse(json: &str) -> ReactDecision {
        let body = extract_json_object(json).unwrap_or_default();
        let parsed: Result<DecisionOutput, _> = serde_json::from_str(&body);
        parsed.map(|d| d.to_decision()).unwrap_or(ReactDecision::Unknown)
    }

    #[test]
    fn parses_has_answer_true() {
        let raw = r#"{"has_answer": true}"#;
        assert_eq!(parse(raw), ReactDecision::HasAnswer);
    }

    #[test]
    fn parses_follow_up() {
        let raw = r#"{"has_answer": false, "follow_up": "When did Shadow arrive?"}"#;
        assert_eq!(
            parse(raw),
            ReactDecision::FollowUp("When did Shadow arrive?".to_string())
        );
    }

    #[test]
    fn parses_from_markdown_fence() {
        let raw = "```json\n{\"has_answer\": true}\n```";
        assert_eq!(parse(raw), ReactDecision::HasAnswer);
    }

    #[test]
    fn parses_with_leading_prose() {
        let raw = "Based on my analysis,\n{\"has_answer\": false, \"follow_up\": \"search X\"}";
        assert_eq!(
            parse(raw),
            ReactDecision::FollowUp("search X".to_string())
        );
    }

    #[test]
    fn empty_follow_up_is_unknown() {
        let raw = r#"{"has_answer": false, "follow_up": ""}"#;
        assert_eq!(parse(raw), ReactDecision::Unknown);
    }

    #[test]
    fn malformed_json_is_unknown() {
        assert_eq!(parse("{\"has_answer\": tr"), ReactDecision::Unknown);
        assert_eq!(parse("no json here"), ReactDecision::Unknown);
    }

    #[test]
    fn empty_query_returns_unknown() {
        assert_eq!(react_decide("", &[]), ReactDecision::Unknown);
        assert_eq!(react_decide("   ", &[]), ReactDecision::Unknown);
    }

    #[test]
    fn prompt_includes_chunks_and_query() {
        let p = build_prompt("What?", &["chunk one content", "chunk two content"]);
        assert!(p.contains("QUESTION: What?"));
        assert!(p.contains("[1] chunk one content"));
        assert!(p.contains("[2] chunk two content"));
        assert!(p.contains("\"has_answer\""));
    }

    #[test]
    fn prompt_default_assumes_missing() {
        let p = build_prompt("q", &["chunk"]);
        assert!(p.contains("Default to assuming the answer is MISSING"));
    }

    #[test]
    fn prompt_truncates_long_chunks() {
        let long = "a".repeat(2000);
        let p = build_prompt("q", &[&long]);
        // No single line should contain 2000 'a's.
        assert!(!p.contains(&"a".repeat(1000)));
    }
}
