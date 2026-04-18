//! LLM-based query decomposition for multi-hop retrieval.
//!
//! Motivation: LOCOMO multi-hop items score 0% across every Sophon
//! retrieval condition at N=80. The failure mode is that a single top-k
//! pass can't surface the full chain of facts — "What did Alice tell Bob
//! before the trip?" needs chunks about Alice, about Bob, and about the
//! trip, scattered across 3+ sessions with different vocabulary.
//!
//! Fix: shell out to Haiku once per multi-hop query, ask for 2-3
//! independent sub-queries covering different facets of the question,
//! retrieve each separately, then RRF-fuse the rankings.
//!
//! Contract:
//! - Single LLM call per query (not N) — bounded ~200-500 ms overhead.
//! - Strict JSON output schema, parsed defensively. Malformed output
//!   returns `None` so the caller falls back to single-pass retrieval.
//! - Returns 2-5 sub-queries; the original query is *not* included
//!   (the caller appends it separately if desired).
//! - Determinism is best-effort: we prompt for a stable, compact JSON
//!   payload, but the LLM side is not under our control.

use crate::llm_client::call_llm;
use serde::Deserialize;

/// Max number of sub-queries to honour from the LLM response. Anything
/// beyond 5 is truncated — bounds retrieval latency.
pub const MAX_SUBQUERIES: usize = 5;

/// Lower bound on sub-queries to consider the decomposition useful.
/// A 1-element response means the LLM thought the question was already
/// single-hop; fall back to the direct retrieval path.
pub const MIN_SUBQUERIES: usize = 2;

#[derive(Debug, Deserialize)]
struct DecomposeOutput {
    #[serde(default)]
    subqueries: Vec<String>,
}

/// Ask the LLM to split `query` into 2-3 retrieval-friendly sub-questions.
///
/// Returns `Some(Vec<String>)` with `MIN_SUBQUERIES..=MAX_SUBQUERIES` items
/// when the LLM returned a usable decomposition, `None` otherwise. Callers
/// should fall back to the original query when `None` is returned.
pub fn decompose_query(query: &str) -> Option<Vec<String>> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }

    let prompt = build_prompt(trimmed);
    let raw = call_llm(&prompt)?;

    let cleaned = extract_json_object(&raw)?;
    let parsed: DecomposeOutput = serde_json::from_str(&cleaned).ok()?;

    let mut subs: Vec<String> = parsed
        .subqueries
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.len() <= 400)
        .collect();

    // Drop exact duplicates of the original query — they add no new signal.
    let q_lower = trimmed.to_lowercase();
    subs.retain(|s| s.to_lowercase() != q_lower);

    subs.truncate(MAX_SUBQUERIES);
    if subs.len() < MIN_SUBQUERIES {
        return None;
    }

    Some(subs)
}

fn build_prompt(query: &str) -> String {
    format!(
        "You are a retrieval assistant. Decompose this question into 2-3 independent \
         sub-questions that together cover the information needed to answer it. Each \
         sub-question must be answerable from a single passage. Focus on the distinct \
         entities, events, or time periods involved.\n\n\
         Return ONLY a JSON object with this exact schema, no prose, no markdown fences:\n\
         {{\"subqueries\": [\"...\", \"...\", \"...\"]}}\n\n\
         If the question is already single-hop and needs no decomposition, return an \
         empty list: {{\"subqueries\": []}}\n\n\
         QUESTION: {query}"
    )
}

/// Strip common LLM wrappers (markdown fences, leading prose) to extract
/// the first balanced JSON object. Returns the object string without the
/// surrounding noise.
fn extract_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();

    // Strip markdown code fences if present.
    let body = if let Some(stripped) = trimmed.strip_prefix("```json") {
        stripped.trim_start_matches('\n').trim_end_matches("```")
    } else if let Some(stripped) = trimmed.strip_prefix("```") {
        stripped.trim_start_matches('\n').trim_end_matches("```")
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
    fn extracts_plain_json() {
        let raw = r#"{"subqueries": ["a", "b"]}"#;
        let cleaned = extract_json_object(raw).unwrap();
        let parsed: DecomposeOutput = serde_json::from_str(&cleaned).unwrap();
        assert_eq!(parsed.subqueries, vec!["a", "b"]);
    }

    #[test]
    fn extracts_from_markdown_fence() {
        let raw = "```json\n{\"subqueries\": [\"x\"]}\n```";
        let cleaned = extract_json_object(raw).unwrap();
        let parsed: DecomposeOutput = serde_json::from_str(&cleaned).unwrap();
        assert_eq!(parsed.subqueries, vec!["x"]);
    }

    #[test]
    fn extracts_from_generic_fence() {
        let raw = "```\n{\"subqueries\": [\"y\"]}\n```";
        let cleaned = extract_json_object(raw).unwrap();
        let parsed: DecomposeOutput = serde_json::from_str(&cleaned).unwrap();
        assert_eq!(parsed.subqueries, vec!["y"]);
    }

    #[test]
    fn extracts_with_leading_prose() {
        let raw = "Sure, here is the decomposition:\n{\"subqueries\": [\"one\", \"two\"]}";
        let cleaned = extract_json_object(raw).unwrap();
        let parsed: DecomposeOutput = serde_json::from_str(&cleaned).unwrap();
        assert_eq!(parsed.subqueries, vec!["one", "two"]);
    }

    #[test]
    fn extract_handles_missing_json() {
        assert!(extract_json_object("no json here").is_none());
    }

    #[test]
    fn extract_handles_unbalanced_braces() {
        assert!(extract_json_object("{\"subqueries\": [\"a\"").is_none());
    }

    #[test]
    fn empty_query_returns_none() {
        assert!(decompose_query("").is_none());
        assert!(decompose_query("   ").is_none());
    }
}
