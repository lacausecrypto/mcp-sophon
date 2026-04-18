//! HyDE-style query rewriting: generate 2-3 hypothetical answer statements
//! for a question, then retrieve against each (plus the original) and fuse
//! the rankings via RRF.
//!
//! Motivation (LOCOMO failure trace, v0.3.1 bench):
//!
//! ```text
//!   Question : "What is Evan's favorite food?"
//!   Gold     : "Ginger snaps are my weakness for sure!"
//!   Overlap  : 0 shared content words
//! ```
//!
//! The question and the answer share zero vocabulary. HashEmbedder (keyword
//! cosine) and BM25 (IDF-weighted keyword) both need lexical overlap to
//! score the answer chunk. Neither can bridge "favorite food" ↔ "weakness
//! for ginger snaps".
//!
//! HyDE (Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
//! Relevance Labels") flips this: instead of searching with the *question*
//! vocabulary, search with the *hypothetical answer* vocabulary. An LLM
//! writes 2-3 plausible answer statements; each becomes a query. The
//! answer's real vocabulary is far more likely to appear in those
//! hypotheticals than in the question.
//!
//! - **Zero ML at runtime**: the LLM call is an out-of-process shell-out to
//!   `claude -p haiku` (or whatever `SOPHON_LLM_CMD` points at), consistent
//!   with Sophon's positioning — no model weights loaded in the binary.
//! - **One call per query**: latency budget is the LLM round-trip, not N.
//! - **Works with any retriever**: the rewrites are plain text, fed through
//!   the same `Retriever::retrieve` surface. Composable with BM25 hybrid
//!   (P1) and multi-hop decomposition (P0).

use crate::llm_client::call_llm;
use serde::Deserialize;

/// Max rewrites to honour from the LLM. Beyond 5 we cap — latency grows
/// linearly with N because each rewrite triggers one retrieval pass.
pub const MAX_REWRITES: usize = 5;

/// Minimum rewrites to consider the LLM output useful. A single rewrite
/// is usually a paraphrase of the question, which adds little signal.
pub const MIN_REWRITES: usize = 2;

#[derive(Debug, Deserialize)]
struct RewriteOutput {
    #[serde(default)]
    rewrites: Vec<String>,
}

/// Ask the LLM to produce 2-3 hypothetical answer statements for `query`.
/// Returns `Some(Vec<String>)` with usable rewrites, or `None` on any
/// failure — callers fall back to plain single-query retrieval.
pub fn hyde_rewrite_query(query: &str) -> Option<Vec<String>> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }
    let prompt = build_prompt(trimmed);
    let raw = call_llm(&prompt)?;
    let body = extract_json_object(&raw)?;
    let parsed: RewriteOutput = serde_json::from_str(&body).ok()?;

    let q_lower = trimmed.to_lowercase();
    let mut rewrites: Vec<String> = parsed
        .rewrites
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.len() <= 500)
        .filter(|s| s.to_lowercase() != q_lower)
        .collect();

    rewrites.truncate(MAX_REWRITES);
    if rewrites.len() < MIN_REWRITES {
        return None;
    }
    Some(rewrites)
}

fn build_prompt(query: &str) -> String {
    format!(
        "You are a retrieval query rewriter. Given a QUESTION, write 3 short \
         hypothetical answer statements that *could* be the answer found in a \
         conversation. The goal is to generate text that uses the vocabulary the \
         real answer would use — NOT to paraphrase the question.\n\n\
         Guidelines:\n\
         - Each rewrite is 1 sentence, 5-20 words, in first-person or \
         third-person depending on the question form.\n\
         - Use varied vocabulary across the 3 rewrites (synonyms, related \
         phrases, idioms).\n\
         - Do NOT copy the question's wording.\n\
         - If the question is about a person's preference, weakness, habit, or \
         emotion, include that framing.\n\n\
         Return ONLY a JSON object with this schema — no prose, no markdown fences:\n\
         {{\"rewrites\": [\"...\", \"...\", \"...\"]}}\n\n\
         QUESTION: {query}"
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
    let end = end?;
    Some(body[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json() {
        let raw = r#"{"rewrites": ["ans1", "ans2", "ans3"]}"#;
        let body = extract_json_object(raw).unwrap();
        let parsed: RewriteOutput = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.rewrites.len(), 3);
    }

    #[test]
    fn parses_markdown_fence() {
        let raw = "```json\n{\"rewrites\": [\"x\", \"y\"]}\n```";
        let body = extract_json_object(raw).unwrap();
        let parsed: RewriteOutput = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.rewrites.len(), 2);
    }

    #[test]
    fn parses_with_leading_prose() {
        let raw = "Sure, here are rewrites:\n{\"rewrites\": [\"a\", \"b\"]}";
        let body = extract_json_object(raw).unwrap();
        let parsed: RewriteOutput = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.rewrites, vec!["a", "b"]);
    }

    #[test]
    fn empty_query_returns_none() {
        assert!(hyde_rewrite_query("").is_none());
        assert!(hyde_rewrite_query("   ").is_none());
    }

    #[test]
    fn missing_json_returns_none() {
        assert!(extract_json_object("no braces here").is_none());
    }

    #[test]
    fn unbalanced_braces_returns_none() {
        assert!(extract_json_object("{\"rewrites\": [\"x\"").is_none());
    }
}
