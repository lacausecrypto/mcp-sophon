//! LLM batch reranking of retrieval candidates.
//!
//! Motivation: BM25 + HashEmbedder rank chunks by lexical/keyword overlap.
//! On LOCOMO-style QA, the chunk that actually contains the answer often
//! ranks 6-12 — past the top-5 cutoff the caller uses. A single Haiku
//! call that scores N candidates against the query (1-10 scale) can
//! re-order them so the answer chunk lands in the visible slice.
//!
//! This is the "query-aware reranking" step from Rec #1 of the v0.3.1
//! bench audit. It intentionally does NOT compute Haiku per-chunk
//! perplexities (as LongLLMLingua does with open-weight models) — the
//! Claude CLI does not expose token log-probs. Instead we ask Haiku to
//! emit a compact JSON score array over all candidates in one call.
//!
//! Cost: one Haiku round-trip. Latency ~500-800 ms for ~20 chunks.
//! Fail-safe: if parsing fails, returns `None` and the caller keeps the
//! original ranking.

use crate::llm_client::call_llm;
use serde::Deserialize;

/// Hard cap on chunks passed to one rerank call. Beyond this we'd start
/// bumping into Haiku's output-token budget or latency budget.
pub const MAX_CHUNKS_PER_RERANK: usize = 20;

/// Max characters per chunk shown to the reranker. Long chunks are
/// truncated — the first ~500 chars are plenty for a relevance judgement.
pub const MAX_CHUNK_CHARS: usize = 500;

#[derive(Debug, Deserialize)]
struct RerankOutput {
    #[serde(default)]
    scores: Vec<f32>,
}

/// Score `chunks` for relevance to `query` via one Haiku call. Returns a
/// vector of the same length with scores in [0, 10] (higher = more
/// relevant), or `None` on any LLM or parse failure.
///
/// The caller combines these scores with the existing RRF ranking — we
/// don't sort here, to keep the function pure and composable.
pub fn rerank_chunks(query: &str, chunks: &[&str]) -> Option<Vec<f32>> {
    let trimmed = query.trim();
    if trimmed.is_empty() || chunks.is_empty() {
        return None;
    }
    let n = chunks.len().min(MAX_CHUNKS_PER_RERANK);

    let prompt = build_prompt(trimmed, &chunks[..n]);
    let raw = call_llm(&prompt)?;
    let body = extract_json_object(&raw)?;

    let parsed: RerankOutput = serde_json::from_str(&body).ok()?;
    if parsed.scores.len() != n {
        // Contract violation: score count must match chunk count.
        return None;
    }
    // Clamp to [0,10] — the LLM sometimes drifts.
    let mut out = parsed.scores;
    for s in out.iter_mut() {
        *s = s.clamp(0.0, 10.0);
    }
    // If the caller passed more than MAX chunks, pad the rest with a
    // neutral 5.0 so downstream code can still use a 1:1 mapping. This
    // keeps the caller's algorithm simple without forcing it to branch
    // on truncation.
    while out.len() < chunks.len() {
        out.push(5.0);
    }
    Some(out)
}

fn build_prompt(query: &str, chunks: &[&str]) -> String {
    let mut listed = String::new();
    for (i, c) in chunks.iter().enumerate() {
        let snippet: String = c.chars().take(MAX_CHUNK_CHARS).collect();
        // Replace newlines to keep the numbered list tight and readable.
        let flat = snippet.replace('\n', " ");
        listed.push_str(&format!("[{}] {}\n", i + 1, flat));
    }

    format!(
        "You are scoring chunks of a long conversation for their relevance \
         to a specific retrieval question. Return a JSON array of numbers, \
         one per chunk, in [0, 10].\n\n\
         Scoring rubric:\n\
         - 10 = chunk literally contains the answer to the question.\n\
         - 7-9 = chunk discusses the same entity/event as the answer.\n\
         - 4-6 = chunk mentions the topic in passing.\n\
         - 1-3 = loosely related (same person, different event).\n\
         - 0 = irrelevant.\n\n\
         Be strict: most chunks should score below 5. Reserve 8+ for chunks \
         you believe the final answer will come from.\n\n\
         Return ONLY this JSON schema, no prose, no markdown fences, no text \
         outside the JSON:\n\
         {{\"scores\": [s1, s2, ..., sN]}}\n\
         The array length MUST equal {n}.\n\n\
         QUESTION: {query}\n\n\
         CHUNKS:\n{listed}",
        n = chunks.len()
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

    fn parse(json: &str, expected_len: usize) -> Option<Vec<f32>> {
        let body = extract_json_object(json)?;
        let parsed: RerankOutput = serde_json::from_str(&body).ok()?;
        if parsed.scores.len() != expected_len {
            return None;
        }
        Some(parsed.scores)
    }

    #[test]
    fn parses_plain_json() {
        let raw = r#"{"scores": [8.0, 2.5, 6.0]}"#;
        let scores = parse(raw, 3).unwrap();
        assert_eq!(scores, vec![8.0, 2.5, 6.0]);
    }

    #[test]
    fn parses_markdown_fence() {
        let raw = "```json\n{\"scores\": [10, 0]}\n```";
        let scores = parse(raw, 2).unwrap();
        assert_eq!(scores, vec![10.0, 0.0]);
    }

    #[test]
    fn parses_with_leading_prose() {
        let raw = "Scoring now:\n{\"scores\": [5, 5, 5]}";
        assert_eq!(parse(raw, 3).unwrap(), vec![5.0; 3]);
    }

    #[test]
    fn length_mismatch_returns_none() {
        let raw = r#"{"scores": [1, 2]}"#;
        assert!(parse(raw, 3).is_none());
    }

    #[test]
    fn malformed_json_returns_none() {
        assert!(parse("not json", 3).is_none());
        assert!(parse(r#"{"scores": [1,"#, 3).is_none());
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(rerank_chunks("", &["chunk"]).is_none());
        assert!(rerank_chunks("q", &[]).is_none());
    }

    #[test]
    fn prompt_numbers_chunks_from_1() {
        let p = build_prompt("test?", &["alpha", "beta"]);
        assert!(p.contains("[1] alpha"));
        assert!(p.contains("[2] beta"));
        assert!(p.contains("QUESTION: test?"));
    }

    #[test]
    fn prompt_truncates_long_chunks() {
        let long = "a".repeat(2000);
        let p = build_prompt("q", &[&long]);
        assert!(!p.contains(&"a".repeat(1000)));
    }

    #[test]
    fn prompt_constrains_array_length() {
        let p = build_prompt("q", &["x", "y", "z"]);
        assert!(p.contains("length MUST equal 3"));
    }
}
