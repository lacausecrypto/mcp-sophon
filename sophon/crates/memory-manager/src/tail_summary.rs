//! Tail summary — compress the retrieval candidates that DIDN'T make the
//! top-K verbatim slice into a short paragraph, so the downstream LLM
//! still has context on "what else was mentioned in the conversation"
//! without paying the full chunk tokens.
//!
//! Motivation (Rec #1, LongLLMLingua hybrid pattern):
//!
//! Current retrieval returns top-5 chunks as-is. Chunks at rank 6-20 are
//! discarded entirely — they might contain cross-references, disclaimers,
//! or side facts that would help the final LLM. Summarising them into a
//! single paragraph of 100-150 words recovers most of that signal at a
//! fraction of the token cost (one Haiku call instead of 15 verbatim
//! chunks = ~3000 tokens → ~150 tokens).
//!
//! Cost: one Haiku call per query. Latency ~500-700 ms. Fail-safe:
//! returns `None` on any LLM/parse failure so the caller can skip.

use crate::llm_client::call_llm;

/// Max chars shown per tail chunk. Tail chunks are less critical than
/// verbatim top-K, so we truncate aggressively.
const MAX_TAIL_CHUNK_CHARS: usize = 400;

/// Max total input characters. Beyond this we drop tail chunks from the
/// end — we prefer a bounded Haiku call over a potentially-truncated one.
const MAX_TOTAL_INPUT_CHARS: usize = 8_000;

/// Summarise the tail chunks into one short paragraph focusing on
/// question-relevant context (names, dates, entities, events).
///
/// Returns the rendered summary text on success. The caller can then
/// append it to the final context block.
pub fn summarise_tail(query: &str, tail_chunks: &[&str]) -> Option<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() || tail_chunks.is_empty() {
        return None;
    }

    let (listed, kept) = format_tail(tail_chunks);
    if kept == 0 {
        return None;
    }
    let prompt = build_prompt(trimmed, &listed, kept);
    let raw = call_llm(&prompt)?;

    let cleaned = raw.trim();
    if cleaned.is_empty() {
        return None;
    }
    // Strip trailing markdown fence artefacts if present.
    let cleaned = cleaned
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if cleaned.is_empty() {
        return None;
    }
    Some(cleaned.to_string())
}

fn format_tail(chunks: &[&str]) -> (String, usize) {
    let mut listed = String::new();
    let mut used = 0usize;
    let mut kept = 0usize;
    for (i, c) in chunks.iter().enumerate() {
        let snippet: String = c.chars().take(MAX_TAIL_CHUNK_CHARS).collect();
        let flat = snippet.replace('\n', " ");
        let entry = format!("[{}] {}\n", i + 1, flat);
        if used + entry.len() > MAX_TOTAL_INPUT_CHARS && kept > 0 {
            break;
        }
        used += entry.len();
        listed.push_str(&entry);
        kept += 1;
    }
    (listed, kept)
}

fn build_prompt(query: &str, listed: &str, kept: usize) -> String {
    format!(
        "You are compressing retrieval candidates ranked {kept}-th and below \
         (the top-ranked chunks are already shown to the final answerer \
         verbatim). Produce a short paragraph of 80-150 words capturing any \
         names, dates, places, or events from the chunks below that might \
         be useful for answering the QUESTION. Drop filler and small talk. \
         Do NOT invent facts — only restate what's present in the chunks. \
         If nothing in the chunks is relevant, reply with exactly: \
         \"(no tail signal)\".\n\n\
         QUESTION: {query}\n\n\
         CHUNKS:\n{listed}\n\
         COMPRESSED TAIL SUMMARY:"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_returns_none() {
        assert!(summarise_tail("", &["chunk"]).is_none());
        assert!(summarise_tail("   ", &["chunk"]).is_none());
    }

    #[test]
    fn empty_chunks_returns_none() {
        assert!(summarise_tail("what?", &[]).is_none());
    }

    #[test]
    fn format_tail_numbers_chunks() {
        let (listed, kept) = format_tail(&["alpha", "beta"]);
        assert_eq!(kept, 2);
        assert!(listed.contains("[1] alpha"));
        assert!(listed.contains("[2] beta"));
    }

    #[test]
    fn format_tail_bounds_total_input_with_many_chunks() {
        // Each chunk is truncated to MAX_TAIL_CHUNK_CHARS (~400), but the
        // total budget MAX_TOTAL_INPUT_CHARS (8 000) caps how many make
        // it into the prompt. 30 chunks × ~410 chars each = ~12 300 →
        // should drop the tail.
        let long = "x".repeat(20_000);
        let refs: Vec<&str> = (0..30).map(|_| long.as_str()).collect();
        let (listed, kept) = format_tail(&refs);
        assert!(
            kept < refs.len(),
            "expected to drop chunks beyond budget, kept all {}",
            kept
        );
        assert!(
            listed.len() <= MAX_TOTAL_INPUT_CHARS + MAX_TAIL_CHUNK_CHARS + 64,
            "listed len {} exceeds bounds",
            listed.len()
        );
    }

    #[test]
    fn format_tail_truncates_long_chunk() {
        let long = "z".repeat(2000);
        let (listed, _) = format_tail(&[&long]);
        // No 1000-char run of 'z' should survive truncation.
        assert!(!listed.contains(&"z".repeat(1000)));
    }

    #[test]
    fn prompt_includes_question_and_no_tail_marker() {
        let p = build_prompt("Who?", "[1] alpha\n", 1);
        assert!(p.contains("QUESTION: Who?"));
        assert!(p.contains("(no tail signal)"));
        assert!(p.contains("[1] alpha"));
    }
}
