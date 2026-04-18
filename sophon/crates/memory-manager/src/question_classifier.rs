//! Adaptive compression budget: classify a question as "factual recall" or
//! "general" and let the handler pick an appropriate retrieval budget.
//!
//! Rationale (LOCOMO v0.3.0 failure trace):
//! - Adversarial questions score 83 % at ctx ≈ 1200 tokens — more context
//!   would HURT (the LLM hallucinates confident wrong answers when it sees
//!   tangentially related chunks).
//! - Single-hop / temporal factual questions score 0–33 % at the same
//!   budget — the answer chunk is often retrieved at rank 6-10, past the
//!   top-5 cutoff, so the LLM never sees it.
//!
//! One budget cannot fit both. An adaptive controller classifies the query
//! once via Haiku and tells the handler whether to stay tight (general,
//! top_k=5, retrieval cap=1000) or open up (factual, top_k=10, cap=2000).
//!
//! Cost: 1 Haiku call per `compress_history` when `SOPHON_ADAPTIVE=1`.
//! Latency overhead ~200 ms. Failure-safe: if the LLM call fails or
//! returns garbage, the classifier returns `None` and the caller defaults
//! to the conservative (general) mode.

use crate::llm_client::call_llm;

/// Compression mode picked per query. The handler maps this to concrete
/// (top_k, max_retrieved_tokens) values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionMode {
    /// Question asks for a specific fact (date, name, number, place,
    /// quantity) buried somewhere in the conversation. Needs wider
    /// retrieval to bring the answer chunk into view.
    FactualRecall,
    /// Open-ended, adversarial, or conversational. Tight context is fine
    /// and often better (less hallucination surface).
    General,
}

/// Classify `query` via a single Haiku call. Returns `None` on any LLM or
/// parse failure — callers should treat that as a signal to stay in the
/// default (General) mode.
pub fn classify_question(query: &str) -> Option<QuestionMode> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }

    let prompt = build_prompt(trimmed);
    let raw = call_llm(&prompt)?;
    parse_mode(&raw)
}

fn build_prompt(query: &str) -> String {
    format!(
        "Classify the retrieval question below into ONE of two categories:\n\n\
         - FACTUAL_RECALL: asks for a specific fact (date, name, place, \
         quantity, event time, count, person-name) that must be located in \
         one or two specific chunks of a long conversation. Examples: \
         \"When did Alice move to Paris?\", \"How many pets does Bob have?\", \
         \"What was the restaurant name?\".\n\
         - GENERAL: open-ended (why, how, what kind of), adversarial (assumes \
         a fact that may not be in the history), or high-level reasoning. \
         Examples: \"Why does Tim find LeBron inspiring?\", \"Is Alice \
         happy?\", \"What do they discuss most?\".\n\n\
         Default to GENERAL when ambiguous. Only pick FACTUAL_RECALL when the \
         question's exact answer is a short token (date, name, count, place).\n\n\
         Return ONLY one of these two words, exactly: FACTUAL_RECALL or GENERAL.\n\
         No explanation, no punctuation, no JSON.\n\n\
         QUESTION: {query}"
    )
}

fn parse_mode(raw: &str) -> Option<QuestionMode> {
    // Be generous: case-insensitive match on the keywords, ignore any
    // markdown fences or trailing prose the LLM might add.
    let upper = raw.to_uppercase();
    let has_factual = upper.contains("FACTUAL_RECALL") || upper.contains("FACTUAL RECALL");
    let has_general = upper.contains("GENERAL");

    if has_factual && !has_general {
        return Some(QuestionMode::FactualRecall);
    }
    if has_general && !has_factual {
        return Some(QuestionMode::General);
    }
    // Both or neither: ambiguous — bail and let the caller default.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_factual_recall_bare() {
        assert_eq!(parse_mode("FACTUAL_RECALL"), Some(QuestionMode::FactualRecall));
    }

    #[test]
    fn parses_general_bare() {
        assert_eq!(parse_mode("GENERAL"), Some(QuestionMode::General));
    }

    #[test]
    fn parses_with_space_variant() {
        assert_eq!(
            parse_mode("FACTUAL RECALL"),
            Some(QuestionMode::FactualRecall)
        );
    }

    #[test]
    fn parses_with_leading_prose() {
        assert_eq!(
            parse_mode("Sure, this is a FACTUAL_RECALL question."),
            Some(QuestionMode::FactualRecall)
        );
    }

    #[test]
    fn parses_lowercase() {
        assert_eq!(parse_mode("general"), Some(QuestionMode::General));
    }

    #[test]
    fn ambiguous_returns_none() {
        // Both tokens present
        assert_eq!(
            parse_mode("could be FACTUAL_RECALL or GENERAL"),
            None
        );
    }

    #[test]
    fn neither_returns_none() {
        assert_eq!(parse_mode("some random output"), None);
        assert_eq!(parse_mode(""), None);
    }

    #[test]
    fn empty_query_returns_none() {
        assert_eq!(classify_question(""), None);
        assert_eq!(classify_question("   "), None);
    }

    #[test]
    fn prompt_lists_both_categories_and_default() {
        let p = build_prompt("anything");
        assert!(p.contains("FACTUAL_RECALL"));
        assert!(p.contains("GENERAL"));
        assert!(p.contains("Default to GENERAL"));
    }
}
