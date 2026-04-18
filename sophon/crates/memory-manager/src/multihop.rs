//! Multi-hop query detection — cheap heuristic to decide when to pay the
//! cost of LLM-based sub-query decomposition.
//!
//! LOCOMO multi-hop items (16/80 at N=80) score 0% across every Sophon
//! condition because a single retrieval pass never surfaces all the chunks
//! needed to connect the dots. But calling Haiku to decompose *every* query
//! would triple the latency of single-hop questions that don't need it.
//!
//! This module returns `true` for queries that show at least one signal
//! consistent with cross-chunk reasoning:
//!
//! - Multiple capitalised entities (2+) — "What did Alice and Bob decide?"
//! - Conjunction between clauses — "X and Y", "both X and Y", "between X and Y"
//! - Temporal chaining — "after", "before", "while", "when X did Y", "then"
//! - Comparative framing — "compare", "difference between", "vs"
//! - Long queries (>14 tokens) — usually carry several constraints
//!
//! False positives waste one Haiku call (~200 ms) and trigger a marginal
//! extra retrieval pass; false negatives keep the status quo. The heuristic
//! is tuned to err on the side of triggering — the LLM decomposer's own
//! output is the final gate (it can return no sub-queries on a trivially
//! single-hop question).

use std::collections::HashSet;

/// Decide whether `query` is likely multi-hop. Cheap, pure, deterministic.
pub fn is_likely_multihop(query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return false;
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let token_count = tokens.len();
    let lower = trimmed.to_lowercase();

    // Signal 1: comparative / conjunctive markers that almost always
    // indicate the answer spans two clauses.
    static STRONG_MARKERS: &[&str] = &[
        "compare",
        "comparison",
        "difference between",
        "differences between",
        "versus",
        " vs ",
        " vs. ",
        "both ",
        "either ",
        "neither ",
        "after ",
        "before ",
        "while ",
        "during ",
        "then ",
        "previously",
        "earlier",
        "and then",
        "as well as",
    ];
    for marker in STRONG_MARKERS {
        if lower.contains(marker) {
            return true;
        }
    }

    // Signal 2: entity count ≥ 2. Capitalised tokens ≥ 3 chars, dedup'd,
    // excluding sentence-initial capitals and common stop-caps.
    let entity_count = count_entities(&tokens);
    if entity_count >= 2 {
        return true;
    }

    // Signal 3: conjoined "and" between content words (not list punctuation).
    // Triggers on "what did X and Y discuss?" but not "yes and no".
    if token_count >= 6 && has_content_and(&tokens) {
        return true;
    }

    // Signal 4: long queries usually carry 2+ constraints.
    if token_count > 14 {
        return true;
    }

    false
}

fn count_entities(tokens: &[&str]) -> usize {
    static STOP_CAPS: &[&str] = &[
        "What", "When", "Where", "Which", "Who", "Whom", "Whose", "How", "Why", "Did", "Do",
        "Does", "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Can", "Could", "Will",
        "Would", "Should", "May", "Might", "Shall", "The", "This", "That", "These", "Those",
        "A", "An", "And", "Or", "But", "If", "Then", "So", "On", "In", "At", "For", "With",
        "To", "From", "By", "Of",
    ];

    let mut seen: HashSet<String> = HashSet::new();
    for (i, raw) in tokens.iter().enumerate() {
        let clean: String = raw
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        if clean.len() < 3 {
            continue;
        }
        let first = clean.chars().next().unwrap_or('a');
        if !first.is_uppercase() {
            continue;
        }
        // The very first token of a query is almost always a sentence
        // starter; discount unless it's already known as a proper noun.
        if i == 0 && STOP_CAPS.iter().any(|s| s.eq_ignore_ascii_case(&clean)) {
            continue;
        }
        if STOP_CAPS.iter().any(|s| s.eq_ignore_ascii_case(&clean)) {
            continue;
        }
        seen.insert(clean.to_lowercase());
    }
    seen.len()
}

fn has_content_and(tokens: &[&str]) -> bool {
    for (i, t) in tokens.iter().enumerate() {
        if t.eq_ignore_ascii_case("and") && i > 1 && i < tokens.len() - 1 {
            // At least one token before and one after must not be a stop word.
            let prev = tokens[i - 1].to_lowercase();
            let next = tokens[i + 1].to_lowercase();
            let is_filler = |w: &str| {
                matches!(
                    w,
                    "yes" | "no" | "ok" | "sure" | "maybe" | "please" | "also" | "too"
                )
            };
            if !is_filler(&prev) && !is_filler(&next) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_single_hop_not_multihop() {
        assert!(!is_likely_multihop("What is your name?"));
        assert!(!is_likely_multihop("How are you?"));
        assert!(!is_likely_multihop("Tell me a joke."));
    }

    #[test]
    fn two_entities_detected() {
        assert!(is_likely_multihop(
            "What did Alice tell Bob about the project?"
        ));
    }

    #[test]
    fn temporal_chaining_detected() {
        assert!(is_likely_multihop("What happened after the meeting?"));
        assert!(is_likely_multihop("Who did she call before leaving?"));
        assert!(is_likely_multihop("What was said while he was away?"));
    }

    #[test]
    fn comparative_detected() {
        assert!(is_likely_multihop("Compare the two approaches."));
        assert!(is_likely_multihop("What's the difference between A and B?"));
    }

    #[test]
    fn conjoined_and_detected() {
        assert!(is_likely_multihop(
            "What does the user think about React and Vue?"
        ));
    }

    #[test]
    fn long_query_triggers() {
        let q = "List every single thing the user mentioned about their hobbies during the last couple of conversations we had";
        assert!(is_likely_multihop(q));
    }

    #[test]
    fn filler_and_not_triggered() {
        assert!(!is_likely_multihop("yes and no"));
        assert!(!is_likely_multihop("ok and thanks"));
    }

    #[test]
    fn empty_and_whitespace_safe() {
        assert!(!is_likely_multihop(""));
        assert!(!is_likely_multihop("   "));
    }
}
