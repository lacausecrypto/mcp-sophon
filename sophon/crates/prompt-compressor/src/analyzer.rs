use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;

#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub topics: Vec<String>,
    pub confidence: f32,
    pub complexity: Complexity,
    pub trigger_keywords: Vec<String>,
    /// Raw lexical terms extracted from the query (lowercased, camelCase
    /// and snake_case split, stopwords dropped). Unlike `topics` — which
    /// route through a frozen keyword dictionary and miss anything
    /// domain-specific — these are the *actual* query words, used for
    /// BM25 section scoring so a query mentioning `compute_section_scores`
    /// can pull in the section that defines it even when no dictionary
    /// topic matched.
    pub query_terms: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Simple,
    Medium,
    Complex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

static TOPIC_KEYWORDS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    HashMap::from([
        (
            "coding",
            vec!["code", "function", "debug", "bug", "refactor", "compile"],
        ),
        ("python", vec!["python", "pandas", "numpy", "py"]),
        (
            "javascript",
            vec!["javascript", "typescript", "node", "react", "frontend"],
        ),
        (
            "math",
            vec![
                "math",
                "equation",
                "formula",
                "proof",
                "calculate",
                "integral",
            ],
        ),
        (
            "creative_writing",
            vec!["story", "poem", "novel", "creative", "character"],
        ),
        (
            "data_analysis",
            vec![
                "dataset",
                "csv",
                "table",
                "statistics",
                "analyze",
                "analysis",
            ],
        ),
        (
            "conversation",
            vec!["hello", "chat", "talk", "explain", "help me"],
        ),
        (
            "safety",
            vec!["safe", "harm", "danger", "security", "exploit", "malware"],
        ),
    ])
});

/// Analyze a user query to determine relevant topics.
pub fn analyze_query(query: &str, history: Option<&[ConversationMessage]>) -> QueryAnalysis {
    let normalized = query.to_lowercase();
    let mut topic_scores: HashMap<String, usize> = HashMap::new();
    let mut trigger_keywords = HashSet::new();

    for (topic, keywords) in TOPIC_KEYWORDS.iter() {
        for keyword in keywords {
            if normalized.contains(keyword) {
                *topic_scores.entry((*topic).to_string()).or_insert(0) += 1;
                trigger_keywords.insert((*keyword).to_string());
            }
        }
    }

    if let Some(messages) = history {
        // Bias toward continuity: recent messages contribute low-weight topic hints.
        for message in messages.iter().rev().take(4) {
            let content = message.content.to_lowercase();
            for (topic, keywords) in TOPIC_KEYWORDS.iter() {
                if keywords.iter().any(|kw| content.contains(kw)) {
                    *topic_scores.entry((*topic).to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    let mut scored_topics = topic_scores.into_iter().collect::<Vec<_>>();
    scored_topics.sort_by(|a, b| b.1.cmp(&a.1));

    let mut topics = scored_topics
        .iter()
        .take(4)
        .map(|(topic, _)| topic.clone())
        .collect::<Vec<_>>();

    if topics.is_empty() {
        topics.push("conversation".to_string());
    }

    let confidence = if scored_topics.is_empty() {
        0.3
    } else {
        let top_score = scored_topics[0].1 as f32;
        let total_score = scored_topics
            .iter()
            .map(|(_, score)| *score as f32)
            .sum::<f32>();
        (top_score / total_score).clamp(0.0, 1.0)
    };

    let complexity = infer_complexity(query);

    QueryAnalysis {
        topics,
        confidence,
        complexity,
        trigger_keywords: trigger_keywords.into_iter().collect(),
        query_terms: tokenize_terms(query),
    }
}

/// Generic English function words dropped from lexical terms — they
/// carry no retrieval signal. Kept deliberately small so short domain
/// terms (`io`, `db`, `os`) survive.
static STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "for", "on",
    "with", "that", "this", "it", "as", "be", "by", "at", "from", "how", "what", "why", "when",
    "where", "which", "who", "do", "does", "did", "can", "could", "should", "would", "will", "i",
    "you", "we", "they", "my", "me", "your", "our", "please", "help", "about", "into", "out", "up",
    "down", "if", "then", "else", "not", "no", "yes",
];

/// Tokenize text into lowercase lexical terms for query-aware scoring.
///
/// Splits on non-alphanumeric boundaries **and** camelCase humps so that
/// `computeSectionScores`, `compute_section_scores` and
/// `compute-section-scores` all yield `[compute, section, scores]`. This
/// is what lets the lexical scorer match identifiers regardless of the
/// casing convention used in the query versus the prompt. Terms shorter
/// than 2 chars and stopwords are dropped.
pub fn tokenize_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    // Was the previous char a lowercase letter or digit? Used to detect
    // the lower→Upper transition that marks a camelCase word boundary.
    let mut prev_lower_or_digit = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if prev_lower_or_digit && ch.is_uppercase() {
                push_term(&mut terms, &mut current);
            }
            current.push(ch);
            prev_lower_or_digit = ch.is_lowercase() || ch.is_numeric();
        } else {
            push_term(&mut terms, &mut current);
            prev_lower_or_digit = false;
        }
    }
    push_term(&mut terms, &mut current);
    terms
}

fn push_term(terms: &mut Vec<String>, current: &mut String) {
    if current.is_empty() {
        return;
    }
    let term = std::mem::take(current).to_lowercase();
    if term.chars().count() >= 2 && !STOPWORDS.contains(&term.as_str()) {
        terms.push(term);
    }
}

fn infer_complexity(query: &str) -> Complexity {
    let token_count = count_tokens(query);
    let lower = query.to_lowercase();

    let has_multi_step = lower.contains(" then ")
        || lower.contains("after that")
        || lower.contains("step by step")
        || lower.contains("compare")
        || lower.contains("tradeoff")
        || lower.contains("architecture");

    let punctuation_density = query
        .chars()
        .filter(|c| [',', ';', ':', '?', '!'].contains(c))
        .count();

    if token_count <= 20 && punctuation_density <= 1 && !has_multi_step {
        Complexity::Simple
    } else if token_count > 80 || punctuation_density >= 4 || has_multi_step {
        Complexity::Complex
    } else {
        Complexity::Medium
    }
}
