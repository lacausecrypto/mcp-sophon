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
