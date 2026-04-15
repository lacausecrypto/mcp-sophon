use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::message::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIndex {
    pub embeddings: Vec<EmbeddedChunk>,
    pub keyword_index: HashMap<String, Vec<String>>,
    pub topic_clusters: Vec<TopicCluster>,
}

impl Default for SemanticIndex {
    fn default() -> Self {
        Self {
            embeddings: Vec::new(),
            keyword_index: HashMap::new(),
            topic_clusters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub message_ids: Vec<String>,
    pub summary: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCluster {
    pub topic: String,
    pub message_ids: Vec<String>,
    pub summary: String,
}

pub fn build_index(messages: &[Message]) -> SemanticIndex {
    let mut embeddings = Vec::new();
    let mut keyword_index: HashMap<String, Vec<String>> = HashMap::new();
    let mut topic_buckets: HashMap<String, Vec<&Message>> = HashMap::new();

    for message in messages {
        let summary = summarize_message(&message.content);
        embeddings.push(EmbeddedChunk {
            message_ids: vec![message.id.clone()],
            summary,
            embedding: embed_text(&message.content),
        });

        for keyword in extract_keywords(&message.content) {
            keyword_index
                .entry(keyword)
                .or_default()
                .push(message.id.clone());
        }

        for topic in infer_topics(&message.content) {
            topic_buckets.entry(topic).or_default().push(message);
        }
    }

    let topic_clusters = topic_buckets
        .into_iter()
        .map(|(topic, msgs)| TopicCluster {
            topic,
            message_ids: msgs.iter().map(|m| m.id.clone()).collect(),
            summary: msgs
                .iter()
                .take(3)
                .map(|m| summarize_message(&m.content))
                .collect::<Vec<_>>()
                .join(" | "),
        })
        .collect();

    SemanticIndex {
        embeddings,
        keyword_index,
        topic_clusters,
    }
}

pub fn search_index(index: &SemanticIndex, query: &str, top_k: usize) -> Vec<String> {
    let query_embedding = embed_text(query);

    let mut scored = index
        .embeddings
        .iter()
        .map(|chunk| {
            (
                chunk.message_ids.clone(),
                cosine_similarity(&query_embedding, &chunk.embedding),
            )
        })
        .collect::<Vec<_>>();

    // NaN scores are sunk to the bottom so they can never outrank real matches.
    // (Previously `unwrap_or(Ordering::Equal)` silently accepted NaN.)
    scored.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.partial_cmp(&a.1).expect("non-NaN f32 compare"),
    });

    scored
        .into_iter()
        .take(top_k)
        .flat_map(|(ids, _)| ids)
        .collect()
}

fn summarize_message(content: &str) -> String {
    content
        .split_terminator(['.', '!', '?'])
        .next()
        .unwrap_or(content)
        .trim()
        .chars()
        .take(180)
        .collect::<String>()
}

fn extract_keywords(content: &str) -> Vec<String> {
    let lowered = content.to_lowercase();
    lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 4)
        .map(|s| s.to_string())
        .collect()
}

fn infer_topics(content: &str) -> Vec<String> {
    let lowered = content.to_lowercase();
    let mut topics = Vec::new();

    if ["code", "function", "compile", "error", "bug"]
        .iter()
        .any(|kw| lowered.contains(kw))
    {
        topics.push("coding".to_string());
    }
    if ["project", "build", "architecture", "feature"]
        .iter()
        .any(|kw| lowered.contains(kw))
    {
        topics.push("project".to_string());
    }
    if ["name", "prefer", "always", "never"]
        .iter()
        .any(|kw| lowered.contains(kw))
    {
        topics.push("preferences".to_string());
    }

    if topics.is_empty() {
        topics.push("general".to_string());
    }

    topics
}

fn embed_text(text: &str) -> Vec<f32> {
    const DIM: usize = 384;
    let mut vec = vec![0f32; DIM];

    for token in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let mut hash = 0u64;
        for b in token.as_bytes() {
            hash = hash
                .wrapping_mul(1099511628211)
                .wrapping_add(*b as u64 + 1469598103934665603);
        }
        let idx = (hash as usize) % DIM;
        vec[idx] += 1.0;
    }

    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }

    vec
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}
