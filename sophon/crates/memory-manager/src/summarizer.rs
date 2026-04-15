use crate::{
    fact_extractor::extract_facts,
    index::{build_index, SemanticIndex},
    message::{CompressedMemory, Message, Role},
};
use sophon_core::tokens::count_tokens;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub max_tokens: usize,
    pub recent_window: usize,
    pub compression_threshold: usize,
    pub use_llm_summarization: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            recent_window: 5,
            compression_threshold: 10,
            use_llm_summarization: false,
        }
    }
}

/// Compress conversation history into memory.
///
/// Short histories are passed through unchanged: if the total raw tokens of
/// the messages already fit inside `max_tokens`, we skip summarization and
/// return all messages in `recent_messages` with an empty summary. This
/// avoids the pathological case where a 4-message chat produces a compressed
/// payload *larger* than its input (observed as a v1 limitation).
pub fn compress_history(messages: &[Message], config: &MemoryConfig) -> CompressedMemory {
    if messages.is_empty() {
        return CompressedMemory {
            summary: "No conversation yet.".to_string(),
            stable_facts: vec![],
            recent_messages: vec![],
            index: SemanticIndex::default(),
            token_count: count_tokens("No conversation yet."),
            original_message_count: 0,
        };
    }

    // Pass-through guard: if the raw history already fits in the budget, the
    // compressor cannot make it smaller — returning the messages as-is is
    // strictly better than synthesizing a summary and an index.
    let raw_tokens: usize = messages.iter().map(|m| m.token_count).sum();
    let overhead_floor = 16; // approximate minimum cost of JSON scaffolding
    if raw_tokens + overhead_floor <= config.max_tokens
        && messages.len() < config.compression_threshold
    {
        return CompressedMemory {
            summary: String::new(),
            stable_facts: vec![],
            recent_messages: messages.to_vec(),
            index: SemanticIndex::default(),
            token_count: raw_tokens,
            original_message_count: messages.len(),
        };
    }

    let keep_recent = config.recent_window.min(messages.len());
    let split_idx = messages.len().saturating_sub(keep_recent);
    let (older, recent) = messages.split_at(split_idx);

    let stable_facts = extract_facts(messages);
    let summary = if messages.len() < config.compression_threshold {
        summarize_recent(messages)
    } else {
        heuristic_summarize(older)
    };

    let mut compressed = CompressedMemory {
        summary,
        stable_facts,
        recent_messages: recent.to_vec(),
        index: build_index(messages),
        token_count: 0,
        original_message_count: messages.len(),
    };

    enforce_budget(&mut compressed, config.max_tokens);

    // Final safety net: if the final compressed payload is still larger than
    // the raw message set, fall back to the pass-through form. This can happen
    // when `enforce_budget` can't shrink further but the summary+index still
    // dominate a tiny conversation.
    if compressed.token_count > raw_tokens && messages.len() < config.compression_threshold {
        return CompressedMemory {
            summary: String::new(),
            stable_facts: vec![],
            recent_messages: messages.to_vec(),
            index: SemanticIndex::default(),
            token_count: raw_tokens,
            original_message_count: messages.len(),
        };
    }

    compressed
}

/// Heuristic summarization (no LLM required).
pub fn heuristic_summarize(messages: &[Message]) -> String {
    if messages.is_empty() {
        return "No previous messages to summarize.".to_string();
    }

    let mut user_questions = Vec::new();
    let mut assistant_points = Vec::new();
    let mut code_mentions = 0usize;

    for message in messages {
        if message.content.contains("```") {
            code_mentions += 1;
        }

        match message.role {
            Role::User => {
                if message.content.contains('?') {
                    user_questions.push(first_sentence(&message.content));
                }
            }
            Role::Assistant => {
                assistant_points.push(first_sentence(&message.content));
            }
            Role::System => {}
        }
    }

    let mut out = String::new();
    if !user_questions.is_empty() {
        out.push_str("User asked about: ");
        out.push_str(&user_questions.into_iter().take(6).collect::<Vec<_>>().join("; "));
        out.push_str(". ");
    }

    if !assistant_points.is_empty() {
        out.push_str("Assistant responses covered: ");
        out.push_str(
            &assistant_points
                .into_iter()
                .take(6)
                .collect::<Vec<_>>()
                .join("; "),
        );
        out.push_str(". ");
    }

    if code_mentions > 0 {
        out.push_str(&format!("Conversation included {} code-oriented message(s).", code_mentions));
    }

    if out.trim().is_empty() {
        "Conversation contains mixed discussion with no clear dominant thread.".to_string()
    } else {
        out.trim().to_string()
    }
}

fn first_sentence(content: &str) -> String {
    content
        .split_terminator(['.', '!', '?'])
        .next()
        .unwrap_or(content)
        .trim()
        .chars()
        .take(200)
        .collect()
}

fn summarize_recent(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .take(5)
        .rev()
        .map(|m| format!("{:?}: {}", m.role, first_sentence(&m.content)))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn enforce_budget(memory: &mut CompressedMemory, max_tokens: usize) {
    let mut summary = memory.summary.clone();
    let mut facts = memory.stable_facts.clone();

    loop {
        let facts_text = facts
            .iter()
            .filter(|f| !f.superseded)
            .map(|f| f.content.clone())
            .collect::<Vec<_>>()
            .join("\n");

        let recent_text = memory
            .recent_messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n");

        let token_count = count_tokens(&format!("{}\n{}\n{}", summary, facts_text, recent_text));
        if token_count <= max_tokens {
            memory.summary = summary;
            memory.stable_facts = facts;
            memory.token_count = token_count;
            break;
        }

        if summary.len() > 120 {
            summary.truncate(summary.len().saturating_sub(80));
            summary.push_str("...");
            continue;
        }

        if !facts.is_empty() {
            facts.pop();
            continue;
        }

        if !memory.recent_messages.is_empty() {
            memory.recent_messages.remove(0);
            continue;
        }

        memory.token_count = token_count;
        break;
    }
}
