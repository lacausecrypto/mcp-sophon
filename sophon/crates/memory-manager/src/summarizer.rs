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
#[tracing::instrument(
    skip_all,
    fields(
        messages = messages.len(),
        max_tokens = config.max_tokens,
        recent_window = config.recent_window,
    ),
)]
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

    // Adaptive recent window: scale with conversation length so that
    // longer histories keep more recent context (log₂ growth). A 500-
    // message conversation gets ~14 recent messages instead of the
    // fixed 5, giving the compressor more signal without blowing up
    // the token budget (the budget enforcer handles overflow later).
    let adaptive_window = if config.recent_window > 0 {
        let base = config.recent_window;
        let log_bonus = (messages.len() as f64).log2().floor() as usize;
        base.max(log_bonus.max(base))
    } else {
        0
    };
    let keep_recent = adaptive_window.min(messages.len());
    let split_idx = messages.len().saturating_sub(keep_recent);
    let (older, recent) = messages.split_at(split_idx);

    let stable_facts = extract_facts(messages);
    // Activate LLM summarization if either the config flag is set OR the
    // SOPHON_LLM_CMD env var is present (implicit opt-in — you wouldn't
    // set the command without wanting to use it). SOPHON_NO_LLM_SUMMARY=1
    // is an explicit opt-out, useful when the LLM command is shared with
    // other features (HyDE, query decomposer, fact cards) and the caller
    // wants only the heuristic summary here — typically in bench harnesses.
    let opt_out = std::env::var("SOPHON_NO_LLM_SUMMARY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let use_llm =
        !opt_out && (config.use_llm_summarization || std::env::var("SOPHON_LLM_CMD").is_ok());
    let summary = if use_llm {
        let target = if messages.len() < config.compression_threshold {
            messages
        } else {
            older
        };
        llm_summarize(target).unwrap_or_else(|| {
            // Fallback to heuristic if LLM call fails
            if messages.len() < config.compression_threshold {
                summarize_recent(messages)
            } else {
                heuristic_summarize(older)
            }
        })
    } else if messages.len() < config.compression_threshold {
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
        out.push_str(
            &user_questions
                .into_iter()
                .take(6)
                .collect::<Vec<_>>()
                .join("; "),
        );
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
        out.push_str(&format!(
            "Conversation included {} code-oriented message(s).",
            code_mentions
        ));
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

/// LLM-backed abstractive summarization. Shells out to the command in
/// `SOPHON_LLM_CMD` (default: `claude -p --model haiku`). Returns
/// `None` on any failure so the caller can fall back to the heuristic.
///
/// **Block-based approach** (v0.2.2): instead of truncating the
/// conversation to 4000 chars (which loses everything after message
/// ~25), we:
///
/// 1. Split messages into blocks of `BLOCK_SIZE` (30 messages each).
/// 2. Summarize each block independently via the LLM (~50-80 words
///    per block summary).
/// 3. If total summaries exceed `MAX_SUMMARY_CHARS`, run a second
///    "summarize the summaries" pass to condense further.
///
/// This ensures every message in a 600-turn conversation contributes
/// to the final summary — the root cause of the N=40 COMP_LLM gap
/// (messages 30-600 were invisible with the old 4000-char truncation).
fn llm_summarize(messages: &[Message]) -> Option<String> {
    const BLOCK_SIZE: usize = 30;
    const MAX_SUMMARY_CHARS: usize = 3000;

    if messages.is_empty() {
        return None;
    }

    // Short conversations: single-pass (no need for blocks)
    if messages.len() <= BLOCK_SIZE {
        return llm_call(&format_transcript(messages), false);
    }

    // Split into blocks and summarize each in parallel — the LLM shell-out
    // is I/O-bound (network + subprocess), so rayon's work-stealing pool
    // saturates the available concurrency without us managing threads.
    // For a 600-turn conversation (20 blocks), this cuts end-to-end
    // summarisation from ~40 s to ~3-5 s on typical LLM rate limits.
    use rayon::prelude::*;
    let blocks: Vec<&[Message]> = messages.chunks(BLOCK_SIZE).collect();
    let total = blocks.len();
    let block_summaries: Vec<String> = blocks
        .par_iter()
        .enumerate()
        .map(|(i, block)| {
            let transcript = format_transcript(block);
            let body = llm_call(&transcript, false).unwrap_or_else(|| heuristic_summarize(block));
            format!("[Block {}/{}] {}", i + 1, total, body)
        })
        .collect();

    let combined = block_summaries.join("\n\n");

    // If combined summaries are short enough, use as-is
    if combined.len() <= MAX_SUMMARY_CHARS {
        return Some(combined);
    }

    // Second pass: summarize the summaries
    llm_call(&combined, true).or(Some(
        // Fallback: truncate to max chars if meta-summary fails
        combined.chars().take(MAX_SUMMARY_CHARS).collect(),
    ))
}

/// Format messages into a transcript string for the LLM prompt.
fn format_transcript(messages: &[Message]) -> String {
    let mut transcript = String::with_capacity(messages.len() * 100);
    for msg in messages {
        let role_tag = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };
        transcript.push_str(&format!("{}: {}\n", role_tag, &msg.content));
    }
    transcript
}

/// Execute a single LLM call. `is_meta` controls whether the prompt
/// asks to summarize raw conversation or to condense existing summaries.
fn llm_call(content: &str, is_meta: bool) -> Option<String> {
    let prompt = if is_meta {
        format!(
            "You are condensing multiple conversation summaries into one coherent summary.\n\
             Merge the facts below into a single paragraph. Preserve all names, dates,\n\
             locations, decisions, and preferences. Remove redundancy. Max 200 words.\n\n\
             SUMMARIES:\n{}\n\nCONDENSED SUMMARY:",
            content
        )
    } else {
        format!(
            "Summarize this conversation segment in a compact paragraph. Focus on:\n\
             - Names, dates, locations mentioned\n\
             - Decisions made and preferences stated\n\
             - Key questions asked and answers given\n\
             - Projects or topics discussed\n\
             Skip small talk. Keep each fact self-contained. Max 100 words.\n\n\
             CONVERSATION:\n{}\n\nSUMMARY:",
            content
        )
    };

    crate::llm_client::call_llm(&prompt)
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
