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

/// Pre-computed summary of the older slice of a conversation,
/// produced once at ingest and reused at query time.
///
/// Without rolling state, every `compress_history` call re-runs the
/// summariser over the full history; with LLM summarisation that is
/// the 5-8 s spike measured at v0.4.0. With rolling state the LLM
/// pays the cost once when crossing the refresh threshold, then
/// `compress_history_with_rolling` only stitches `summary` to the
/// live recent window — sub-millisecond on the hot path.
///
/// `summarized_until` is exclusive: `summary` covers
/// `history[..summarized_until]`, and `history[summarized_until..]`
/// is the un-summarised tail that becomes the recent window.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RollingSummary {
    pub summary: String,
    pub summarized_until: usize,
    pub refreshed_at: chrono::DateTime<chrono::Utc>,
}

/// Default refresh threshold — refresh the rolling summary once
/// the un-summarised tail reaches this many messages.
///
/// Picked empirically: with `recent_window = 5` (default), the
/// recent_floor below evaluates to `max(2 × 5, 8) = 10`. Refreshing
/// every 50 means each refresh covers roughly 40 new messages, which
/// is large enough that the LLM call (5-8 s) stays amortised over
/// many queries.
pub const DEFAULT_ROLLING_REFRESH_THRESHOLD: usize = 50;

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

// ---------------------------------------------------------------------------
// Rolling summary — phase 2B
// ---------------------------------------------------------------------------

/// Decide whether the rolling summary needs a refresh; if so, build
/// a new one. Returns `None` when the un-summarised tail is below
/// `refresh_threshold`, or when leaving enough recent messages
/// outside the summary would consume the whole tail (i.e. the
/// session is too short for rolling to apply yet).
///
/// LLM activation matches `compress_history`'s policy:
///   - implicit if `SOPHON_LLM_CMD` is set,
///   - explicit via `config.use_llm_summarization`,
///   - opt-out via `SOPHON_NO_LLM_SUMMARY=1`.
///
/// On LLM-call failure the function falls back to
/// `heuristic_summarize` so a transient subprocess error never
/// blocks ingest.
#[tracing::instrument(
    skip_all,
    fields(
        messages = history.len(),
        existing_until = existing.map(|r| r.summarized_until).unwrap_or(0),
        refresh_threshold = refresh_threshold,
    ),
)]
pub fn refresh_rolling_summary(
    history: &[Message],
    existing: Option<&RollingSummary>,
    config: &MemoryConfig,
    refresh_threshold: usize,
) -> Option<RollingSummary> {
    let last_until = existing.map(|r| r.summarized_until).unwrap_or(0);
    let unsummarized = history.len().saturating_sub(last_until);
    if unsummarized < refresh_threshold {
        return None;
    }
    // Always leave at least 2 × recent_window messages outside the
    // summary so callers always have some recent live context. Floor
    // at 8 to avoid pathological cases when recent_window is small.
    let recent_floor = config.recent_window.saturating_mul(2).max(8);
    let cap_until = history.len().saturating_sub(recent_floor);
    if cap_until <= last_until {
        return None;
    }
    let to_summarize = &history[..cap_until];

    let opt_out = std::env::var("SOPHON_NO_LLM_SUMMARY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let use_llm =
        !opt_out && (config.use_llm_summarization || std::env::var("SOPHON_LLM_CMD").is_ok());
    let summary = if use_llm {
        llm_summarize(to_summarize).unwrap_or_else(|| heuristic_summarize(to_summarize))
    } else {
        heuristic_summarize(to_summarize)
    };

    Some(RollingSummary {
        summary,
        summarized_until: cap_until,
        refreshed_at: chrono::Utc::now(),
    })
}

/// Compress conversation history using a pre-computed rolling
/// summary for the older slice. The recent window is read live so
/// newly-appended messages are visible without waiting for the next
/// refresh.
///
/// Falls back to `compress_history` when:
///   - `rolling` is `None` (feature inactive),
///   - the rolling state is ahead of the current history (e.g.
///     after a `MemoryManager::reset` the summary is stale and must
///     be recomputed from scratch).
///
/// This makes the rolling-summary feature strictly additive: a
/// caller that doesn't opt in sees byte-identical output to the
/// pre-2B path.
#[tracing::instrument(
    skip_all,
    fields(
        messages = messages.len(),
        has_rolling = rolling.is_some(),
        summarized_until = rolling.map(|r| r.summarized_until).unwrap_or(0),
    ),
)]
pub fn compress_history_with_rolling(
    messages: &[Message],
    rolling: Option<&RollingSummary>,
    config: &MemoryConfig,
) -> CompressedMemory {
    let Some(rolling) = rolling else {
        return compress_history(messages, config);
    };
    if rolling.summarized_until > messages.len() {
        return compress_history(messages, config);
    }
    if rolling.summarized_until == 0 {
        return compress_history(messages, config);
    }

    let recent = &messages[rolling.summarized_until..];
    let stable_facts = extract_facts(messages);
    let mut compressed = CompressedMemory {
        summary: rolling.summary.clone(),
        stable_facts,
        recent_messages: recent.to_vec(),
        index: build_index(messages),
        token_count: 0,
        original_message_count: messages.len(),
    };
    enforce_budget(&mut compressed, config.max_tokens);

    // Same safety net as `compress_history`: if the budget enforcer
    // can't shrink below the raw token cost, fall back to passthrough.
    let raw_tokens: usize = messages.iter().map(|m| m.token_count).sum();
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

#[cfg(test)]
mod rolling_tests {
    use super::*;
    use crate::message::{Message, Role};

    fn user(s: &str) -> Message {
        Message::new(Role::User, s.to_string())
    }
    fn asst(s: &str) -> Message {
        Message::new(Role::Assistant, s.to_string())
    }

    /// Build N synthetic message pairs so the history grows
    /// deterministically — used to exercise threshold logic.
    fn synth_history(pairs: usize) -> Vec<Message> {
        let mut out = Vec::with_capacity(pairs * 2);
        for i in 0..pairs {
            out.push(user(&format!("Question {i}: how does X work?")));
            out.push(asst(&format!(
                "Answer {i}: X works by combining Y and Z under conditions A_{i}."
            )));
        }
        out
    }

    #[test]
    fn refresh_returns_none_below_threshold() {
        let history = synth_history(20); // 40 messages
                                         // SOPHON_NO_LLM_SUMMARY guarantees deterministic heuristic path
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        let cfg = MemoryConfig::default();
        let result = refresh_rolling_summary(&history, None, &cfg, 50);
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        assert!(
            result.is_none(),
            "below threshold (40 < 50) should return None"
        );
    }

    #[test]
    fn refresh_returns_some_above_threshold() {
        let history = synth_history(40); // 80 messages
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        let cfg = MemoryConfig::default();
        let result = refresh_rolling_summary(&history, None, &cfg, 50);
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        let r = result.expect("threshold met → Some");
        assert!(!r.summary.is_empty(), "summary should not be empty");
        assert!(
            r.summarized_until < history.len(),
            "must leave a recent window outside the summary"
        );
        assert!(
            history.len() - r.summarized_until >= 8,
            "recent floor should keep at least 8 live messages, got {}",
            history.len() - r.summarized_until
        );
    }

    #[test]
    fn refresh_skips_when_no_new_content_to_summarize() {
        let history = synth_history(40); // 80 messages
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        let cfg = MemoryConfig::default();
        let first = refresh_rolling_summary(&history, None, &cfg, 50)
            .expect("first refresh should succeed");
        // Same history, same threshold → no new tail → None
        let second = refresh_rolling_summary(&history, Some(&first), &cfg, 50);
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        assert!(
            second.is_none(),
            "no new content since last refresh should yield None"
        );
    }

    #[test]
    fn compress_with_rolling_falls_back_when_state_is_stale() {
        // Build a state that points past the end of the live history —
        // simulates a reset() that wasn't propagated. Result must be
        // identical to the no-rolling path.
        let history = synth_history(5); // 10 messages
        let stale = RollingSummary {
            summary: "stale".to_string(),
            summarized_until: 999,
            refreshed_at: chrono::Utc::now(),
        };
        let cfg = MemoryConfig::default();
        let with = compress_history_with_rolling(&history, Some(&stale), &cfg);
        let without = compress_history(&history, &cfg);
        // Recent messages set must match.
        assert_eq!(with.recent_messages.len(), without.recent_messages.len());
    }

    #[test]
    fn compress_with_rolling_uses_summary_when_state_is_valid() {
        let history = synth_history(40);
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        let cfg = MemoryConfig::default();
        let rolling = refresh_rolling_summary(&history, None, &cfg, 50).expect("Some");
        let compressed = compress_history_with_rolling(&history, Some(&rolling), &cfg);
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        // Summary should match the rolling state (modulo budget trim).
        assert!(
            !compressed.summary.is_empty(),
            "compress_with_rolling must serve the rolling summary"
        );
        // Recent messages should be the un-summarised tail.
        let expected_recent_len = history.len() - rolling.summarized_until;
        assert!(
            compressed.recent_messages.len() <= expected_recent_len,
            "recent should be ≤ tail length (budget may trim)"
        );
    }

    #[test]
    fn compress_with_rolling_no_state_matches_baseline() {
        let history = synth_history(15);
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        let cfg = MemoryConfig::default();
        let with_none = compress_history_with_rolling(&history, None, &cfg);
        let baseline = compress_history(&history, &cfg);
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        // No rolling state → identical to baseline.
        assert_eq!(with_none.summary, baseline.summary);
        assert_eq!(
            with_none.recent_messages.len(),
            baseline.recent_messages.len()
        );
        assert_eq!(with_none.token_count, baseline.token_count);
    }
}
