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
    compress_history_query(messages, config, None)
}

/// Query-aware variant of [`compress_history`] (Tier 3). When `query` is
/// provided, the summary of the dropped older messages is biased toward
/// lines relevant to the question instead of being built blind at compress
/// time. `None` reproduces the original behaviour exactly.
pub fn compress_history_query(
    messages: &[Message],
    config: &MemoryConfig,
    query: Option<&str>,
) -> CompressedMemory {
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
            // Fallback to the deterministic summariser if the LLM call fails
            if messages.len() < config.compression_threshold {
                deterministic_summarize(messages, query)
            } else {
                deterministic_summarize(older, query)
            }
        })
    } else if messages.len() < config.compression_threshold {
        deterministic_summarize(messages, query)
    } else {
        deterministic_summarize(older, query)
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

/// Deterministic summariser selector. Defaults to the fact-preserving
/// extractive summariser; `SOPHON_LEGACY_SUMMARY=1` restores the old
/// first-sentence/topic-list heuristic (kept for A/B and back-compat).
fn deterministic_summarize(messages: &[Message], query: Option<&str>) -> String {
    let legacy = std::env::var("SOPHON_LEGACY_SUMMARY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if legacy {
        heuristic_summarize(messages)
    } else {
        extractive_summarize_query(messages, query)
    }
}

/// Information-density score for a candidate line. Rewards the tokens that
/// carry recoverable facts — numbers, identifiers, paths, versions, code,
/// units — and zeroes out boilerplate. Deterministic and allocation-light.
fn signal_score(line: &str) -> u32 {
    let t = line.trim();
    if t.chars().count() < 8 {
        return 0;
    }
    let lower = t.to_lowercase();
    for bp in [
        "walk me through",
        "tell me about",
        "files:",
        "what did",
        "explain ",
        "summarize",
    ] {
        if lower.starts_with(bp) {
            return 0;
        }
    }
    let mut score = 0u32;
    for tok in t.split(|c: char| c.is_whitespace()) {
        let tok = tok.trim_matches(|c: char| {
            !c.is_alphanumeric() && !matches!(c, '_' | '/' | '.' | '%' | '$')
        });
        if tok.is_empty() {
            continue;
        }
        let chars: Vec<char> = tok.chars().collect();
        let has_digit = chars.iter().any(|c| c.is_ascii_digit());
        let has_underscore = tok.contains('_');
        let has_slash = tok.contains('/');
        let internal_caps = chars
            .iter()
            .enumerate()
            .any(|(i, &c)| i > 0 && c.is_ascii_uppercase() && chars[i - 1].is_ascii_lowercase());
        let dot_ext = [
            ".rs", ".md", ".py", ".toml", ".json", ".lock", ".yml", ".txt",
        ]
        .iter()
        .any(|e| tok.contains(e));
        if has_digit {
            score += 3;
        }
        if has_underscore || has_slash || internal_caps || dot_ext {
            score += 2;
        }
        if tok.ends_with('%') || tok.starts_with('$') {
            score += 2;
        }
    }
    if t.contains('`') {
        score += 4;
    }
    for u in [
        "mb", "kb", "gb", " ms", "loc", "token", "test", "commit", "version", "ratio", "pts",
        "bench", "error", "warning",
    ] {
        if lower.contains(u) {
            score += 1;
        }
    }
    score
}

/// Fact-preserving extractive summary. Replaces the first-sentence-only
/// heuristic: every line of the older messages is scored by information
/// density (`signal_score`) and the highest-signal, de-duplicated lines are
/// kept within a character budget, ordered by score so that the downstream
/// budget enforcer (which trims the tail) drops the least informative lines
/// first. Specific facts buried past a message's first sentence now survive.
pub fn extractive_summarize(messages: &[Message]) -> String {
    extractive_summarize_query(messages, None)
}

/// Query-aware extractive summary (Tier 3 fix). Identical to
/// [`extractive_summarize`] when `query` is `None`, but when a query is
/// provided, lines that lexically overlap the query are ranked **first**
/// (then by information density, then original order). The old default
/// summarised the dropped tail blind to the question, so a buried fact the
/// caller is actually asking about could lose the budget race to a denser
/// but irrelevant line. Ranking relevance-first lifts exactly the lines
/// that answer the query into the summary.
pub fn extractive_summarize_query(messages: &[Message], query: Option<&str>) -> String {
    if messages.is_empty() {
        return "No previous messages to summarize.".to_string();
    }
    const MAX_UNIT_CHARS: usize = 240;
    // Budget is relative to the input so the summary is always a real
    // compression (~4x) of the older slice, never a near-copy — clamped so
    // tiny histories still get a floor and huge ones don't blow up. The
    // downstream token enforcer trims further against the hard max_tokens.
    let input_chars: usize = messages.iter().map(|m| m.content.len()).sum();
    let budget_chars: usize = (input_chars / 4).clamp(200, 2800);

    let q_terms = query.map(query_terms).unwrap_or_default();

    // Candidate units: (order, role, text, signal_score, query_relevance).
    let mut units: Vec<(usize, Role, String, u32, u32)> = Vec::new();
    let mut order = 0usize;
    for m in messages {
        for raw_line in m.content.split('\n') {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let chunks: Vec<&str> = if line.chars().count() > 220 {
                line.split_terminator(['.', '!', '?']).collect()
            } else {
                vec![line]
            };
            for ch in chunks {
                let ch = ch.trim();
                if ch.chars().count() < 8 {
                    continue;
                }
                let text: String = ch.chars().take(MAX_UNIT_CHARS).collect();
                let sc = signal_score(ch);
                let rel = relevance_score(ch, &q_terms);
                units.push((order, m.role, text, sc, rel));
                order += 1;
            }
        }
    }
    if units.is_empty() {
        return heuristic_summarize(messages);
    }

    // Rank: query relevance desc, then signal desc, then original order asc
    // (stable, deterministic). With no query every relevance is 0, so this
    // reduces exactly to the previous signal-only ranking.
    let mut idx: Vec<usize> = (0..units.len()).collect();
    idx.sort_by(|&a, &b| {
        units[b]
            .4
            .cmp(&units[a].4)
            .then(units[b].3.cmp(&units[a].3))
            .then(units[a].0.cmp(&units[b].0))
    });

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut chosen: Vec<usize> = Vec::new();
    let mut chars = 0usize;
    for &i in &idx {
        // Keep a unit if it carries information density OR matches the query;
        // drop only pure boilerplate that is also irrelevant to the question.
        if units[i].3 == 0 && units[i].4 == 0 {
            continue;
        }
        let key = units[i].2.to_lowercase();
        if !seen.insert(key) {
            continue;
        }
        if chars + units[i].2.len() > budget_chars && !chosen.is_empty() {
            break;
        }
        chosen.push(i);
        chars += units[i].2.len() + 3;
    }
    if chosen.is_empty() {
        return heuristic_summarize(messages);
    }

    let mut out = String::with_capacity(chars);
    for &i in &chosen {
        let tag = match units[i].1 {
            Role::User => "Q",
            Role::Assistant => "A",
            Role::System => "S",
        };
        out.push_str(tag);
        out.push_str(": ");
        out.push_str(&units[i].2);
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Stopwords dropped from query terms — generic words carry no targeting
/// signal. Small on purpose so domain terms survive.
static QUERY_STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "was",
    "were",
    "are",
    "you",
    "your",
    "our",
    "with",
    "that",
    "this",
    "from",
    "about",
    "into",
    "tell",
    "explain",
    "summarize",
    "summarise",
    "give",
    "show",
    "list",
    "describe",
];

/// Tokenize a query into distinct lowercase terms (alphanumeric runs of
/// length ≥ 3, minus stopwords) used to score line relevance.
fn query_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in query.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let t = raw.trim_matches('_').to_lowercase();
        if t.chars().count() < 3 || QUERY_STOPWORDS.contains(&t.as_str()) {
            continue;
        }
        if seen.insert(t.clone()) {
            terms.push(t);
        }
    }
    terms
}

/// How many distinct query terms appear (case-insensitive substring) in a
/// candidate line. Substring rather than token-equality so `cache.rs`
/// matches a query mentioning `cache`, and `oldest_entry_age_seconds`
/// matches `age`.
fn relevance_score(line: &str, q_terms: &[String]) -> u32 {
    if q_terms.is_empty() {
        return 0;
    }
    let lower = line.to_lowercase();
    q_terms
        .iter()
        .filter(|t| lower.contains(t.as_str()))
        .count() as u32
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
            // Back the cut point up to a UTF-8 char boundary — `truncate`
            // panics mid-character, and summaries legitimately contain
            // multibyte glyphs (e.g. the `→` in "511 → 301"). The
            // query-aware path surfaces such lines more often, which is how
            // this latent bug first bit.
            let mut cut = summary.len().saturating_sub(80);
            while cut > 0 && !summary.is_char_boundary(cut) {
                cut -= 1;
            }
            summary.truncate(cut);
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
    // Rolling summary is built at ingest time, before any query exists —
    // necessarily query-blind. Query-aware biasing happens at compress time.
    let summary = if use_llm {
        llm_summarize(to_summarize).unwrap_or_else(|| deterministic_summarize(to_summarize, None))
    } else {
        deterministic_summarize(to_summarize, None)
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
    fn extractive_keeps_facts_buried_past_first_sentence() {
        // The fact (binary size) lives in the 3rd sentence — the old
        // first-sentence heuristic dropped it; the extractive path must keep it.
        let msgs = vec![asst(
            "docs(readme): full rewrite. This commit reworks the intro prose. \
             The stale binary size claim was 7.2 MB; the real binary is 5.2 MB. \
             It also corrects the test count from 303 to 405 in README.md.",
        )];
        let s = extractive_summarize(&msgs);
        assert!(s.contains("5.2"), "must keep buried numeric fact, got: {s}");
        assert!(
            s.contains("405") || s.contains("README.md"),
            "must keep buried identifier/count, got: {s}"
        );
        // The old heuristic only kept the first sentence — confirm the contrast.
        let legacy = heuristic_summarize(&msgs);
        assert!(
            !legacy.contains("5.2"),
            "legacy heuristic should NOT contain the buried fact (it keeps only the first sentence)"
        );
    }

    #[test]
    fn extractive_is_deterministic() {
        let msgs = vec![
            asst("bench: real_session_holistic.py adds 4 dimensions weighted 35/30/20/15."),
            asst("feat: --anonymise flag scrubs paths in real_session_* benches."),
        ];
        assert_eq!(extractive_summarize(&msgs), extractive_summarize(&msgs));
    }

    // Tier 3: with a tight budget, a query must pull its relevant (but not
    // densest) line into the summary ahead of unrelated dense lines.
    #[test]
    fn query_aware_surfaces_relevant_line() {
        // Many dense, numeric lines crowd a small budget; the database line
        // carries less raw "signal" and would lose the budget race blind.
        // A query about the database must flip it into the summary.
        let mut body = String::new();
        for i in 0..20 {
            body.push_str(&format!(
                "Metric row {i}: latency {i}.5 ms p99 across {i}096 shards in region us-east-{i}.\n"
            ));
        }
        body.push_str("The database migration moved us to postgres 16 for durability.\n");
        let msgs = vec![asst(&body)];

        let blind = extractive_summarize_query(&msgs, None);
        let with_q = extractive_summarize_query(&msgs, Some("which database did we migrate to"));

        assert!(
            !blind.to_lowercase().contains("postgres"),
            "blind summary should be crowded out by denser metric rows, got: {blind}"
        );
        assert!(
            with_q.to_lowercase().contains("postgres"),
            "query about database must surface the postgres line, got: {with_q}"
        );
    }

    // The query-aware path must keep a relevant line even when it carries
    // little intrinsic "signal" (no numbers/identifiers) — that line would
    // be dropped by the blind density filter.
    #[test]
    fn query_aware_keeps_low_signal_relevant_line() {
        let msgs = vec![asst(
            "The build pipeline runs fast. \
             The deployment owner is Priya on the platform team. \
             Numbers everywhere: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ms.",
        )];
        let with_q = extractive_summarize_query(&msgs, Some("who is the deployment owner"));
        assert!(
            with_q.to_lowercase().contains("priya"),
            "low-signal but query-relevant line must survive, got: {with_q}"
        );
    }

    // Regression: enforce_budget used to `String::truncate` at a byte
    // offset, panicking when the summary contained multibyte glyphs (the
    // `→` arrows common in this repo's commit/bench lines). A tight budget
    // forces the truncate branch; it must not panic.
    #[test]
    fn enforce_budget_handles_multibyte_summary() {
        let mut msgs = Vec::new();
        for i in 0..30 {
            msgs.push(asst(&format!(
                "bench row {i}: compressed 589 → 473 tokens, ratio 19.7% on file_{i}.rs"
            )));
        }
        let cfg = MemoryConfig {
            max_tokens: 60,
            ..MemoryConfig::default()
        };
        // Must not panic regardless of query-aware ranking surfacing arrows.
        let q = compress_history_query(&msgs, &cfg, Some("what ratio did the bench reach"));
        assert!(q.token_count <= 60 || !q.summary.is_empty());
        let blind = compress_history(&msgs, &cfg);
        let _ = blind;
    }

    // No query → byte-for-byte identical to the original blind summary, so
    // the ingest/rolling path and existing behaviour are untouched.
    #[test]
    fn query_none_is_unchanged_behaviour() {
        let msgs = vec![
            asst("commit a1b2c3 bumped version 0.5.4 and fixed cache.rs:95."),
            asst("benchmark improved recall from 28.1% to 35.3% at equal budget."),
        ];
        assert_eq!(
            extractive_summarize_query(&msgs, None),
            extractive_summarize(&msgs)
        );
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
