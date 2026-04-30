//! Message chunker — splits raw conversation messages into retrievable units.
//!
//! Strategy:
//! 1. Each fenced code block is extracted as its own chunk (preserves
//!    indentation and language tag).
//! 2. The remaining prose is split at sentence boundaries (`. ! ?` followed
//!    by whitespace, or paragraph breaks) and packed into chunks of
//!    `target_size` tokens, with a small overlap to preserve continuity.
//! 3. Chunk IDs are deterministic: `sha256(content)[..16]`. Re-indexing the
//!    same conversation yields the same IDs, so the chunk store stays
//!    idempotent.

use std::collections::HashSet;

use chrono::{DateTime, Datelike, Duration, Utc};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sophon_core::tokens::count_tokens;

/// A retrievable unit of conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub id: String,
    pub source_message_indices: Vec<usize>,
    pub content: String,
    pub token_count: usize,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<String>,
    pub chunk_type: ChunkType,
    /// Embedding stored alongside the chunk so the index can be rebuilt
    /// without re-running the embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    UserStatement,
    AssistantResponse,
    SystemMessage,
    CodeBlock,
    /// Agent tool invocation — Anthropic `{"type":"tool_use",
    /// "name":"…","input":{…}}` shape, or the equivalent in other
    /// MCP-style transcripts. Stored with a normalised
    /// `tool:NAME({sorted_args_json})` content form so two
    /// identical calls share the same chunk id and dedup
    /// naturally.
    ToolUse,
    /// Result of a tool call. Less stable for dedup (tool_use_id
    /// varies per invocation, results often include timestamps
    /// or process state) but tagged so the retriever can prefer
    /// or down-weight them per query.
    ToolResult,
}

/// Configuration for the chunker.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in tokens. Sentences are packed up to this size.
    pub target_size: usize,
    /// Hard upper bound on a single chunk; used when a single sentence
    /// exceeds `target_size`.
    pub max_size: usize,
    /// Overlap (in tokens) between consecutive prose chunks.
    pub overlap: usize,
    /// Extract fenced code blocks as their own chunks.
    pub extract_code: bool,
    /// When true, detect topic boundaries between consecutive sentences
    /// using Jaccard similarity on word sets. A boundary is inserted when
    /// the Jaccard coefficient drops below 0.3, forcing a chunk break even
    /// if the buffer is under `target_size`.
    pub use_topic_splitting: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 128,
            max_size: 256,
            overlap: 32,
            extract_code: true,
            use_topic_splitting: false,
        }
    }
}

/// Trait-free message shape so we don't need to depend on `memory-manager`.
#[derive(Debug, Clone)]
pub struct ChunkInputMessage<'a> {
    pub index: usize,
    pub role: ChunkInputRole,
    pub content: &'a str,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub enum ChunkInputRole {
    User,
    Assistant,
    System,
}

impl ChunkInputRole {
    fn to_chunk_type(self) -> ChunkType {
        match self {
            Self::User => ChunkType::UserStatement,
            Self::Assistant => ChunkType::AssistantResponse,
            Self::System => ChunkType::SystemMessage,
        }
    }
}

static CODE_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```").expect("valid code regex"));

/// Detect an Anthropic-shaped `tool_use` block in the message
/// content and return a normalised canonical form. The canonical
/// form is `tool:NAME({sorted_args_json})` so two calls with the
/// same name and args (even if argument keys are emitted in
/// different orders) produce byte-identical chunks and dedup via
/// `chunk_id`.
///
/// Recognised shapes:
///
/// * Plain JSON object: `{"type":"tool_use","name":"X","input":{…}}`
/// * Pretty-printed equivalent (any indentation).
/// * String form `Tool call: X({"key": "value"})` — common in
///   transcripts saved by clients that flatten the JSON for
///   readability.
///
/// Returns `None` when no recognisable tool-use shape is found.
/// Conservative: when in doubt, treat as prose so we don't merge
/// unrelated content via false-positive dedup.
pub(crate) fn detect_tool_use(content: &str) -> Option<String> {
    let trimmed = content.trim();

    // Shape A: JSON object — must start with `{`.
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if v.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                let name = v.get("name").and_then(|n| n.as_str())?;
                let input = v.get("input").cloned().unwrap_or(serde_json::Value::Null);
                return Some(format!("tool:{}({})", name, normalise_json(&input)));
            }
        }
    }

    // Shape B: flattened `Tool call: NAME(json)` line.
    if let Some(rest) = trimmed.strip_prefix("Tool call: ") {
        if let Some(open) = rest.find('(') {
            let name = rest[..open].trim();
            let args_str = rest[open + 1..].trim_end_matches(')');
            if !name.is_empty() {
                let parsed = serde_json::from_str::<serde_json::Value>(args_str)
                    .unwrap_or_else(|_| serde_json::Value::String(args_str.to_string()));
                return Some(format!("tool:{}({})", name, normalise_json(&parsed)));
            }
        }
    }

    None
}

/// Detect an Anthropic-shaped `tool_result` block. Returns the raw
/// content with the wrapping metadata stripped — we keep the
/// content because it's the actually-useful payload, but tag it
/// `ChunkType::ToolResult` so the retriever can weight it
/// differently.
pub(crate) fn detect_tool_result(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    if v.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
        return None;
    }
    // Two valid shapes for content: a string or an array of content blocks.
    let content_field = v.get("content")?;
    let extracted = match content_field {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    Some(extracted)
}

/// Recursively serialise a JSON value with object keys sorted, so
/// `{a: 1, b: 2}` and `{b: 2, a: 1}` produce identical strings.
/// `serde_json` doesn't sort by default; the BTreeMap detour does.
fn normalise_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            // BTreeMap iterates in key order.
            let sorted: std::collections::BTreeMap<&String, &serde_json::Value> =
                map.iter().collect();
            let mut s = String::from("{");
            for (i, (k, val)) in sorted.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push('"');
                s.push_str(k);
                s.push_str("\":");
                s.push_str(&normalise_json(val));
            }
            s.push('}');
            s
        }
        serde_json::Value::Array(items) => {
            let mut s = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&normalise_json(item));
            }
            s.push(']');
            s
        }
        _ => v.to_string(),
    }
}

/// Public entry point. Takes any iterator of `ChunkInputMessage` and emits
/// the chunks in document order.
pub fn chunk_messages(messages: &[ChunkInputMessage<'_>], config: &ChunkConfig) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for msg in messages {
        // Tool-call detection runs first so identical invocations
        // dedup via the canonical form regardless of how the
        // surrounding message was punctuated. Detected tool calls
        // skip the prose / code-block path entirely — they're
        // structurally distinct content.
        if let Some(canonical) = detect_tool_use(msg.content) {
            let chunk = build_chunk(
                canonical,
                ChunkType::ToolUse,
                msg.index,
                msg.timestamp,
                msg.session_id,
            );
            if seen.insert(chunk.id.clone()) {
                chunks.push(chunk);
            }
            continue;
        }
        if let Some(extracted) = detect_tool_result(msg.content) {
            let chunk = build_chunk(
                extracted,
                ChunkType::ToolResult,
                msg.index,
                msg.timestamp,
                msg.session_id,
            );
            if seen.insert(chunk.id.clone()) {
                chunks.push(chunk);
            }
            continue;
        }

        let mut prose = resolve_temporal_refs(msg.content, msg.timestamp);

        if config.extract_code {
            for code in CODE_BLOCK_RE.captures_iter(msg.content) {
                let body = code.get(1).map(|m| m.as_str()).unwrap_or("");
                if !body.trim().is_empty() {
                    let chunk = build_chunk(
                        body.to_string(),
                        ChunkType::CodeBlock,
                        msg.index,
                        msg.timestamp,
                        msg.session_id,
                    );
                    if seen.insert(chunk.id.clone()) {
                        chunks.push(chunk);
                    }
                }
            }
            // Strip the code blocks from the resolved prose so they aren't
            // re-chunked. Use `&prose` (not `msg.content`) to preserve the
            // temporal resolution from line 115.
            prose = CODE_BLOCK_RE.replace_all(&prose, "").to_string();
        }

        if prose.trim().is_empty() {
            continue;
        }

        let sentences = split_sentences(&prose);
        let topic_breaks = if config.use_topic_splitting {
            detect_topic_breaks(&sentences)
        } else {
            HashSet::new()
        };
        let prose_chunks = pack_sentences(
            &sentences,
            config,
            msg.index,
            msg.timestamp,
            msg.session_id,
            msg.role.to_chunk_type(),
            &topic_breaks,
        );

        for chunk in prose_chunks {
            if seen.insert(chunk.id.clone()) {
                chunks.push(chunk);
            }
        }
    }

    chunks
}

/// Hand-rolled sentence splitter. The Rust `regex` crate doesn't support
/// look-ahead, so we scan char-by-char: split after `.`, `!`, `?`, or
/// `\n\n` when followed by whitespace. Conservative — we'd rather keep
/// two sentences together than slice mid-thought.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        buf.push(c);
        let is_terminator = c == '.' || c == '!' || c == '?';
        let is_paragraph_break = c == '\n' && chars.peek() == Some(&'\n');

        if is_terminator {
            // Look at the next char: if it's whitespace, this terminator
            // ends a sentence.
            if matches!(chars.peek(), Some(c) if c.is_whitespace()) {
                let s = buf.trim().to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                buf.clear();
            }
        } else if is_paragraph_break {
            // Consume the second newline.
            chars.next();
            let s = buf.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            buf.clear();
        }
    }

    let tail = buf.trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }

    if out.is_empty() && !text.trim().is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// Compute the word set for a sentence: lowercased, whitespace-split,
/// with trailing/leading punctuation stripped so that "restaurants." and
/// "restaurant" can still overlap.
fn word_set(sentence: &str) -> HashSet<String> {
    sentence
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

/// Pre-pass: find indices where a topic boundary exists. An index `i` in the
/// returned set means there is a topic break *before* sentence `i` — i.e.
/// sentences `i-1` and `i` belong to different topics.
fn detect_topic_breaks(sentences: &[String]) -> HashSet<usize> {
    let mut breaks = HashSet::new();
    if sentences.len() < 2 {
        return breaks;
    }
    let mut prev_words = word_set(&sentences[0]);
    for i in 1..sentences.len() {
        let cur_words = word_set(&sentences[i]);
        if jaccard(&prev_words, &cur_words) < 0.3 {
            breaks.insert(i);
        }
        prev_words = cur_words;
    }
    breaks
}

fn pack_sentences(
    sentences: &[String],
    config: &ChunkConfig,
    msg_index: usize,
    timestamp: DateTime<Utc>,
    session_id: Option<&str>,
    chunk_type: ChunkType,
    topic_breaks: &HashSet<usize>,
) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut buf_tokens = 0usize;

    let flush = |buf: &mut Vec<String>, buf_tokens: &mut usize, out: &mut Vec<Chunk>| {
        if buf.is_empty() {
            return;
        }
        let content = buf.join(" ");
        out.push(build_chunk(
            content, chunk_type, msg_index, timestamp, session_id,
        ));
        buf.clear();
        *buf_tokens = 0;
    };

    for (i, sentence) in sentences.iter().enumerate() {
        let s_tokens = count_tokens(sentence);

        // A single sentence longer than max_size: force-split on whitespace.
        if s_tokens > config.max_size {
            flush(&mut buf, &mut buf_tokens, &mut out);
            for piece in hard_split(sentence, config.max_size) {
                let content = piece;
                out.push(build_chunk(
                    content, chunk_type, msg_index, timestamp, session_id,
                ));
            }
            continue;
        }

        // Topic break: flush the current buffer before starting the new topic.
        if topic_breaks.contains(&i) && !buf.is_empty() {
            flush(&mut buf, &mut buf_tokens, &mut out);
        }

        if buf_tokens + s_tokens > config.target_size && !buf.is_empty() {
            flush(&mut buf, &mut buf_tokens, &mut out);
        }

        buf.push(sentence.clone());
        buf_tokens += s_tokens;
    }
    flush(&mut buf, &mut buf_tokens, &mut out);

    apply_overlap(
        out,
        config.overlap,
        chunk_type,
        msg_index,
        timestamp,
        session_id,
    )
}

fn hard_split(sentence: &str, max_tokens: usize) -> Vec<String> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut out = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut buf_tokens = 0usize;
    for w in words {
        let wt = count_tokens(w);
        if buf_tokens + wt > max_tokens && !buf.is_empty() {
            out.push(buf.join(" "));
            buf.clear();
            buf_tokens = 0;
        }
        buf.push(w);
        buf_tokens += wt;
    }
    if !buf.is_empty() {
        out.push(buf.join(" "));
    }
    out
}

fn apply_overlap(
    chunks: Vec<Chunk>,
    overlap_tokens: usize,
    chunk_type: ChunkType,
    msg_index: usize,
    timestamp: DateTime<Utc>,
    session_id: Option<&str>,
) -> Vec<Chunk> {
    if overlap_tokens == 0 || chunks.len() < 2 {
        return chunks;
    }
    let mut out = Vec::with_capacity(chunks.len());
    let mut prev_tail: Option<String> = None;
    for chunk in chunks {
        let content = if let Some(tail) = prev_tail.take() {
            // Re-build the chunk with the tail prepended so retrieval can
            // bridge sentence boundaries.
            let merged = format!("{} {}", tail, chunk.content);
            build_chunk(merged, chunk_type, msg_index, timestamp, session_id)
        } else {
            chunk.clone()
        };
        // Capture the tail of this chunk for the next iteration.
        prev_tail = Some(tail_tokens(&chunk.content, overlap_tokens));
        out.push(content);
    }
    out
}

fn tail_tokens(content: &str, n: usize) -> String {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.is_empty() || n == 0 {
        return String::new();
    }
    // Walk backwards, adding one word at a time and measuring the token
    // count of the *joined* candidate string (BPE token boundaries shift
    // depending on surrounding context, so counting individual words is
    // insufficient).
    let mut best_start = words.len();
    for start in (0..words.len()).rev() {
        let candidate = words[start..].join(" ");
        let tc = count_tokens(&candidate);
        if tc > n && best_start < words.len() {
            break;
        }
        best_start = start;
        if tc >= n {
            break;
        }
    }
    words[best_start..].join(" ")
}

fn build_chunk(
    content: String,
    chunk_type: ChunkType,
    msg_index: usize,
    timestamp: DateTime<Utc>,
    session_id: Option<&str>,
) -> Chunk {
    let token_count = count_tokens(&content);
    let id = chunk_id(&content);
    Chunk {
        id,
        source_message_indices: vec![msg_index],
        content,
        token_count,
        timestamp,
        session_id: session_id.map(|s| s.to_string()),
        chunk_type,
        embedding: None,
    }
}

/// Stable 16-hex-char ID derived from a SHA-256 of the content. Re-indexing
/// the same chunk yields the same ID, so the store skips duplicates without
/// any application-level bookkeeping.
pub fn chunk_id(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let mut s = String::with_capacity(16);
    for b in &digest[..8] {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Resolve relative temporal references ("last week", "yesterday", etc.)
/// to absolute dates based on the message's timestamp. Appends the
/// resolved date in brackets after the original phrase so both forms
/// are searchable: "last week [week of 2023-10-06]".
///
/// This fixes the N=40 temporal_reasoning failures where "last month"
/// in a message dated 2023-10-13 should resolve to "September 2023"
/// but the chunk stored the raw "last month" which didn't match
/// date-oriented queries.
fn resolve_temporal_refs(text: &str, timestamp: DateTime<Utc>) -> String {
    static PATTERNS: Lazy<Vec<(Regex, Box<dyn Fn(DateTime<Utc>) -> String + Send + Sync>)>> =
        Lazy::new(|| {
            vec![
                (
                    Regex::new(r"(?i)\byesterday\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        let d = ts - Duration::days(1);
                        format!("yesterday [{}]", d.format("%Y-%m-%d"))
                    }) as Box<dyn Fn(DateTime<Utc>) -> String + Send + Sync>,
                ),
                (
                    Regex::new(r"(?i)\blast week\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        let d = ts - Duration::weeks(1);
                        format!("last week [week of {}]", d.format("%Y-%m-%d"))
                    }),
                ),
                (
                    Regex::new(r"(?i)\blast month\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        let d = ts - Duration::days(30);
                        let month_names = [
                            "",
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                        ];
                        format!(
                            "last month [{} {}]",
                            month_names[d.month() as usize],
                            d.year()
                        )
                    }),
                ),
                (
                    Regex::new(r"(?i)\blast year\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| format!("last year [{}]", ts.year() - 1)),
                ),
                (
                    Regex::new(r"(?i)\bthis week\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        format!("this week [week of {}]", ts.format("%Y-%m-%d"))
                    }),
                ),
                (
                    Regex::new(r"(?i)\bthis month\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        let month_names = [
                            "",
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                        ];
                        format!(
                            "this month [{} {}]",
                            month_names[ts.month() as usize],
                            ts.year()
                        )
                    }),
                ),
                (
                    Regex::new(r"(?i)\btoday\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| format!("today [{}]", ts.format("%Y-%m-%d"))),
                ),
                (
                    Regex::new(r"(?i)\btomorrow\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        let d = ts + Duration::days(1);
                        format!("tomorrow [{}]", d.format("%Y-%m-%d"))
                    }),
                ),
                (
                    Regex::new(r"(?i)\blast friday\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        // Walk back to the most recent Friday before ts
                        let weekday = ts.weekday().num_days_from_monday(); // Mon=0..Sun=6
                        let days_back = if weekday >= 4 {
                            weekday - 4
                        } else {
                            weekday + 3
                        };
                        let d = ts - Duration::days(days_back as i64);
                        format!("last Friday [{}]", d.format("%Y-%m-%d"))
                    }),
                ),
                (
                    Regex::new(r"(?i)\brecently\b").unwrap(),
                    Box::new(|ts: DateTime<Utc>| {
                        format!("recently [around {}]", ts.format("%Y-%m"))
                    }),
                ),
            ]
        });

    let mut result = text.to_string();
    for (re, resolver) in PATTERNS.iter() {
        if re.is_match(&result) {
            let resolved = resolver(timestamp);
            result = re.replace_all(&result, resolved.as_str()).to_string();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg<'a>(idx: usize, role: ChunkInputRole, content: &'a str) -> ChunkInputMessage<'a> {
        ChunkInputMessage {
            index: idx,
            role,
            content,
            timestamp: Utc::now(),
            session_id: None,
        }
    }

    #[test]
    fn empty_messages_produces_no_chunks() {
        let cfg = ChunkConfig::default();
        let chunks = chunk_messages(&[], &cfg);
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_sentence_one_chunk() {
        let cfg = ChunkConfig::default();
        let m = msg(0, ChunkInputRole::User, "Alice recommended a restaurant.");
        let chunks = chunk_messages(&[m], &cfg);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Alice"));
        assert_eq!(chunks[0].chunk_type, ChunkType::UserStatement);
    }

    #[test]
    fn long_message_packs_into_multiple_chunks() {
        let cfg = ChunkConfig {
            target_size: 20,
            max_size: 40,
            overlap: 0,
            extract_code: true,
            use_topic_splitting: false,
        };
        let big = (0..30)
            .map(|i| format!("This is sentence number {}.", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_messages(&[msg(0, ChunkInputRole::User, &big)], &cfg);
        assert!(
            chunks.len() > 1,
            "expected packing, got {} chunks",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.token_count <= cfg.max_size + 4,
                "chunk too big: {}",
                c.token_count
            );
        }
    }

    #[test]
    fn code_block_extracted_separately() {
        let cfg = ChunkConfig::default();
        let body =
            "Here's a snippet:\n```rust\nfn main() { println!(\"hi\"); }\n```\nThat's the example.";
        let chunks = chunk_messages(&[msg(0, ChunkInputRole::Assistant, body)], &cfg);
        assert!(chunks
            .iter()
            .any(|c| matches!(c.chunk_type, ChunkType::CodeBlock)));
        // Prose around the code is also a chunk.
        assert!(chunks
            .iter()
            .any(|c| matches!(c.chunk_type, ChunkType::AssistantResponse)));
    }

    #[test]
    fn tail_tokens_counts_tokens_not_words() {
        // "hello" is 1 token but "world" is also 1 token in cl100k_base.
        // A multi-byte / multi-token word should cause fewer words to be
        // taken than the plain word-count approach would yield.
        let content = "aaa bbb ccc ddd eee";
        // Ask for 2 tokens of overlap. With the old word-based approach this
        // would return the last 2 whitespace-delimited words regardless of
        // their actual token cost. The new implementation counts real tokens.
        let tail = tail_tokens(content, 2);
        let actual = count_tokens(&tail);
        // The tail must be <= 2 tokens (it may be exactly 2 if each word is
        // 1 token, but must never exceed the requested budget).
        assert!(
            actual <= 2,
            "tail_tokens produced {} tokens (content: {:?}), expected <= 2",
            actual,
            tail,
        );
        // It should contain at least 1 token.
        assert!(
            actual >= 1,
            "tail_tokens produced 0 tokens, expected at least 1",
        );
    }

    #[test]
    fn duplicate_content_yields_same_id() {
        let cfg = ChunkConfig::default();
        let chunks = chunk_messages(
            &[
                msg(0, ChunkInputRole::User, "Same content here please."),
                msg(1, ChunkInputRole::User, "Same content here please."),
            ],
            &cfg,
        );
        // Dedup by id means we only get one chunk back.
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn topic_coherent_text_stays_in_one_chunk() {
        let cfg = ChunkConfig {
            target_size: 256,
            max_size: 512,
            overlap: 0,
            extract_code: false,
            use_topic_splitting: true,
        };
        // All sentences share heavy keyword overlap so Jaccard stays >= 0.3.
        let text = "I love Italian food and Italian restaurants. \
                     Italian restaurants serve the best Italian food. \
                     The best Italian food comes from Italian restaurants near my house.";
        let chunks = chunk_messages(&[msg(0, ChunkInputRole::User, text)], &cfg);
        assert_eq!(
            chunks.len(),
            1,
            "topic-coherent sentences should stay in one chunk, got {}",
            chunks.len()
        );
        assert!(chunks[0].content.contains("Italian"));
    }

    #[test]
    fn distinct_topics_split_into_separate_chunks() {
        let cfg = ChunkConfig {
            // Large target so token limit alone wouldn't cause a split.
            target_size: 512,
            max_size: 1024,
            overlap: 0,
            extract_code: false,
            use_topic_splitting: true,
        };
        // Two clearly distinct topics with zero keyword overlap.
        let text = "I love Italian restaurants. Italian food is the best cuisine. \
                     The weather forecast predicts heavy rain tomorrow. \
                     Thunderstorms are expected throughout the weekend.";
        let chunks = chunk_messages(&[msg(0, ChunkInputRole::User, text)], &cfg);
        assert!(
            chunks.len() >= 2,
            "distinct topics should split into at least 2 chunks, got {}",
            chunks.len()
        );
        // First chunk should be about food, second about weather.
        assert!(
            chunks[0].content.contains("Italian"),
            "first chunk should contain the restaurant topic"
        );
        assert!(
            chunks.last().unwrap().content.contains("rain")
                || chunks.last().unwrap().content.contains("Thunderstorms"),
            "last chunk should contain the weather topic"
        );
    }

    // -----------------------------------------------------------------------
    // Tool-call detection + dedup (phase 3 / tier 1B)
    // -----------------------------------------------------------------------

    #[test]
    fn detect_tool_use_anthropic_shape() {
        let blob = r#"{"type":"tool_use","name":"read_file","input":{"path":"src/main.rs"}}"#;
        let canonical = detect_tool_use(blob).expect("should detect");
        assert!(canonical.starts_with("tool:read_file("));
        assert!(canonical.contains("\"path\":\"src/main.rs\""));
    }

    #[test]
    fn detect_tool_use_normalises_arg_order() {
        // Same call, different key emission order — must produce
        // byte-identical canonical strings.
        let a = r#"{"type":"tool_use","name":"grep","input":{"pattern":"foo","path":"src"}}"#;
        let b = r#"{"type":"tool_use","name":"grep","input":{"path":"src","pattern":"foo"}}"#;
        let ca = detect_tool_use(a).unwrap();
        let cb = detect_tool_use(b).unwrap();
        assert_eq!(
            ca, cb,
            "arg-order normalisation must produce identical canonical"
        );
    }

    #[test]
    fn detect_tool_use_flat_string_form() {
        let blob = r#"Tool call: read_file({"path": "Cargo.toml"})"#;
        let canonical = detect_tool_use(blob).expect("flat form should be recognised");
        assert!(canonical.starts_with("tool:read_file("));
        assert!(canonical.contains("Cargo.toml"));
    }

    #[test]
    fn detect_tool_use_returns_none_for_prose() {
        assert!(detect_tool_use("hello, can you read the file?").is_none());
        assert!(detect_tool_use("{\"id\": 42}").is_none());
        assert!(detect_tool_use("").is_none());
    }

    #[test]
    fn detect_tool_result_extracts_content_string() {
        let blob = r#"{"type":"tool_result","tool_use_id":"abc","content":"file contents here"}"#;
        let extracted = detect_tool_result(blob).unwrap();
        assert_eq!(extracted, "file contents here");
    }

    #[test]
    fn detect_tool_result_extracts_array_form() {
        let blob = r#"{"type":"tool_result","tool_use_id":"abc","content":[{"type":"text","text":"line 1"},{"type":"text","text":"line 2"}]}"#;
        let extracted = detect_tool_result(blob).unwrap();
        assert_eq!(extracted, "line 1\nline 2");
    }

    fn tool_msg(idx: usize, content: &'static str) -> ChunkInputMessage<'static> {
        ChunkInputMessage {
            index: idx,
            role: ChunkInputRole::Assistant,
            content,
            timestamp: Utc::now(),
            session_id: None,
        }
    }

    #[test]
    fn chunker_dedups_repeated_identical_tool_calls() {
        // Same tool, same args, called 5 times — should produce
        // exactly one chunk after dedup.
        let raw = r#"{"type":"tool_use","name":"read_file","input":{"path":"src/lib.rs"}}"#;
        let msgs: Vec<ChunkInputMessage<'_>> = (0..5).map(|i| tool_msg(i, raw)).collect();
        let cfg = ChunkConfig::default();
        let chunks = chunk_messages(&msgs, &cfg);
        assert_eq!(
            chunks.len(),
            1,
            "5 identical tool calls must dedup to 1 chunk, got {}",
            chunks.len()
        );
        assert!(matches!(chunks[0].chunk_type, ChunkType::ToolUse));
        assert!(chunks[0].content.starts_with("tool:read_file("));
    }

    #[test]
    fn chunker_keeps_distinct_tool_calls_distinct() {
        // Same tool, DIFFERENT args — must remain 2 separate chunks.
        let calls = vec![
            r#"{"type":"tool_use","name":"read_file","input":{"path":"a.rs"}}"#,
            r#"{"type":"tool_use","name":"read_file","input":{"path":"b.rs"}}"#,
        ];
        let msgs: Vec<ChunkInputMessage<'_>> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| tool_msg(i, c))
            .collect();
        let cfg = ChunkConfig::default();
        let chunks = chunk_messages(&msgs, &cfg);
        assert_eq!(chunks.len(), 2, "different args → different chunks");
    }

    #[test]
    fn chunker_tags_tool_results_separately() {
        let result_blob = r#"{"type":"tool_result","tool_use_id":"abc","content":"ok"}"#;
        let msgs = vec![tool_msg(0, result_blob)];
        let chunks = chunk_messages(&msgs, &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert!(matches!(chunks[0].chunk_type, ChunkType::ToolResult));
        assert_eq!(chunks[0].content, "ok");
    }
}
