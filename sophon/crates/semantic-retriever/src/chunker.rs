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

use chrono::{DateTime, Utc};
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
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 128,
            max_size: 256,
            overlap: 32,
            extract_code: true,
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

/// Public entry point. Takes any iterator of `ChunkInputMessage` and emits
/// the chunks in document order.
pub fn chunk_messages(messages: &[ChunkInputMessage<'_>], config: &ChunkConfig) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for msg in messages {
        let mut prose = msg.content.to_string();

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
            // Strip the code blocks from the prose so they aren't re-chunked.
            prose = CODE_BLOCK_RE.replace_all(msg.content, "").to_string();
        }

        if prose.trim().is_empty() {
            continue;
        }

        let sentences = split_sentences(&prose);
        let prose_chunks = pack_sentences(
            &sentences,
            config,
            msg.index,
            msg.timestamp,
            msg.session_id,
            msg.role.to_chunk_type(),
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

fn pack_sentences(
    sentences: &[String],
    config: &ChunkConfig,
    msg_index: usize,
    timestamp: DateTime<Utc>,
    session_id: Option<&str>,
    chunk_type: ChunkType,
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

    for sentence in sentences {
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
}
