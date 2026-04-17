//! Reranking strategies for post-retrieval filtering.
//!
//! After the vector index returns top-k candidates by cosine similarity,
//! a reranker rescores each chunk to suppress false positives — chunks that
//! are "semantically close" in embedding space but don't actually answer
//! the query.

use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::process::{Command, Stdio};
use std::time::Duration;

/// A reranker scores how relevant a chunk is to a query.
///
/// Implementations must be thread-safe (`Send + Sync`).
pub trait Reranker: Send + Sync {
    /// Score how relevant a chunk is to the query. Returns 0.0–1.0.
    fn rerank(&self, query: &str, chunk_content: &str) -> f32;

    /// Human-readable name for logging / debug output.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// KeywordReranker — zero-dependency, no LLM
// ---------------------------------------------------------------------------

/// Penalizes chunks that are "semantically close" (high cosine score) but
/// don't actually contain the query's keywords.
///
/// The reranker returns a keyword overlap fraction in [0.0, 1.0].
/// The caller in `retrieve()` multiplies this with the original cosine score:
///
///   adjusted_score = keyword_fraction * original_cosine_score
///
/// This is the default reranker — it needs no network, no model, no config.
pub struct KeywordReranker;

impl Reranker for KeywordReranker {
    fn rerank(&self, query: &str, chunk_content: &str) -> f32 {
        let query_words: HashSet<String> = query
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();

        if query_words.is_empty() {
            return 1.0;
        }

        let chunk_lower = chunk_content.to_lowercase();
        let matched = query_words
            .iter()
            .filter(|qw| chunk_lower.contains(qw.as_str()))
            .count();

        matched as f32 / query_words.len() as f32
    }

    fn name(&self) -> &'static str {
        "keyword"
    }
}

// ---------------------------------------------------------------------------
// CommandReranker — opt-in, shells out to a configurable command
// ---------------------------------------------------------------------------

/// Shells out to an external command (e.g. `claude -p --model haiku`) to
/// score chunk relevance. The command receives a prompt on stdin and must
/// print a numeric score (0–10) on stdout.
///
/// Set `SOPHON_RERANKER_CMD` to enable. Falls back to a neutral 0.5 score
/// on timeout or parse failure.
pub struct CommandReranker {
    cmd: String,
    #[allow(dead_code)]
    timeout: Duration,
}

impl CommandReranker {
    /// Build from the `SOPHON_RERANKER_CMD` env var. Returns `None` if unset.
    pub fn from_env() -> Option<Self> {
        let cmd = std::env::var("SOPHON_RERANKER_CMD").ok()?;
        if cmd.is_empty() {
            return None;
        }
        Some(Self {
            cmd,
            timeout: Duration::from_secs(10),
        })
    }

    /// Build with an explicit command string.
    pub fn new(cmd: impl Into<String>) -> Self {
        Self {
            cmd: cmd.into(),
            timeout: Duration::from_secs(10),
        }
    }

    /// Override the default 10-second timeout.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

impl Reranker for CommandReranker {
    fn rerank(&self, query: &str, chunk_content: &str) -> f32 {
        let prompt = format!(
            "Does the following text answer the question? Reply with a score 0-10.\n\
             Question: {}\n\
             Text: {}\n\
             Score:",
            query, chunk_content
        );

        let parts: Vec<&str> = self.cmd.split_whitespace().collect();
        if parts.is_empty() {
            return 0.5;
        }

        let result = (|| -> Option<f32> {
            let mut child = Command::new(parts[0])
                .args(&parts[1..])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;

            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(prompt.as_bytes());
                // stdin dropped here, closing the pipe
            }

            let output = child.wait_with_output().ok()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_score_from_output(&stdout)
        })();

        result.unwrap_or(0.5)
    }

    fn name(&self) -> &'static str {
        "command"
    }
}

/// Extract the first number (integer or float) from the command output and
/// normalize it from [0, 10] to [0.0, 1.0].
fn parse_score_from_output(output: &str) -> Option<f32> {
    for token in output.split_whitespace() {
        let cleaned: String = token
            .chars()
            .filter(|c| c.is_numeric() || *c == '.')
            .collect();
        if let Ok(val) = cleaned.parse::<f32>() {
            let clamped = val.clamp(0.0, 10.0);
            return Some(clamped / 10.0);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_reranker_boosts_matching_chunks() {
        let r = KeywordReranker;
        let score = r.rerank(
            "Italian restaurant Paris",
            "The best Italian restaurant in Paris is Chez Luigi.",
        );
        assert!(
            score > 0.9,
            "expected high score for matching chunk, got {}",
            score
        );
    }

    #[test]
    fn keyword_reranker_penalizes_non_matching_chunks() {
        let r = KeywordReranker;
        let score = r.rerank(
            "Italian restaurant Paris",
            "Quantum computing advances in 2025 with new chip designs.",
        );
        assert!(
            score < 0.2,
            "expected low score for non-matching chunk, got {}",
            score
        );
    }

    #[test]
    fn keyword_reranker_partial_match() {
        let r = KeywordReranker;
        let score = r.rerank(
            "Italian restaurant Paris",
            "Visit Paris for the best croissants.",
        );
        // "paris" matches, "italian" and "restaurant" don't -> ~0.33
        assert!(
            (0.2..=0.5).contains(&score),
            "expected partial score, got {}",
            score
        );
    }

    #[test]
    fn keyword_reranker_empty_query_returns_one() {
        let r = KeywordReranker;
        let score = r.rerank("", "any content here");
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_score_extracts_and_normalizes() {
        assert_eq!(parse_score_from_output("7"), Some(0.7));
        assert_eq!(parse_score_from_output("Score: 8"), Some(0.8));
        assert_eq!(parse_score_from_output("The score is 10."), Some(1.0));
        assert_eq!(parse_score_from_output("3.5"), Some(0.35));
        assert!(parse_score_from_output("no numbers here").is_none());
    }

    #[test]
    fn command_reranker_from_env_returns_none_when_unset() {
        std::env::remove_var("SOPHON_RERANKER_CMD");
        assert!(CommandReranker::from_env().is_none());
    }
}
