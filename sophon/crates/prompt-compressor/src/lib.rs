pub mod analyzer;
pub mod cache;
pub mod compressor;
pub mod parser;

use std::collections::HashMap;

use analyzer::{analyze_query, ConversationMessage};
use cache::PromptCache;
pub use compressor::CompressionConfig;
use compressor::{compress_prompt, CompressionResult};
use parser::{parse_prompt, ParseError, ParsedPrompt};

/// High-level prompt compression facade.
#[derive(Debug)]
pub struct PromptCompressor {
    cache: PromptCache,
    default_config: CompressionConfig,
}

impl Default for PromptCompressor {
    fn default() -> Self {
        Self {
            cache: PromptCache::new(256),
            default_config: CompressionConfig::default(),
        }
    }
}

impl PromptCompressor {
    pub fn with_config(cfg: CompressionConfig) -> Self {
        Self {
            cache: PromptCache::new(256),
            default_config: cfg,
        }
    }

    pub fn parse(&self, prompt: &str) -> Result<ParsedPrompt, ParseError> {
        parse_prompt(prompt)
    }

    pub fn compress(
        &mut self,
        prompt: &str,
        query: &str,
        history: Option<&[ConversationMessage]>,
        max_tokens: Option<usize>,
    ) -> Result<CompressionResult, ParseError> {
        self.compress_with_scores(prompt, query, history, max_tokens, None)
    }

    /// Like [`compress`] but accepts optional per-section semantic
    /// similarity scores (e.g. from BGE-small). Sections with a score
    /// above the internal threshold are auto-included even if no
    /// keyword matched. This is the "loop ≈ iteration" fix for
    /// structured prompts.
    pub fn compress_with_scores(
        &mut self,
        prompt: &str,
        query: &str,
        history: Option<&[ConversationMessage]>,
        max_tokens: Option<usize>,
        section_scores: Option<&HashMap<String, f32>>,
    ) -> Result<CompressionResult, ParseError> {
        let parsed = parse_prompt(prompt)?;
        let analysis = analyze_query(query, history);
        let mut config = self.default_config.clone();
        if let Some(max) = max_tokens {
            config.max_tokens = max;
        }

        // When no semantic scores are provided, the cache is still valid
        // (same keyword-only path). When scores are provided, skip the
        // cache because the scores change per query embedding.
        if section_scores.is_none() {
            let cache_key = self.cache.make_key(
                &parsed.content_hash,
                query,
                config.max_tokens,
                config.min_tokens,
            );
            if let Some(cached) = self.cache.get(&cache_key) {
                return Ok(cached);
            }

            let result = compress_prompt(&parsed, &analysis, &config, None);
            self.cache.insert(cache_key, result.clone());
            return Ok(result);
        }

        Ok(compress_prompt(&parsed, &analysis, &config, section_scores))
    }
}
