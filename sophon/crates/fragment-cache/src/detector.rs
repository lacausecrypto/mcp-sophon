use std::collections::HashMap;

use regex::Regex;

use crate::store::{Fragment, FragmentCategory};

#[derive(Debug, Clone)]
pub struct DetectedFragment {
    pub content: String,
    pub category: FragmentCategory,
    pub confidence: f32,
}

impl DetectedFragment {
    pub fn into_fragment(self) -> Fragment {
        let tags = vec![format!("confidence:{:.2}", self.confidence)];
        Fragment::from_content(self.content, self.category, tags)
    }
}

pub const BOILERPLATE_SIGNATURES: &[(&str, FragmentCategory)] = &[
    ("import React", FragmentCategory::CodeBoilerplate),
    ("from typing import", FragmentCategory::CodeBoilerplate),
    ("#[derive(", FragmentCategory::CodeBoilerplate),
    ("\"$schema\":", FragmentCategory::DataStructure),
    ("export default function", FragmentCategory::CodeBoilerplate),
];

/// Detect potentially reusable fragments in content.
pub fn detect_fragments(content: &str, min_tokens: usize) -> Vec<DetectedFragment> {
    let mut detected = Vec::new();

    let code_block_regex = Regex::new(r"```[\w]*\n([\s\S]+?)```").expect("valid regex");
    for cap in code_block_regex.captures_iter(content) {
        let block = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        if sophon_core::tokens::count_tokens(block) >= min_tokens {
            detected.push(DetectedFragment {
                content: block.to_string(),
                category: FragmentCategory::CodeBoilerplate,
                confidence: 0.8,
            });
        }
    }

    detected.extend(detect_repeated_patterns(content, min_tokens));

    for (signature, category) in BOILERPLATE_SIGNATURES {
        if content.contains(signature) {
            detected.push(DetectedFragment {
                content: signature.to_string(),
                category: *category,
                confidence: 0.65,
            });
        }
    }

    dedupe_detected(detected)
}

fn detect_repeated_patterns(content: &str, min_tokens: usize) -> Vec<DetectedFragment> {
    // Split into paragraphs (blank-line delimited) and slide a window of
    // 1..=max_window consecutive paragraphs. `max_window` used to be a hard
    // const of 12 — now it adapts to the paragraph count so that documents
    // with many small paragraphs can still detect large repeated blocks
    // (up to half of the document, bounded to 64 to keep the O(P² × C) work
    // reasonable). Callers who want stricter control can set
    // SOPHON_FRAGMENT_MAX_WINDOW in the environment.
    const HARD_CAP: usize = 64;

    let paragraphs: Vec<&str> = content
        .split("\n\n")
        .map(|p| p.trim_matches('\n'))
        .filter(|p| !p.trim().is_empty())
        .collect();

    if paragraphs.is_empty() {
        return Vec::new();
    }

    let max_window: usize = std::env::var("SOPHON_FRAGMENT_MAX_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (paragraphs.len() / 2).clamp(12, HARD_CAP));

    // Collect candidate blocks (content, token_count) that occur at least
    // twice in the original text.
    let mut candidates: Vec<(String, usize)> = Vec::new();

    for start in 0..paragraphs.len() {
        let upper = (start + max_window).min(paragraphs.len());
        for end in (start + 1)..=upper {
            let block = paragraphs[start..end].join("\n\n");
            let tokens = sophon_core::tokens::count_tokens(&block);
            if tokens < min_tokens {
                // Smaller windows starting here only get smaller; but we need
                // to keep growing because tokens may accumulate across paragraphs.
                continue;
            }
            // Count non-overlapping occurrences in the raw content.
            let occurrences = count_non_overlapping(content, &block);
            if occurrences >= 2 {
                candidates.push((block, tokens));
            }
        }
    }

    if candidates.is_empty() {
        // Fall back to the original single-paragraph pass for short repeats.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for p in &paragraphs {
            *counts.entry((*p).to_string()).or_insert(0) += 1;
        }
        return counts
            .into_iter()
            .filter(|(frag, count)| {
                *count >= 2 && sophon_core::tokens::count_tokens(frag) >= min_tokens
            })
            .map(|(frag, _)| DetectedFragment {
                content: frag,
                category: FragmentCategory::Template,
                confidence: 0.7,
            })
            .collect();
    }

    // Dedup overlapping candidates: prefer the largest block, drop any candidate
    // that is wholly contained in an already-kept block (and vice versa).
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    let mut kept: Vec<(String, usize)> = Vec::new();
    for (block, tokens) in candidates {
        let overlaps = kept
            .iter()
            .any(|(k, _)| k.contains(&block) || block.contains(k));
        if !overlaps {
            kept.push((block, tokens));
        }
    }

    kept.into_iter()
        .map(|(block, _)| DetectedFragment {
            content: block,
            category: FragmentCategory::Template,
            confidence: 0.75,
        })
        .collect()
}

/// Count how many times `needle` occurs in `haystack` without overlap.
fn count_non_overlapping(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        count += 1;
        start += pos + needle.len();
    }
    count
}

fn dedupe_detected(detected: Vec<DetectedFragment>) -> Vec<DetectedFragment> {
    let mut seen = HashMap::<String, DetectedFragment>::new();
    for item in detected {
        let key = item.content.clone();
        let replace = match seen.get(&key) {
            Some(existing) => item.confidence > existing.confidence,
            None => true,
        };
        if replace {
            seen.insert(key, item);
        }
    }
    seen.into_values().collect()
}
