use std::collections::{HashMap, HashSet};

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

    // Near-duplicate pre-pass: catch paragraphs that differ only by a number
    // or minor token variation (Jaccard >= 0.95).
    detected.extend(detect_near_duplicates(content, min_tokens, 0.95));

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

/// Jaccard similarity over whitespace tokens.
fn jaccard(a: &str, b: &str) -> f32 {
    let ta: HashSet<&str> = a.split_whitespace().collect();
    let tb: HashSet<&str> = b.split_whitespace().collect();
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Near-duplicate detection: for each unique paragraph, check Jaccard
/// similarity against other paragraphs. If similarity >= threshold, treat
/// the first occurrence as a reusable fragment.
fn detect_near_duplicates(
    content: &str,
    min_tokens: usize,
    threshold: f32,
) -> Vec<DetectedFragment> {
    let paragraphs: Vec<&str> = content
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    if paragraphs.len() < 2 {
        return Vec::new();
    }

    let mut detected = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();

    for i in 0..paragraphs.len() {
        if used.contains(&i) {
            continue;
        }
        let tokens_i = sophon_core::tokens::count_tokens(paragraphs[i]);
        if tokens_i < min_tokens {
            continue;
        }
        let mut has_near_dup = false;
        for j in (i + 1)..paragraphs.len() {
            if used.contains(&j) {
                continue;
            }
            // Skip exact matches — those are handled by the existing pass.
            if paragraphs[i] == paragraphs[j] {
                continue;
            }
            if jaccard(paragraphs[i], paragraphs[j]) >= threshold {
                has_near_dup = true;
                used.insert(j);
            }
        }
        if has_near_dup {
            used.insert(i);
            detected.push(DetectedFragment {
                content: paragraphs[i].to_string(),
                category: FragmentCategory::Template,
                confidence: 0.72,
            });
        }
    }

    detected
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_duplicate_paragraphs_detected() {
        // Two paragraphs that differ only by a single number should be
        // caught as near-duplicates (Jaccard >= 0.95). The paragraph must
        // be long enough that one differing token keeps Jaccard above 0.95.
        // With ~40 shared tokens and 1 differing, Jaccard = 40/42 ≈ 0.952.
        let shared = "The quick brown fox jumped over the lazy dog and then \
                       ran across the wide open meadow before stopping at the \
                       edge of the forest where the tall oak trees swayed gently \
                       in the warm afternoon breeze while birds sang their songs";
        let content = format!("{shared} item 1\n\n{shared} item 2");
        let detected = detect_fragments(&content, 1);
        let near_dup = detected.iter().find(|d| d.content.contains("item 1"));
        assert!(
            near_dup.is_some(),
            "expected near-duplicate detection for paragraphs differing by a single number, got: {:?}",
            detected,
        );
    }

    #[test]
    fn exact_duplicates_still_detected() {
        let content = "repeated paragraph here\n\nrepeated paragraph here";
        let detected = detect_fragments(content, 1);
        assert!(
            !detected.is_empty(),
            "exact duplicates should still be detected",
        );
    }

    #[test]
    fn dissimilar_paragraphs_not_flagged() {
        let content = "The quick brown fox jumped over the lazy dog.\n\n\
                        A completely different sentence about unrelated topics.";
        let detected = detect_near_duplicates(content, 1, 0.95);
        assert!(
            detected.is_empty(),
            "dissimilar paragraphs should not be flagged as near-duplicates, got: {:?}",
            detected,
        );
    }

    #[test]
    fn jaccard_identical_strings() {
        assert!((jaccard("a b c", "a b c") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_strings() {
        assert!((jaccard("a b c", "x y z")).abs() < f32::EPSILON);
    }
}
