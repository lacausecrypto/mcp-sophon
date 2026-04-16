use std::collections::{HashMap, HashSet};

use sophon_core::tokens::count_tokens;

use crate::{
    analyzer::{Complexity, QueryAnalysis},
    parser::{ParsedPrompt, PromptSection},
};

/// Minimum cosine similarity for a section to be auto-included via
/// semantic scoring. Tuned so that genuinely related content
/// ("iteration" ↔ "loop") clears the bar (~0.65+) while noise
/// ("weather" ↔ "rust errors") stays below (~0.2–0.4).
const SEMANTIC_INCLUDE_THRESHOLD: f32 = 0.55;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct CompressionConfig {
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub include_headers: bool,
    pub topic_mappings: HashMap<String, Vec<String>>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2_000,
            min_tokens: 0,
            include_headers: true,
            topic_mappings: default_topic_mappings(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub compressed_prompt: String,
    pub token_count: usize,
    pub included_sections: Vec<String>,
    pub excluded_sections: Vec<String>,
    pub compression_ratio: f32,
}

pub const DEFAULT_TOPIC_MAPPINGS: &[(&str, &[&str])] = &[
    (
        "coding",
        &["code_formatting", "programming_*", "technical_*"],
    ),
    ("python", &["code_formatting", "python_*", "programming_*"]),
    ("javascript", &["code_formatting", "javascript_*", "web_*"]),
    ("math", &["math_*", "formatting_numbers", "latex_*"]),
    (
        "creative_writing",
        &["creative_*", "writing_style", "tone_*"],
    ),
    ("data_analysis", &["data_*", "tables", "charts", "csv_*"]),
    ("conversation", &["tone_*", "personality", "response_style"]),
    ("safety", &["safety_*", "refusal_*", "harmful_*"]),
];

fn default_topic_mappings() -> HashMap<String, Vec<String>> {
    DEFAULT_TOPIC_MAPPINGS
        .iter()
        .map(|(topic, patterns)| {
            (
                (*topic).to_string(),
                patterns
                    .iter()
                    .map(|p| (*p).to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Compress a prompt based on query analysis.
///
/// `section_scores` is an optional map of `section_id → cosine_similarity`
/// produced by an external embedder (e.g. BGE-small). When provided,
/// sections with a score above `SEMANTIC_INCLUDE_THRESHOLD` are auto-
/// included alongside the keyword-matched ones — this closes the gap
/// where the keyword dictionary misses semantic equivalents ("loop" ≠
/// "iteration" in keyword space, but ≈ 0.85 in embedding space).
pub fn compress_prompt(
    parsed: &ParsedPrompt,
    analysis: &QueryAnalysis,
    config: &CompressionConfig,
    section_scores: Option<&HashMap<String, f32>>,
) -> CompressionResult {
    let mut included = HashSet::new();
    // Sections that were explicitly matched by topic routing — these are the
    // whole reason the caller ran compression, so they must survive budget
    // trimming longer than generic non-core sections.
    let mut topic_matched: HashSet<String> = HashSet::new();

    for core in &parsed.core_sections {
        included.insert(core.clone());
    }

    // Topic-driven section matching from configured pattern rules.
    for topic in &analysis.topics {
        if let Some(patterns) = config.topic_mappings.get(topic) {
            include_by_patterns(parsed, patterns, &mut included, &mut topic_matched);
        }
    }

    // Direct section-topic match fallback.
    for section in &parsed.sections {
        if section
            .topics
            .iter()
            .any(|topic| analysis.topics.iter().any(|t| t == topic))
        {
            included.insert(section.id.clone());
            topic_matched.insert(section.id.clone());
        }
    }

    // Semantic section scoring — when an embedder provides cosine
    // similarity scores, include sections above the threshold even if
    // no keyword matched. This is the "loop ≈ iteration" fix.
    if let Some(scores) = section_scores {
        for section in &parsed.sections {
            if let Some(&score) = scores.get(&section.id) {
                if score >= SEMANTIC_INCLUDE_THRESHOLD {
                    included.insert(section.id.clone());
                    topic_matched.insert(section.id.clone());
                }
            }
        }
    }

    resolve_dependencies(parsed, &mut included);

    let mut selected = select_sections_in_order(parsed, &included);
    if selected.is_empty() {
        if let Some(section) = parsed.sections.first() {
            selected.push(section);
        }
    }
    trim_to_budget(&mut selected, config.max_tokens, &topic_matched);

    let effective_min_tokens = config.min_tokens.min(config.max_tokens);
    if count_selected_tokens(&selected) < effective_min_tokens {
        backfill_to_minimum(parsed, &mut selected, effective_min_tokens);
    }

    // Complexity scaling
    if matches!(analysis.complexity, Complexity::Simple) {
        keep_top_n_non_core(parsed, &mut selected, 2);
    }

    // Enforce the token budget even when every remaining section is core
    // priority. trim_to_budget only removes priority>0 sections, so a single
    // giant core section (common for plain-text prompts) can overflow. In that
    // case, truncate the largest remaining section to fit.
    let owned_truncated: Option<PromptSection> = {
        let over = count_selected_tokens(&selected) > config.max_tokens;
        let empty_fallback = selected.is_empty();
        if over || empty_fallback {
            let source = if empty_fallback {
                parsed.sections.first()
            } else {
                selected.iter().max_by_key(|s| s.token_count).copied()
            };
            source.map(|s| truncate_section(s, config.max_tokens))
        } else {
            None
        }
    };
    if let Some(ref trunc) = owned_truncated {
        // Replace the section with matching id (or add it if it's the empty-fallback path).
        if let Some(pos) = selected.iter().position(|s| s.id == trunc.id) {
            selected[pos] = trunc;
        } else {
            selected.push(trunc);
        }
    }

    let compressed_prompt = reconstruct_prompt(&selected, config.include_headers);
    let token_count = count_tokens(&compressed_prompt);

    let included_sections = selected.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
    let excluded_sections = parsed
        .sections
        .iter()
        .filter(|s| !included_sections.iter().any(|id| id == &s.id))
        .map(|s| s.id.clone())
        .collect::<Vec<_>>();

    // compression_ratio = compressed / original, clamped to [0.0, 1.0].
    // Lower is better. 0.0 = fully elided, 1.0 = no compression achieved.
    let compression_ratio = if parsed.total_tokens == 0 {
        1.0
    } else if token_count == 0 {
        0.0
    } else {
        (token_count as f32 / parsed.total_tokens as f32).min(1.0)
    };

    CompressionResult {
        compressed_prompt,
        token_count,
        included_sections,
        excluded_sections,
        compression_ratio,
    }
}

fn include_by_patterns(
    parsed: &ParsedPrompt,
    patterns: &[String],
    included: &mut HashSet<String>,
    topic_matched: &mut HashSet<String>,
) {
    for pattern in patterns {
        for section in &parsed.sections {
            if matches_pattern(&section.id, pattern) || matches_pattern(&section.name, pattern) {
                included.insert(section.id.clone());
                topic_matched.insert(section.id.clone());
            }
        }
    }
}

fn matches_pattern(value: &str, pattern: &str) -> bool {
    if pattern.ends_with('*') {
        let prefix = pattern.trim_end_matches('*');
        value.starts_with(prefix)
    } else {
        value == pattern
    }
}

fn resolve_dependencies(parsed: &ParsedPrompt, included: &mut HashSet<String>) {
    let mut changed = true;
    while changed {
        changed = false;
        let current = included.clone();

        for section_id in current {
            if let Some(section) = parsed.sections.iter().find(|s| s.id == section_id) {
                for dep in &section.dependencies {
                    if parsed.sections.iter().any(|candidate| candidate.id == *dep)
                        && included.insert(dep.clone())
                    {
                        changed = true;
                    }
                }
            }
        }
    }
}

fn select_sections_in_order<'a>(
    parsed: &'a ParsedPrompt,
    included: &HashSet<String>,
) -> Vec<&'a PromptSection> {
    parsed
        .sections
        .iter()
        .filter(|section| included.contains(&section.id))
        .collect()
}

fn trim_to_budget(
    selected: &mut Vec<&PromptSection>,
    max_tokens: usize,
    topic_matched: &HashSet<String>,
) {
    if count_selected_tokens(selected) <= max_tokens {
        return;
    }

    selected.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(a.token_count.cmp(&b.token_count))
    });

    // First pass: drop non-topic-matched sections with priority > 0 (largest first).
    // These are the "generic" sections that have nothing to do with the query.
    while count_selected_tokens(selected) > max_tokens {
        let idx = selected
            .iter()
            .enumerate()
            .rfind(|(_, s)| s.priority > 0 && !topic_matched.contains(&s.id))
            .map(|(idx, _)| idx);
        match idx {
            Some(i) => {
                selected.remove(i);
            }
            None => break,
        }
    }

    // Second pass: only if still over budget, start removing topic-matched sections
    // (lowest-priority / largest first). This preserves at least some topic signal
    // but respects the hard max_tokens ceiling.
    while count_selected_tokens(selected) > max_tokens {
        let idx = selected
            .iter()
            .enumerate()
            .rfind(|(_, s)| s.priority > 0)
            .map(|(idx, _)| idx);
        match idx {
            Some(i) => {
                selected.remove(i);
            }
            None => break,
        }
    }

    selected.sort_by(|a, b| a.id.cmp(&b.id));
}

fn backfill_to_minimum<'a>(
    parsed: &'a ParsedPrompt,
    selected: &mut Vec<&'a PromptSection>,
    min_tokens: usize,
) {
    let mut picked = selected
        .iter()
        .map(|s| s.id.as_str())
        .collect::<HashSet<_>>();
    let candidates = parsed
        .sections
        .iter()
        .filter(|s| !picked.contains(s.id.as_str()))
        .collect::<Vec<_>>();

    for section in candidates {
        selected.push(section);
        picked.insert(section.id.as_str());
        if count_selected_tokens(selected) >= min_tokens {
            break;
        }
    }

    selected.sort_by(|a, b| a.id.cmp(&b.id));
}

fn keep_top_n_non_core<'a>(
    parsed: &'a ParsedPrompt,
    selected: &mut Vec<&'a PromptSection>,
    n: usize,
) {
    let core_ids = parsed
        .core_sections
        .iter()
        .map(|s| s.as_str())
        .collect::<HashSet<_>>();

    let mut non_core = selected
        .iter()
        .filter(|s| !core_ids.contains(s.id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    non_core.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(a.token_count.cmp(&b.token_count))
    });

    let keep_non_core = non_core
        .into_iter()
        .take(n)
        .map(|s| s.id.clone())
        .collect::<HashSet<_>>();

    selected.retain(|s| core_ids.contains(s.id.as_str()) || keep_non_core.contains(&s.id));
}

/// Clone a section with its content truncated to roughly `max_tokens` tokens.
/// Uses a char-proportional cut since re-tokenizing per cut would be slow;
/// errs on the side of *under* the budget.
fn truncate_section(section: &PromptSection, max_tokens: usize) -> PromptSection {
    if max_tokens == 0 || section.token_count <= max_tokens {
        return section.clone();
    }
    let ratio = max_tokens as f64 / section.token_count as f64;
    let mut cut = ((section.content.chars().count() as f64) * ratio * 0.9) as usize;
    if cut == 0 {
        cut = 1;
    }
    let content: String = section.content.chars().take(cut).collect();
    let token_count = count_tokens(&content);
    PromptSection {
        id: section.id.clone(),
        name: section.name.clone(),
        content,
        token_count,
        topics: section.topics.clone(),
        priority: section.priority,
        dependencies: section.dependencies.clone(),
    }
}

fn count_selected_tokens(selected: &[&PromptSection]) -> usize {
    selected.iter().map(|s| s.token_count).sum()
}

fn reconstruct_prompt(selected: &[&PromptSection], include_headers: bool) -> String {
    let mut output = String::new();
    for section in selected {
        if include_headers {
            output.push_str(&format!("<{}>\n", section.name));
            output.push_str(section.content.trim());
            output.push_str(&format!("\n</{}>\n\n", section.name));
        } else {
            output.push_str(section.content.trim());
            output.push_str("\n\n");
        }
    }
    output.trim().to_string()
}
