use std::collections::{HashMap, HashSet};

use sophon_core::tokens::count_tokens;

use crate::{
    analyzer::{tokenize_terms, QueryAnalysis},
    parser::{ParsedPrompt, PromptSection},
};

/// Minimum cosine similarity for a section to be auto-included via
/// semantic scoring. Tuned so that genuinely related content
/// ("iteration" ↔ "loop") clears the bar (~0.65+) while noise
/// ("weather" ↔ "rust errors") stays below (~0.2–0.4).
const SEMANTIC_INCLUDE_THRESHOLD: f32 = 0.55;

/// Minimum *normalized* BM25 lexical score (0.0–1.0, top section = 1.0)
/// for a section to be auto-included by the default lexical scorer. This
/// is the query-aware default that replaces relying solely on the frozen
/// topic-keyword dictionary: a section sharing a meaningful (high-IDF)
/// query term — e.g. a function name the dictionary never heard of — is
/// pulled in even with no topic match.
const LEXICAL_INCLUDE_THRESHOLD: f32 = 0.3;

/// BM25 term-frequency saturation and length-normalization constants.
/// Standard Robertson/Sparck-Jones defaults; the "corpus" here is the
/// set of sections in the single prompt being compressed.
const BM25_K1: f32 = 1.5;
const BM25_B: f32 = 0.75;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct CompressionConfig {
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub include_headers: bool,
    pub topic_mappings: HashMap<String, Vec<String>>,
    /// Fraction of `max_tokens` the compressor tries to fill when the
    /// relevance-selected sections leave budget unused. Filling spare
    /// budget with the next-most-relevant sections strictly increases
    /// recall at zero token cost (we are already under budget) and kills
    /// the catastrophic under-fill the bench surfaced — "9 tokens emitted
    /// of ~700 allowed". Naive truncation always fills 100% of the
    /// budget; under-filling is pure loss against it. Set to 0.0 to
    /// disable backfill and keep pure relevance selection.
    pub backfill_ratio: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2_000,
            min_tokens: 0,
            include_headers: true,
            topic_mappings: default_topic_mappings(),
            backfill_ratio: 0.9,
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
#[tracing::instrument(
    skip_all,
    fields(
        sections = parsed.sections.len(),
        max_tokens = config.max_tokens,
        has_semantic_scores = section_scores.is_some(),
    ),
)]
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

    // Lexical section scoring (default, no embedder required). BM25 over
    // the prompt's own sections using the *raw* query terms, not the
    // frozen topic dictionary — this is the query-aware default that
    // catches domain-specific identifiers the keyword dictionary misses.
    // Sections above the threshold are pulled in and marked topic_matched
    // so they survive budget trimming.
    let lexical_scores = lexical_section_scores(&analysis.query_terms, &parsed.sections);
    for section in &parsed.sections {
        if let Some(&score) = lexical_scores.get(&section.id) {
            if score >= LEXICAL_INCLUDE_THRESHOLD {
                included.insert(section.id.clone());
                topic_matched.insert(section.id.clone());
            }
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
    // Drop sections whose normalized content duplicates one already
    // selected (e.g. a prompt that repeats the same block under two
    // headers). Keeping both just burns budget on a redundant copy; the
    // freed budget is reclaimed by backfill for genuinely new content.
    dedup_selected_sections(&mut selected);
    if selected.is_empty() {
        if let Some(section) = parsed.sections.first() {
            selected.push(section);
        }
    }
    trim_to_budget(
        &mut selected,
        config.max_tokens,
        &topic_matched,
        config.include_headers,
        &lexical_scores,
    );

    // Fill spare budget with the next-most-relevant sections. The target
    // is `backfill_ratio × max_tokens` (but never below an explicit
    // `min_tokens` floor, and never above the hard ceiling). Ordering is
    // by descending lexical relevance, then core-first, then document
    // order — so the budget is spent on the best remaining content. This
    // is what stops Sophon from emitting a tiny fraction of an ample
    // budget and losing to naive truncation.
    let effective_min_tokens = config.min_tokens.min(config.max_tokens);
    let backfill_target = ((config.max_tokens as f32 * config.backfill_ratio) as usize)
        .max(effective_min_tokens)
        .min(config.max_tokens);
    if count_selected_cost(&selected, config.include_headers) < backfill_target {
        backfill_to_budget(
            parsed,
            &mut selected,
            backfill_target,
            config.max_tokens,
            config.include_headers,
            &lexical_scores,
        );
    }

    // Enforce the token budget even when every remaining section is core
    // priority. trim_to_budget only removes priority>0 sections, so a single
    // giant core section (common for plain-text prompts) can overflow. In that
    // case, truncate the largest remaining section to fit.
    let owned_truncated: Option<PromptSection> = {
        let over = count_selected_cost(&selected, config.include_headers) > config.max_tokens;
        let empty_fallback = selected.is_empty();
        if over || empty_fallback {
            let source = if empty_fallback {
                parsed.sections.first()
            } else {
                selected.iter().max_by_key(|s| s.token_count).copied()
            };
            // Leave room for this section's header so the reconstructed
            // output (which wraps content in <name>…</name>) still fits.
            source.map(|s| {
                let header = if config.include_headers {
                    header_overhead(&s.name)
                } else {
                    0
                };
                truncate_section(
                    s,
                    config.max_tokens.saturating_sub(header),
                    &analysis.query_terms,
                )
            })
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
    include_headers: bool,
    lexical_scores: &HashMap<String, f32>,
) {
    if count_selected_cost(selected, include_headers) <= max_tokens {
        return;
    }

    selected.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(a.token_count.cmp(&b.token_count))
    });

    // First pass: drop non-topic-matched sections with priority > 0 (largest first).
    // These are the "generic" sections that have nothing to do with the query.
    while count_selected_cost(selected, include_headers) > max_tokens {
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

    // Second pass: only if still over budget, start removing topic-matched
    // sections — but drop the *least relevant* first (lowest lexical score),
    // so a small generic dictionary match yields before the large section
    // that actually overlaps the query. Without this, the largest section
    // goes first, which is often the most relevant one (it has the most
    // matching terms) — silently undoing the lexical selection under budget
    // pressure. Tie-break: least important (highest priority number), then
    // largest token count to free the most budget per removal.
    while count_selected_cost(selected, include_headers) > max_tokens {
        let target = selected
            .iter()
            .enumerate()
            .filter(|(_, s)| s.priority > 0)
            .min_by(|(_, a), (_, b)| {
                let sa = lexical_scores.get(&a.id).copied().unwrap_or(0.0);
                let sb = lexical_scores.get(&b.id).copied().unwrap_or(0.0);
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.priority.cmp(&b.priority))
                    .then(a.token_count.cmp(&b.token_count))
            })
            .map(|(idx, _)| idx);
        match target {
            Some(i) => {
                selected.remove(i);
            }
            None => break,
        }
    }

    selected.sort_by(|a, b| a.id.cmp(&b.id));
}

/// Fill spare budget by adding not-yet-selected sections, most-relevant
/// first, until `target_tokens` is reached. A section is added only when
/// it still fits under the hard `max_tokens` ceiling; bigger sections are
/// skipped (not a hard stop) so smaller relevant ones can still top off
/// the budget. Ordering: descending lexical score, then core-first
/// (priority ascending), then document order for stable, deterministic
/// output.
fn backfill_to_budget<'a>(
    parsed: &'a ParsedPrompt,
    selected: &mut Vec<&'a PromptSection>,
    target_tokens: usize,
    max_tokens: usize,
    include_headers: bool,
    lexical_scores: &HashMap<String, f32>,
) {
    let picked = selected
        .iter()
        .map(|s| s.id.as_str())
        .collect::<HashSet<_>>();

    let mut candidates = parsed
        .sections
        .iter()
        .enumerate()
        .filter(|(_, s)| !picked.contains(s.id.as_str()))
        .collect::<Vec<_>>();

    candidates.sort_by(|(ia, a), (ib, b)| {
        let sa = lexical_scores.get(&a.id).copied().unwrap_or(0.0);
        let sb = lexical_scores.get(&b.id).copied().unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.priority.cmp(&b.priority))
            .then(ia.cmp(ib))
    });

    // Diversity guard (MMR-lite): never spend backfill budget on a section
    // whose normalized content duplicates one already selected.
    let mut seen: HashSet<String> = selected.iter().map(|s| dedup_key(&s.content)).collect();

    let mut total = count_selected_cost(selected, include_headers);
    for (_, section) in candidates {
        if total >= target_tokens {
            break;
        }
        let key = dedup_key(&section.content);
        if seen.contains(&key) {
            continue;
        }
        let cost = section_cost(section, include_headers);
        if total + cost <= max_tokens {
            selected.push(section);
            seen.insert(key);
            total += cost;
        }
    }

    selected.sort_by(|a, b| a.id.cmp(&b.id));
}

/// Normalized content key for exact-duplicate detection: lowercased with
/// all whitespace runs collapsed to a single space. Two sections with the
/// same key carry the same information, so only one is worth budget.
fn dedup_key(content: &str) -> String {
    content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Remove sections whose normalized content duplicates an earlier one in
/// `selected`, keeping the first occurrence (document order is preserved
/// by the caller). Exact (normalized) match only — never drops a section
/// that carries distinct content.
fn dedup_selected_sections(selected: &mut Vec<&PromptSection>) {
    let mut seen: HashSet<String> = HashSet::new();
    selected.retain(|s| seen.insert(dedup_key(&s.content)));
}

/// Score every section against the query with BM25, treating the prompt's
/// sections as the document corpus. Returns scores normalized so the
/// best-matching section is 1.0 (empty map when no query term hits any
/// section). This is the deterministic, ML-free, query-aware relevance
/// signal used by default — high-IDF rare terms (identifiers, error
/// names) dominate, which is exactly what the frozen topic dictionary
/// cannot do.
fn lexical_section_scores(
    query_terms: &[String],
    sections: &[PromptSection],
) -> HashMap<String, f32> {
    let mut scores = HashMap::new();
    if query_terms.is_empty() || sections.is_empty() {
        return scores;
    }

    // Distinct query terms — repeats add nothing to a section's BM25 sum.
    let q_terms: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();

    // Tokenize each section once.
    let docs: Vec<(&str, Vec<String>)> = sections
        .iter()
        .map(|s| (s.id.as_str(), tokenize_terms(&s.content)))
        .collect();

    let n = docs.len() as f32;
    let total_len: usize = docs.iter().map(|(_, t)| t.len()).sum();
    let avgdl = if docs.is_empty() {
        0.0
    } else {
        total_len as f32 / n
    };

    // Document frequency per query term.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for (_, terms) in &docs {
        let present: HashSet<&str> = terms.iter().map(|t| t.as_str()).collect();
        for q in &q_terms {
            if present.contains(*q) {
                *df.entry(*q).or_insert(0) += 1;
            }
        }
    }

    for (id, terms) in &docs {
        let dl = terms.len() as f32;
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in terms {
            if q_terms.contains(t.as_str()) {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
        }

        let mut score = 0.0f32;
        for (term, &freq) in &tf {
            let df_t = *df.get(term).unwrap_or(&0) as f32;
            // BM25 idf with the standard +1 shift to keep it non-negative.
            let idf = (((n - df_t + 0.5) / (df_t + 0.5)) + 1.0).ln();
            let f = freq as f32;
            let denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl.max(1.0)));
            score += idf * (f * (BM25_K1 + 1.0)) / denom.max(f32::EPSILON);
        }
        if score > 0.0 {
            scores.insert((*id).to_string(), score);
        }
    }

    // Normalize so the top section is 1.0 — lets a single threshold work
    // regardless of query length or absolute BM25 magnitude.
    let max = scores.values().copied().fold(0.0f32, f32::max);
    if max > 0.0 {
        for v in scores.values_mut() {
            *v /= max;
        }
    } else {
        scores.clear();
    }

    scores
}

/// Clone a section with its content truncated to roughly `max_tokens` tokens.
///
/// When query terms are available, this keeps the lines that actually
/// overlap the query (in original order) instead of a blind prefix — so an
/// answer buried in the middle/end of a long section survives the cut
/// (bench prompt-011 lost exactly this way). With no query terms or no
/// overlap it falls back to the char-proportional prefix cut.
fn truncate_section(
    section: &PromptSection,
    max_tokens: usize,
    query_terms: &[String],
) -> PromptSection {
    if max_tokens == 0 || section.token_count <= max_tokens {
        return section.clone();
    }
    let content = query_aware_truncate(&section.content, max_tokens, query_terms)
        .unwrap_or_else(|| prefix_truncate(&section.content, section.token_count, max_tokens));
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

/// Char-proportional prefix cut. Errs on the side of *under* the budget
/// (the `* 0.9`) since re-tokenizing per cut would be slow.
fn prefix_truncate(content: &str, section_tokens: usize, max_tokens: usize) -> String {
    let ratio = max_tokens as f64 / section_tokens.max(1) as f64;
    let mut cut = ((content.chars().count() as f64) * ratio * 0.9) as usize;
    if cut == 0 {
        cut = 1;
    }
    content.chars().take(cut).collect()
}

/// Keep the highest query-overlap lines of `content` (in original order)
/// up to `max_tokens`. Returns `None` when there is nothing better than a
/// prefix cut to do: no query terms, a single line, or no line overlaps
/// the query at all. Always keeps at least the single best line, then
/// fills the remaining budget with the next-best lines; the result is
/// clamped with a prefix cut so a giant best-line can't overshoot.
fn query_aware_truncate(
    content: &str,
    max_tokens: usize,
    query_terms: &[String],
) -> Option<String> {
    if query_terms.is_empty() {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 1 {
        return None;
    }
    let q: Vec<String> = query_terms
        .iter()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    // Distinct query-term hits per line.
    let hits: Vec<usize> = lines
        .iter()
        .map(|line| {
            let lower = line.to_lowercase();
            q.iter().filter(|t| lower.contains(t.as_str())).count()
        })
        .collect();
    if !hits.iter().any(|h| *h > 0) {
        return None;
    }

    // Visit lines best-overlap first (ties → original order); always keep
    // the top line even if it alone exceeds budget (clamped afterwards),
    // then add further lines only while they fit.
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)));

    let mut keep = vec![false; lines.len()];
    let mut used = 0usize;
    for idx in order {
        let cost = count_tokens(lines[idx]) + 1; // ~newline
        if used > 0 && used + cost > max_tokens {
            continue; // a smaller later line may still fit
        }
        keep[idx] = true;
        used += cost;
        if used >= max_tokens {
            break;
        }
    }

    let kept: String = lines
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(l, _)| *l)
        .collect::<Vec<_>>()
        .join("\n");
    if kept.is_empty() {
        return None;
    }

    // Clamp: a single oversized best-line must not blow the budget.
    let kept_tokens = count_tokens(&kept);
    if kept_tokens > max_tokens {
        Some(prefix_truncate(&kept, kept_tokens, max_tokens))
    } else {
        Some(kept)
    }
}

/// Token cost of the `<name>…</name>` wrapper that [`reconstruct_prompt`]
/// adds around a section when headers are on. Budget math must include
/// this or the reconstructed output overshoots `max_tokens` by the
/// per-section header overhead.
fn header_overhead(name: &str) -> usize {
    count_tokens(&format!("<{name}>\n</{name}>\n\n"))
}

/// Effective budget cost of a section in the reconstructed output:
/// content tokens plus the header wrapper when headers are enabled.
fn section_cost(section: &PromptSection, include_headers: bool) -> usize {
    if include_headers {
        section.token_count + header_overhead(&section.name)
    } else {
        section.token_count
    }
}

fn count_selected_cost(selected: &[&PromptSection], include_headers: bool) -> usize {
    selected
        .iter()
        .map(|s| section_cost(s, include_headers))
        .sum()
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

#[cfg(test)]
mod truncate_tests {
    use super::*;

    fn section(content: &str) -> PromptSection {
        PromptSection {
            id: "s".into(),
            name: "s".into(),
            token_count: count_tokens(content),
            content: content.into(),
            topics: vec![],
            priority: 1,
            dependencies: vec![],
        }
    }

    fn section_id(id: &str, content: &str) -> PromptSection {
        PromptSection {
            id: id.into(),
            ..section(content)
        }
    }

    #[test]
    fn dedup_removes_normalized_duplicate_keeps_distinct() {
        let a = section_id("a", "The router dispatches to handlers.");
        // Same content, different whitespace/case → duplicate key.
        let b = section_id("b", "the router   dispatches\nto handlers.");
        let c = section_id("c", "A completely different section about caching.");
        let mut selected = vec![&a, &b, &c];
        dedup_selected_sections(&mut selected);
        let ids: Vec<&str> = selected.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "c"], "kept first dup + distinct, dropped b");
    }

    #[test]
    fn dedup_key_is_whitespace_and_case_insensitive() {
        assert_eq!(dedup_key("Foo  Bar\n baz"), dedup_key("foo bar BAZ"));
        assert_ne!(dedup_key("foo bar"), dedup_key("foo baz"));
    }

    #[test]
    fn query_aware_keeps_relevant_line_over_prefix() {
        // The answer sits at the END of a long section full of irrelevant
        // preamble. A blind prefix cut would drop it; query-aware must keep it.
        let mut content = String::new();
        for i in 0..40 {
            content.push_str(&format!("preamble filler line number {i} about nothing\n"));
        }
        content.push_str("the release binary size is 5.2 MB\n");
        let sec = section(&content);
        let budget = sec.token_count / 4;

        let query: Vec<String> = ["binary", "size", "release"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = truncate_section(&sec, budget, &query);

        assert!(
            out.content.contains("5.2 MB"),
            "query-aware truncation must keep the answer line, got:\n{}",
            out.content
        );
        assert!(out.token_count <= budget, "must respect the budget");
    }

    #[test]
    fn falls_back_to_prefix_without_query() {
        let content = (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let sec = section(&content);
        let budget = sec.token_count / 4;
        let out = truncate_section(&sec, budget, &[]);
        // No query → prefix behaviour: starts at the beginning.
        assert!(out.content.starts_with("line 0"));
        assert!(out.token_count <= budget);
    }

    #[test]
    fn no_overlap_falls_back_to_prefix() {
        let content = (0..30)
            .map(|i| format!("alpha beta gamma {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let sec = section(&content);
        let budget = sec.token_count / 4;
        let query = vec!["zzzznomatch".to_string()];
        let out = truncate_section(&sec, budget, &query);
        assert!(out.content.starts_with("alpha beta gamma 0"));
        assert!(out.token_count <= budget);
    }
}
