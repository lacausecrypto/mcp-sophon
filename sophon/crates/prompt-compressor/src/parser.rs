use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sophon_core::{hashing::hash_content, tokens::count_tokens};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSection {
    pub id: String,
    pub name: String,
    pub content: String,
    pub token_count: usize,
    pub topics: Vec<String>,
    pub priority: u8,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParsedPrompt {
    pub sections: Vec<PromptSection>,
    pub core_sections: Vec<String>,
    pub total_tokens: usize,
    pub content_hash: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("prompt is empty")]
    EmptyPrompt,
}

static XML_SECTION_RE: Lazy<Regex> =
    Lazy::new(|| {
        Regex::new(r"<([a-zA-Z][a-zA-Z0-9_-]*)>([^<]*)</([a-zA-Z][a-zA-Z0-9_-]*)>")
            .expect("valid XML regex")
    });
static MARKDOWN_HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^##+\s+(.+)$").expect("valid markdown regex"));
static NUMBERED_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\d+\.\s+(.+)$").expect("valid numbered regex"));
static RULE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^(RULE|IMPORTANT|NOTE):\s*(.+)$").expect("valid rule regex"));
static DEP_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?im)^depends_on:\s*(.+)$").expect("valid dep regex"));

/// Parse a system prompt into semantic sections.
pub fn parse_prompt(prompt: &str) -> Result<ParsedPrompt, ParseError> {
    if prompt.trim().is_empty() {
        return Err(ParseError::EmptyPrompt);
    }

    let raw_sections = if contains_xml_sections(prompt) {
        parse_xml_sections(prompt)
    } else if MARKDOWN_HEADER_RE.is_match(prompt) {
        parse_markdown_sections(prompt)
    } else if NUMBERED_RE.is_match(prompt) {
        parse_numbered_sections(prompt)
    } else if RULE_RE.is_match(prompt) {
        parse_rule_sections(prompt)
    } else {
        vec![("general".to_string(), prompt.trim().to_string())]
    };

    let mut sections = Vec::with_capacity(raw_sections.len());
    let mut seen_ids = HashSet::new();

    for (name, content) in raw_sections {
        let normalized_name = normalize_section_name(&name);
        let mut section_id = normalized_name.clone();
        if !seen_ids.insert(section_id.clone()) {
            let mut i = 2usize;
            while !seen_ids.insert(format!("{}_{i}", normalized_name)) {
                i += 1;
            }
            section_id = format!("{}_{i}", normalized_name);
        }

        let token_count = count_tokens(&content);
        let topics = extract_section_topics(&content);
        let dependencies = extract_dependencies(&content);
        let priority = infer_priority(&normalized_name, &content);

        sections.push(PromptSection {
            id: section_id,
            name: normalized_name,
            content,
            token_count,
            topics,
            priority,
            dependencies,
        });
    }

    let core_sections = sections
        .iter()
        .filter(|s| s.priority == 0)
        .map(|s| s.id.clone())
        .collect::<Vec<_>>();

    Ok(ParsedPrompt {
        total_tokens: sections.iter().map(|s| s.token_count).sum(),
        content_hash: hash_content(prompt),
        core_sections,
        sections,
    })
}

fn contains_xml_sections(prompt: &str) -> bool {
    XML_SECTION_RE.is_match(prompt)
}

fn parse_xml_sections(prompt: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();

    for cap in XML_SECTION_RE.captures_iter(prompt) {
        let name = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        let closing = cap.get(3).map(|m| m.as_str()).unwrap_or_default();
        if name != closing {
            continue;
        }
        let content = cap.get(2).map(|m| m.as_str()).unwrap_or_default().trim();

        // If this is a root wrapper containing nested tags, peel one level and continue.
        if (name.eq_ignore_ascii_case("system") || name.eq_ignore_ascii_case("prompt"))
            && XML_SECTION_RE.is_match(content)
        {
            sections.extend(parse_xml_sections(content));
            continue;
        }

        if !content.is_empty() {
            sections.push((name.to_string(), content.to_string()));
        }
    }

    if sections.is_empty() {
        vec![("general".to_string(), prompt.trim().to_string())]
    } else {
        sections
    }
}

fn parse_markdown_sections(prompt: &str) -> Vec<(String, String)> {
    let mut positions = Vec::new();
    for m in MARKDOWN_HEADER_RE.find_iter(prompt) {
        positions.push((m.start(), m.end()));
    }

    if positions.is_empty() {
        return vec![("general".to_string(), prompt.trim().to_string())];
    }

    let mut sections = Vec::new();

    for (idx, (start, end)) in positions.iter().enumerate() {
        let header_line = &prompt[*start..*end];
        let name = header_line.trim_start_matches('#').trim();
        let content_start = *end;
        let content_end = positions
            .get(idx + 1)
            .map(|(next_start, _)| *next_start)
            .unwrap_or(prompt.len());
        let content = prompt[content_start..content_end].trim();

        if !content.is_empty() {
            sections.push((name.to_string(), content.to_string()));
        }
    }

    if sections.is_empty() {
        vec![("general".to_string(), prompt.trim().to_string())]
    } else {
        sections
    }
}

fn parse_numbered_sections(prompt: &str) -> Vec<(String, String)> {
    let mut positions = Vec::new();
    for cap in NUMBERED_RE.captures_iter(prompt) {
        if let Some(m) = cap.get(0) {
            let name = cap
                .get(1)
                .map(|x| x.as_str().trim().to_string())
                .unwrap_or_else(|| "section".to_string());
            positions.push((m.start(), m.end(), name));
        }
    }

    if positions.is_empty() {
        return vec![("general".to_string(), prompt.trim().to_string())];
    }

    let mut sections = Vec::new();
    for (idx, (_, end, name)) in positions.iter().enumerate() {
        let content_end = positions
            .get(idx + 1)
            .map(|(start, _, _)| *start)
            .unwrap_or(prompt.len());
        let content = prompt[*end..content_end].trim();
        if !content.is_empty() {
            sections.push((name.clone(), content.to_string()));
        }
    }

    sections
}

fn parse_rule_sections(prompt: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    for cap in RULE_RE.captures_iter(prompt) {
        let label = cap.get(1).map(|m| m.as_str()).unwrap_or("rule");
        let content = cap.get(2).map(|m| m.as_str()).unwrap_or("{}");
        sections.push((label.to_string(), content.trim().to_string()));
    }

    if sections.is_empty() {
        vec![("general".to_string(), prompt.trim().to_string())]
    } else {
        sections
    }
}

fn normalize_section_name(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn extract_dependencies(content: &str) -> Vec<String> {
    DEP_RE
        .captures_iter(content)
        .flat_map(|cap| {
            cap.get(1)
                .map(|m| {
                    m.as_str()
                        .split(',')
                        .map(|s| normalize_section_name(s))
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect()
}

fn infer_priority(name: &str, content: &str) -> u8 {
    let name_lower = name.to_lowercase();
    let content_lower = content.to_lowercase();

    if name_lower.contains("core")
        || name_lower.contains("identity")
        || name_lower.contains("safety")
        || content_lower.contains("you are")
        || content_lower.contains("you must")
        || content_lower.contains("must not")
        || content_lower.contains("never")
    {
        0
    } else if content_lower.contains("important")
        || content_lower.contains("policy")
        || content_lower.contains("required")
        || name_lower.contains("policy")
    {
        1
    } else if content_lower.contains("style")
        || content_lower.contains("format")
        || content_lower.contains("guideline")
        || name_lower.contains("format")
    {
        2
    } else {
        3
    }
}

/// Analyze section content to extract topics.
pub fn extract_section_topics(content: &str) -> Vec<String> {
    static TOPIC_KEYWORDS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
        HashMap::from([
            (
                "coding",
                vec!["code", "function", "class", "variable", "syntax", "programming"],
            ),
            (
                "safety",
                vec!["safe", "harm", "dangerous", "inappropriate", "refuse"],
            ),
            (
                "formatting",
                vec!["format", "markdown", "bullet", "list", "header"],
            ),
            (
                "math",
                vec!["calculate", "equation", "formula", "mathematical"],
            ),
            (
                "creative_writing",
                vec!["write", "story", "creative", "fiction", "poem"],
            ),
            (
                "data_analysis",
                vec!["table", "dataset", "csv", "analysis", "statistics"],
            ),
            (
                "conversation",
                vec!["tone", "friendly", "conversation", "personality", "style"],
            ),
            (
                "python",
                vec!["python", "pandas", "numpy", "py"],
            ),
            (
                "javascript",
                vec!["javascript", "typescript", "node", "react"],
            ),
        ])
    });

    let normalized = content.to_lowercase();
    let mut topics = Vec::new();

    for (topic, keywords) in TOPIC_KEYWORDS.iter() {
        if keywords.iter().any(|keyword| normalized.contains(keyword)) {
            topics.push((*topic).to_string());
        }
    }

    if topics.is_empty() {
        topics.push("general".to_string());
    }

    topics
}
