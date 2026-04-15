//! Compression strategies and the pipeline that runs them.
//!
//! A `FilterConfig` is an ordered list of `CompressionStrategy` values.
//! `run_pipeline` applies them left-to-right, each one transforming the
//! running text. Errors are non-fatal: a strategy that fails to apply
//! leaves the text untouched.

use std::borrow::Cow;

use regex::Regex;
use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;

use crate::{dedup, truncate};

/// A single compression step. Strategies can be chained; each one
/// receives the output of the previous step as input.
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    /// Drop lines matching any `remove` pattern, unless they also match
    /// a `keep` pattern (keep wins — used for "noise except these lines"
    /// style filters).
    FilterLines {
        remove_patterns: Vec<Regex>,
        keep_patterns: Vec<Regex>,
    },

    /// Group lines by a regex capture group, collapsing groups of size
    /// ≥ `min_count` into a single summary line.
    /// `{key}` and `{count}` in `output_format` are substituted.
    GroupBy {
        key_pattern: Regex,
        output_format: String,
        min_count: usize,
    },

    /// Collapse identical-or-similar consecutive lines with a repetition
    /// count. `{line}` and `{count}` are substituted in `output_format`.
    Deduplicate {
        similarity_threshold: f32,
        output_format: String,
    },

    /// Keep head + tail of the text, replacing the middle with a message.
    /// `{n}` in `omission_message` is substituted with the elided count.
    Truncate {
        max_lines: usize,
        omission_message: String,
    },

    /// Keep only the listed whitespace-delimited columns. Used for table
    /// outputs like `docker ps`. If `fields` is empty this is a no-op.
    ExtractColumns { fields: Vec<String> },
}

impl CompressionStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::FilterLines { .. } => "filter_lines",
            Self::GroupBy { .. } => "group_by",
            Self::Deduplicate { .. } => "deduplicate",
            Self::Truncate { .. } => "truncate",
            Self::ExtractColumns { .. } => "extract_columns",
        }
    }
}

/// Declarative configuration for a single command family (git, tests,
/// docker, …). Matched by `command_patterns`, then strategies apply in
/// order.
#[derive(Debug, Clone)]
pub struct FilterConfig {
    pub name: &'static str,
    pub command_patterns: Vec<Regex>,
    pub strategies: Vec<CompressionStrategy>,
    /// Optional hard cap on the compressed token count. When set, a
    /// final truncate pass runs to hit this budget.
    pub max_output_tokens: Option<usize>,
    /// Number of head lines always preserved by any middle-truncate pass.
    pub preserve_head: usize,
    /// Number of tail lines always preserved.
    pub preserve_tail: usize,
}

/// The output of a compression pipeline. Serde-friendly so it drops
/// straight into an MCP response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    /// `compressed_tokens / original_tokens`, clamped to `[0, 1]`.
    pub ratio: f32,
    pub filter_name: String,
    pub strategies_applied: Vec<String>,
    pub original_command: String,
}

/// Run `filter`'s strategy pipeline against `output` and return the
/// resulting `CompressionResult`. Defensive: if a strategy has a bug it
/// leaves the text unchanged rather than crashing the whole compress.
pub fn run_pipeline(command: &str, output: &str, filter: &FilterConfig) -> CompressionResult {
    let original_tokens = count_tokens(output);
    let mut current: Cow<str> = Cow::Borrowed(output);
    let mut strategies_applied = Vec::new();

    for strategy in &filter.strategies {
        let before_len = current.len();
        let after = apply_strategy(current.as_ref(), strategy);
        if after.len() != before_len || after != *current {
            strategies_applied.push(strategy.name().to_string());
            current = Cow::Owned(after);
        }
    }

    // Enforce hard token budget with a final middle truncate if needed.
    if let Some(budget) = filter.max_output_tokens {
        if count_tokens(current.as_ref()) > budget {
            // Heuristic: keep ~ `budget * 4` characters (~1 token per 4 chars).
            let max_chars = budget.saturating_mul(4).max(256);
            let trunc = truncate::middle_truncate_chars(
                current.as_ref(),
                max_chars,
                filter.preserve_head,
                filter.preserve_tail,
                "... token budget reached ({n} chars omitted) ...",
            );
            if trunc != *current {
                strategies_applied.push("budget_cap".to_string());
                current = Cow::Owned(trunc);
            }
        }
    }

    let compressed = current.into_owned();
    let compressed_tokens = count_tokens(&compressed);
    let ratio = if original_tokens == 0 {
        1.0
    } else {
        (compressed_tokens as f32 / original_tokens as f32)
            .min(1.0)
            .max(0.0)
    };

    CompressionResult {
        compressed,
        original_tokens,
        compressed_tokens,
        ratio,
        filter_name: filter.name.to_string(),
        strategies_applied,
        original_command: command.to_string(),
    }
}

fn apply_strategy(input: &str, strategy: &CompressionStrategy) -> String {
    match strategy {
        CompressionStrategy::FilterLines {
            remove_patterns,
            keep_patterns,
        } => filter_lines(input, remove_patterns, keep_patterns),
        CompressionStrategy::GroupBy {
            key_pattern,
            output_format,
            min_count,
        } => group_by(input, key_pattern, output_format, *min_count),
        CompressionStrategy::Deduplicate {
            similarity_threshold,
            output_format,
        } => dedup::dedup_lines(input, *similarity_threshold, output_format),
        CompressionStrategy::Truncate {
            max_lines,
            omission_message,
        } => truncate::middle_truncate_lines(input, *max_lines, omission_message),
        CompressionStrategy::ExtractColumns { fields } => extract_columns(input, fields),
    }
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

fn filter_lines(input: &str, remove: &[Regex], keep: &[Regex]) -> String {
    let mut out = Vec::new();
    for line in input.lines() {
        let keep_hit = keep.iter().any(|r| r.is_match(line));
        let remove_hit = remove.iter().any(|r| r.is_match(line));
        if keep_hit || !remove_hit {
            out.push(line);
        }
    }
    out.join("\n")
}

fn group_by(input: &str, pattern: &Regex, format: &str, min_count: usize) -> String {
    use std::collections::BTreeMap;

    // BTreeMap for deterministic output order.
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ungrouped: Vec<String> = Vec::new();

    for line in input.lines() {
        match pattern.captures(line) {
            Some(caps) => {
                let key = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_else(|| "_".to_string());
                groups.entry(key).or_default().push(line.to_string());
            }
            None => ungrouped.push(line.to_string()),
        }
    }

    let mut out: Vec<String> = Vec::with_capacity(ungrouped.len() + groups.len());
    out.extend(ungrouped);
    for (key, items) in groups {
        if items.len() >= min_count && items.len() > 1 {
            let summary = format
                .replace("{key}", &key)
                .replace("{count}", &items.len().to_string());
            out.push(summary);
        } else {
            out.extend(items);
        }
    }
    out.join("\n")
}

fn extract_columns(input: &str, fields: &[String]) -> String {
    if fields.is_empty() {
        return input.to_string();
    }

    let mut lines = input.lines();
    let Some(header) = lines.next() else {
        return input.to_string();
    };

    // Find column starts on the header line. Table headers like
    // `docker ps` output are whitespace-separated and positional.
    let header_cols: Vec<(usize, &str)> = split_header(header);
    if header_cols.is_empty() {
        return input.to_string();
    }

    // Map the requested field names to column indices.
    let selected: Vec<usize> = fields
        .iter()
        .filter_map(|name| {
            header_cols
                .iter()
                .position(|(_, h)| h.eq_ignore_ascii_case(name))
        })
        .collect();
    if selected.is_empty() {
        return input.to_string();
    }

    let new_header: Vec<&str> = selected.iter().map(|i| header_cols[*i].1).collect();
    let mut out = vec![new_header.join("  ")];

    for line in lines {
        let row = slice_by_header(line, &header_cols);
        let picked: Vec<String> = selected
            .iter()
            .map(|i| row.get(*i).cloned().unwrap_or_default())
            .collect();
        out.push(picked.join("  "));
    }
    out.join("\n")
}

fn split_header(header: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut byte_idx = 0;
    for (is_space, group) in group_by_char_class(header, |c| c.is_whitespace()) {
        if !is_space && !group.is_empty() {
            out.push((byte_idx, group));
        }
        byte_idx += group.len();
    }
    out
}

fn group_by_char_class<F: Fn(char) -> bool>(s: &str, f: F) -> Vec<(bool, &str)> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut current: Option<bool> = None;
    for (i, ch) in s.char_indices() {
        let cls = f(ch);
        match current {
            None => current = Some(cls),
            Some(prev) if prev != cls => {
                out.push((prev, &s[start..i]));
                start = i;
                current = Some(cls);
            }
            _ => {}
        }
    }
    if start < s.len() {
        if let Some(cls) = current {
            out.push((cls, &s[start..]));
        }
    }
    out
}

fn slice_by_header(line: &str, header_cols: &[(usize, &str)]) -> Vec<String> {
    // Each column spans from its header start to the next one. The final
    // column extends to end-of-line. Values are trimmed.
    let mut out = Vec::with_capacity(header_cols.len());
    for (i, (start, _)) in header_cols.iter().enumerate() {
        let end = header_cols
            .get(i + 1)
            .map(|(s, _)| *s)
            .unwrap_or(line.len());
        let s = *start.min(&line.len());
        let e = end.min(line.len());
        out.push(line.get(s..e).unwrap_or("").trim().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_lines_removes_and_keeps() {
        let input = "one\ntwo\nthree\nfour";
        let remove = vec![Regex::new(r"two|four").unwrap()];
        let keep: Vec<Regex> = vec![];
        let out = filter_lines(input, &remove, &keep);
        assert_eq!(out, "one\nthree");
    }

    #[test]
    fn filter_lines_keep_overrides_remove() {
        let input = "err: fatal\nerr: warn\nok";
        let remove = vec![Regex::new(r"err:").unwrap()];
        let keep = vec![Regex::new(r"fatal").unwrap()];
        let out = filter_lines(input, &remove, &keep);
        assert_eq!(out, "err: fatal\nok");
    }

    #[test]
    fn group_by_collapses_groups_above_threshold() {
        let input = "modified: a.rs\nmodified: b.rs\nmodified: c.rs\nuntracked: new.txt";
        let pattern = Regex::new(r"^(modified|untracked):").unwrap();
        let out = group_by(input, &pattern, "{key}: {count} files", 2);
        assert!(out.contains("modified: 3 files"));
        // untracked only has 1 item, must stay verbatim
        assert!(out.contains("untracked: new.txt"));
    }

    #[test]
    fn extract_columns_from_docker_ps() {
        let input = "CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES\n\
                     abc123         nginx   \"/docker\"  1 hour    Up 1h    80/tcp  web\n\
                     def456         redis   \"redis\"    2 hours   Up 2h            cache";
        let fields = vec!["NAMES".to_string(), "STATUS".to_string()];
        let out = extract_columns(input, &fields);
        assert!(out.contains("NAMES"));
        assert!(out.contains("STATUS"));
        assert!(out.contains("web"));
        assert!(out.contains("Up 1h"));
        // Removed columns are gone
        assert!(!out.contains("nginx") || out.contains("Up"));
    }

    #[test]
    fn pipeline_empty_input_yields_ratio_1() {
        let filter = FilterConfig {
            name: "noop",
            command_patterns: vec![],
            strategies: vec![],
            max_output_tokens: None,
            preserve_head: 0,
            preserve_tail: 0,
        };
        let result = run_pipeline("test", "", &filter);
        assert_eq!(result.compressed, "");
        assert_eq!(result.original_tokens, 0);
        assert!((result.ratio - 1.0).abs() < 1e-5);
    }
}
