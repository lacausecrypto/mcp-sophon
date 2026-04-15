//! Line-level deduplication.
//!
//! A line is "similar" to another if their normalized forms
//! (timestamps/numbers stripped) are identical, OR their Jaccard
//! similarity over whitespace-split tokens is >= `threshold`. This
//! catches both exact repeats ("Processing item 1 / 2 / 3 …") and near
//! duplicates ("ERROR: connection failed (5 times)").
//!
//! For determinism we use a linear pass, not a k-NN across the whole
//! file: a line is collapsed into the most recent canonical if they
//! match, otherwise it starts a new canonical. This is O(n·k) where k
//! is the number of unique canonicals — acceptable for command outputs
//! that rarely have > 10k distinct lines.

use once_cell::sync::Lazy;
use regex::Regex;

static NUMERIC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+").expect("valid numeric regex"));
static TS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
        .expect("valid ts regex")
});

/// Normalize a line for similarity comparison: strip ISO timestamps
/// and numeric runs so `"foo 42"` and `"foo 7"` collapse.
pub fn normalize(line: &str) -> String {
    let stage1 = TS_RE.replace_all(line, "<TS>");
    let stage2 = NUMERIC_RE.replace_all(&stage1, "<N>");
    stage2.trim().to_string()
}

/// Jaccard similarity over whitespace tokens.
fn jaccard(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;
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

/// Deduplicate `input` line-by-line. Lines that collapse into the same
/// canonical get a `(xN)` suffix (or whatever `output_format` prescribes).
///
/// `output_format` supports `{line}` and `{count}` placeholders.
pub fn dedup_lines(input: &str, threshold: f32, output_format: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    #[derive(Debug)]
    struct Bucket {
        canonical: String,
        normalized: String,
        count: usize,
    }

    let mut buckets: Vec<Bucket> = Vec::new();

    for line in input.lines() {
        let normalized = normalize(line);
        // Look for an existing bucket that matches.
        let mut matched = false;
        for b in buckets.iter_mut().rev() {
            if b.normalized == normalized || jaccard(&b.normalized, &normalized) >= threshold {
                b.count += 1;
                matched = true;
                break;
            }
        }
        if !matched {
            buckets.push(Bucket {
                canonical: line.to_string(),
                normalized,
                count: 1,
            });
        }
    }

    buckets
        .into_iter()
        .map(|b| {
            if b.count == 1 {
                b.canonical
            } else {
                output_format
                    .replace("{line}", &b.canonical)
                    .replace("{count}", &b.count.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_numbers() {
        assert_eq!(normalize("processing item 42"), "processing item <N>");
        assert_eq!(normalize("processing item 7"), "processing item <N>");
    }

    #[test]
    fn normalize_strips_timestamps() {
        let n = normalize("2026-04-15T10:00:00Z INFO: hello");
        assert!(n.starts_with("<TS>"));
    }

    #[test]
    fn dedup_collapses_near_duplicates() {
        let input = "Processing item 1\nProcessing item 2\nProcessing item 3\nERROR: boom\nProcessing item 4";
        let out = dedup_lines(input, 0.9, "{line} (x{count})");
        assert!(out.contains("(x4)"), "expected collapsed line: {}", out);
        assert!(out.contains("ERROR: boom"));
    }

    #[test]
    fn dedup_keeps_singletons_verbatim() {
        let input = "only one\nanother unique";
        let out = dedup_lines(input, 0.9, "{line} (x{count})");
        assert_eq!(out, "only one\nanother unique");
    }

    #[test]
    fn dedup_empty_input() {
        assert_eq!(dedup_lines("", 0.9, "{line}"), "");
    }
}
