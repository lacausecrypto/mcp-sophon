//! Generic fallback — applies to anything the other filters don't
//! catch. Drops empty lines, deduplicates, truncates middle if long.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

pub fn generic_filter() -> FilterConfig {
    FilterConfig {
        name: "generic",
        command_patterns: vec![rx(r".*")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![rx(r"^\s*$")],
                keep_patterns: vec![],
            },
            CompressionStrategy::Deduplicate {
                similarity_threshold: 0.85,
                output_format: "{line} (x{count})".to_string(),
            },
            CompressionStrategy::Truncate {
                max_lines: 120,
                omission_message: "... {n} lines omitted ...".to_string(),
            },
        ],
        max_output_tokens: Some(1500),
        preserve_head: 10,
        preserve_tail: 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn generic_dedups_and_strips_blank_lines() {
        let input = "useful line 1\n\nuseful line 2\n\n\nrepeated\nrepeated\nrepeated";
        let f = generic_filter();
        let r = run_pipeline("unknown_cmd", input, &f);
        assert!(r.compressed.contains("(x3)"));
        // Blank lines gone
        assert!(!r.compressed.contains("\n\n\n"));
    }

    #[test]
    fn generic_shrinks_very_long_input() {
        // Use unique content per line so normalize() doesn't collapse
        // them all into a single dedup bucket. We want to exercise the
        // truncate strategy.
        let words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
            "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
        ];
        let input = (0..1000)
            .map(|i| {
                format!(
                    "{} {} {}",
                    words[i % words.len()],
                    words[(i / words.len()) % words.len()],
                    words[(i / (words.len() * words.len())) % words.len()],
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let f = generic_filter();
        let r = run_pipeline("very_long", &input, &f);
        // Either the truncate strategy fired OR dedup collapsed many
        // lines — both count as "generic did its job". Assert strong
        // compression rather than a specific marker string.
        assert!(r.compressed.lines().count() < 300, "{} lines", r.compressed.lines().count());
        assert!(r.ratio < 0.5, "ratio = {}", r.ratio);
    }
}
