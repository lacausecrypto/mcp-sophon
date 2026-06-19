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
            // Try structural JSON compression first — if the input
            // parses as JSON, this is a much higher-leverage win
            // than per-line dedup. Non-JSON input is short-circuit
            // rejected (single-byte sniff) so this is essentially
            // free for plain text.
            CompressionStrategy::JsonStructural {
                keep_first_items: 5,
                max_string_chars: 240,
            },
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
        // The two non-repeated lines are deliberately dissimilar so fuzzy
        // dedup won't merge them — that keeps this a clean test of
        // identical-line dedup + blank stripping, with no distinct-line
        // loss that would (correctly) trip the safety floor.
        let input = "alpha header summary\n\nzeta footer total\n\n\nrepeated\nrepeated\nrepeated";
        let f = generic_filter();
        let r = run_pipeline("unknown_cmd", input, &f);
        assert!(r.compressed.contains("(x3)"));
        // Blank lines gone
        assert!(!r.compressed.contains("\n\n\n"));
        // The dedup win is real (identical lines) and no distinct line was
        // lost, so the safety floor must NOT engage and revert it.
        assert!(!r.strategies_applied.iter().any(|s| s == "safety_floor"));
    }

    // F4 safety floor: when generic heuristics (fuzzy dedup here) collapse
    // *distinct* lines and would drop content that a plain truncation
    // keeps, the floor reverts to truncation so we never do worse.
    #[test]
    fn generic_safety_floor_never_worse_than_truncation() {
        // Many distinct-but-similar lines: fuzzy dedup is tempted to merge
        // them, destroying the unique payload on each line.
        let input = (0..200)
            .map(|i| format!("distinct payload row number {i} value={i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let f = generic_filter();
        let r = run_pipeline("some_unknown_tool --list", &input, &f);

        let baseline = crate::truncate::head_truncate_tokens(&input, r.compressed_tokens.max(1));
        let comp_cov = crate::truncate::unique_line_coverage(&input, &r.compressed);
        let base_cov = crate::truncate::unique_line_coverage(&input, &baseline);
        // The guarantee: compressed preserves at least as many distinct
        // original lines as truncation to the same budget.
        assert!(
            comp_cov >= base_cov,
            "compress_output ({comp_cov}) must not preserve fewer lines than truncation ({base_cov})"
        );
    }

    #[test]
    fn generic_safety_floor_exempts_json() {
        // JSON structural compression reformats the text; the floor must
        // not clobber that genuine structural win.
        let items = (0..50)
            .map(|i| format!("{{\"id\": {i}, \"name\": \"item-{i}\"}}"))
            .collect::<Vec<_>>()
            .join(",\n");
        let input = format!("[{items}]");
        let f = generic_filter();
        let r = run_pipeline("curl https://api.example.com/items", &input, &f);
        assert!(r.strategies_applied.iter().any(|s| s == "json_structural"));
        assert!(!r.strategies_applied.iter().any(|s| s == "safety_floor"));
    }

    #[test]
    fn generic_shrinks_very_long_input() {
        // Use unique content per line so normalize() doesn't collapse
        // them all into a single dedup bucket. We want to exercise the
        // truncate strategy.
        let words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike", "november",
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
        assert!(
            r.compressed.lines().count() < 300,
            "{} lines",
            r.compressed.lines().count()
        );
        assert!(r.ratio < 0.5, "ratio = {}", r.ratio);
    }
}
