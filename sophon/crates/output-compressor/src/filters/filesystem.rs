//! Filesystem navigation filters — ls, grep/rg, find.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `ls -la` and `tree`: group homogeneous entries by extension, truncate
/// long listings.
pub fn ls_filter() -> FilterConfig {
    FilterConfig {
        name: "ls_tree",
        command_patterns: vec![rx(r"^\s*ls(\s|$)"), rx(r"^\s*tree(\s|$)")],
        strategies: vec![
            CompressionStrategy::GroupBy {
                key_pattern: rx(r"\.([a-zA-Z0-9]{1,10})\s*$"),
                output_format: "*.{key}: {count} files".to_string(),
                min_count: 4,
            },
            CompressionStrategy::Truncate {
                max_lines: 40,
                omission_message: "... {n} more entries ...".to_string(),
            },
        ],
        max_output_tokens: Some(500),
        preserve_head: 5,
        preserve_tail: 0,
    }
}

/// `grep` / `rg`: group matches by file, truncate if many files.
pub fn grep_filter() -> FilterConfig {
    FilterConfig {
        name: "grep",
        command_patterns: vec![rx(r"^\s*(grep|rg|ripgrep|ag|ack)\b")],
        strategies: vec![
            CompressionStrategy::GroupBy {
                key_pattern: rx(r"^([^:]+):\d+"),
                output_format: "{key}: {count} matches".to_string(),
                min_count: 5,
            },
            CompressionStrategy::Truncate {
                max_lines: 80,
                omission_message: "... {n} more matches ...".to_string(),
            },
        ],
        max_output_tokens: Some(800),
        preserve_head: 15,
        preserve_tail: 5,
    }
}

/// `find`: group results by parent directory.
pub fn find_filter() -> FilterConfig {
    FilterConfig {
        name: "find",
        command_patterns: vec![rx(r"^\s*find\s")],
        strategies: vec![
            CompressionStrategy::GroupBy {
                key_pattern: rx(r"^(.+)/[^/]+$"),
                output_format: "{key}/: {count} entries".to_string(),
                min_count: 5,
            },
            CompressionStrategy::Truncate {
                max_lines: 40,
                omission_message: "... {n} more entries ...".to_string(),
            },
        ],
        max_output_tokens: Some(500),
        preserve_head: 10,
        preserve_tail: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn ls_truncates_long_listing() {
        let input = (1..=100)
            .map(|i| format!("-rw-r--r--  1 u  g   42 Jan  1 10:00 file_{}.txt", i))
            .collect::<Vec<_>>()
            .join("\n");
        let f = ls_filter();
        let r = run_pipeline("ls -la", &input, &f);
        // Many .txt files → grouping should kick in
        assert!(
            r.compressed.contains("txt: ") || r.compressed.contains("more entries"),
            "expected grouping or truncation: {}",
            r.compressed
        );
        assert!(r.ratio < 0.6);
    }

    #[test]
    fn grep_never_below_truncation() {
        // grep output is list-like: every `file:line: match` row is signal.
        // The GroupBy heuristic would collapse same-file matches into a
        // bare `file: N matches` count — which destroys the matches the
        // caller grepped for. The F4 safety floor guarantees we never end
        // up below a plain truncation: the verbatim matches survive.
        let input = r#"src/main.rs:10:fn main()
src/main.rs:20:    println!()
src/main.rs:30:    exit(0)
src/main.rs:40:}
src/main.rs:50:// end
src/lib.rs:5:pub fn api()
src/lib.rs:10:}"#;
        let f = grep_filter();
        let r = run_pipeline("grep -rn foo", input, &f);

        let baseline = crate::truncate::head_truncate_tokens(input, r.compressed_tokens.max(1));
        let comp_cov = crate::truncate::unique_line_coverage(input, &r.compressed);
        let base_cov = crate::truncate::unique_line_coverage(input, &baseline);
        // The guarantee holds regardless of whether grouping or the floor
        // produced the output: never fewer matches than truncation. (On a
        // small output like this, grouping ties truncation and is kept; on
        // a large multi-file grep the floor engages — see
        // `grep_safety_floor_preserves_matches`.)
        assert!(
            comp_cov >= base_cov,
            "grep must not preserve fewer matches ({comp_cov}) than truncation ({base_cov})"
        );
    }

    // F4 (bench output-050): grep over many matches in one file used to
    // collapse them to `file: N matches`, destroying the verbatim match
    // text that the caller actually grepped for — 0% recall where plain
    // truncation got 100%. The safety floor must keep us ≥ truncation.
    #[test]
    fn grep_safety_floor_preserves_matches() {
        // Many `pub fn` matches across several files — over min_count, so
        // GroupBy is tempted to collapse them into useless count lines.
        let input = (0..8)
            .flat_map(|file| {
                (0..6).map(move |i| {
                    format!(
                        "src/mod_{file}.rs:{}:pub fn handler_{file}_{i}(req: Req) -> Resp",
                        i * 3
                    )
                })
            })
            .collect::<Vec<_>>()
            .join("\n");
        let f = grep_filter();
        let r = run_pipeline("grep -rn 'pub fn' src", &input, &f);

        let baseline = crate::truncate::head_truncate_tokens(&input, r.compressed_tokens.max(1));
        let comp_cov = crate::truncate::unique_line_coverage(&input, &r.compressed);
        let base_cov = crate::truncate::unique_line_coverage(&input, &baseline);
        assert!(
            comp_cov >= base_cov,
            "grep must not preserve fewer matches ({comp_cov}) than truncation ({base_cov}); \
             got: {}",
            r.compressed
        );
        // And concretely: a real signature survives verbatim, not just a
        // `N matches` summary.
        assert!(
            r.compressed.contains("pub fn handler_0_0("),
            "verbatim match lost: {}",
            r.compressed
        );
        assert!(!r.compressed.contains(" matches"));
    }

    #[test]
    fn find_groups_by_directory() {
        let input = (1..=10)
            .map(|i| format!("./src/tests/test_{}.rs", i))
            .collect::<Vec<_>>()
            .join("\n");
        let f = find_filter();
        let r = run_pipeline("find . -name '*.rs'", &input, &f);
        assert!(
            r.compressed.contains("./src/tests/: 10 entries"),
            "got: {}",
            r.compressed
        );
    }
}
