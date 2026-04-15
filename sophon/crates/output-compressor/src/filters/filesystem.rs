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
    fn grep_groups_by_file() {
        let input = r#"src/main.rs:10:fn main()
src/main.rs:20:    println!()
src/main.rs:30:    exit(0)
src/main.rs:40:}
src/main.rs:50:// end
src/lib.rs:5:pub fn api()
src/lib.rs:10:}"#;
        let f = grep_filter();
        let r = run_pipeline("grep -rn foo", input, &f);
        // src/main.rs has 5 matches ≥ min_count=5 → grouped
        assert!(
            r.compressed.contains("src/main.rs: 5 matches"),
            "expected grouping: {}",
            r.compressed
        );
        // src/lib.rs has 2 matches < min_count → kept verbatim
        assert!(r.compressed.contains("src/lib.rs:5:"));
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
