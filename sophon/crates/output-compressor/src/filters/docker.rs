//! Docker filters — container listing + log deduplication.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `docker ps` / `docker container ls`: keep only the operationally
/// useful columns (NAMES, STATUS, PORTS).
pub fn docker_ps_filter() -> FilterConfig {
    FilterConfig {
        name: "docker_ps",
        command_patterns: vec![
            rx(r"^\s*docker\s+(ps|container\s+ls)"),
            rx(r"^\s*podman\s+(ps|container\s+ls)"),
        ],
        strategies: vec![CompressionStrategy::ExtractColumns {
            fields: vec![
                "NAMES".to_string(),
                "STATUS".to_string(),
                "PORTS".to_string(),
            ],
        }],
        max_output_tokens: Some(300),
        preserve_head: 1,
        preserve_tail: 0,
    }
}

/// `docker logs`: collapse repeated lines with counts, truncate.
pub fn docker_logs_filter() -> FilterConfig {
    FilterConfig {
        name: "docker_logs",
        command_patterns: vec![
            rx(r"^\s*docker\s+logs"),
            rx(r"^\s*podman\s+logs"),
            rx(r"^\s*kubectl\s+logs"),
        ],
        strategies: vec![
            CompressionStrategy::Deduplicate {
                similarity_threshold: 0.9,
                output_format: "{line} (x{count})".to_string(),
            },
            CompressionStrategy::Truncate {
                max_lines: 80,
                omission_message: "... {n} log lines omitted ...".to_string(),
            },
        ],
        max_output_tokens: Some(800),
        preserve_head: 5,
        preserve_tail: 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn docker_ps_keeps_only_selected_columns() {
        let input = "CONTAINER ID   IMAGE          COMMAND    CREATED        STATUS        PORTS                NAMES\n\
                     abc123         nginx:latest   nginx      2 hours ago    Up 2 hours    0.0.0.0:80->80/tcp   web_server\n\
                     def456         redis:7        redis      3 hours ago    Up 3 hours                         cache_store";
        let f = docker_ps_filter();
        let r = run_pipeline("docker ps", input, &f);
        assert!(r.compressed.contains("NAMES"));
        assert!(r.compressed.contains("STATUS"));
        assert!(r.compressed.contains("web_server"));
        assert!(r.compressed.contains("Up 2 hours"));
        // CONTAINER ID / IMAGE / COMMAND dropped
        assert!(!r.compressed.contains("CONTAINER ID"));
        assert!(!r.compressed.contains("nginx:latest"));
    }

    #[test]
    fn docker_logs_deduplicates_repeats() {
        let input = "2026-04-15T10:00:00 INFO: processing item 1\n\
                     2026-04-15T10:00:01 INFO: processing item 2\n\
                     2026-04-15T10:00:02 INFO: processing item 3\n\
                     2026-04-15T10:00:03 INFO: processing item 4\n\
                     2026-04-15T10:00:04 ERROR: connection lost\n\
                     2026-04-15T10:00:05 INFO: processing item 5";
        let f = docker_logs_filter();
        let r = run_pipeline("docker logs svc", input, &f);
        assert!(r.compressed.contains("(x"));
        assert!(r.compressed.contains("ERROR"));
    }
}
