//! curl / httpie filters — truncate large JSON responses, preserve
//! HTTP status lines.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `curl`, `httpie`, `http ` — if the output looks like JSON, truncate
/// to 50 lines. Always keep HTTP status lines.
pub fn curl_json_filter() -> FilterConfig {
    FilterConfig {
        name: "curl_json",
        command_patterns: vec![rx(r"^\s*curl\b"), rx(r"^\s*httpie\b"), rx(r"^\s*http\s")],
        strategies: vec![
            // Drop blank lines
            CompressionStrategy::FilterLines {
                remove_patterns: vec![rx(r"^\s*$")],
                keep_patterns: vec![rx(r"^HTTP/"), rx(r"^\s*\{"), rx(r"^\s*\["), rx(r"^[<>*]")],
            },
            // Truncate long JSON bodies to 50 lines
            CompressionStrategy::Truncate {
                max_lines: 50,
                omission_message: "... ({n} lines omitted) ...".to_string(),
            },
        ],
        max_output_tokens: Some(500),
        preserve_head: 10,
        preserve_tail: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn curl_json_truncates_large_response() {
        let mut lines = Vec::new();
        lines.push("HTTP/1.1 200 OK".to_string());
        lines.push("{".to_string());
        // Generate a large JSON-like body (80 lines)
        for i in 0..80 {
            lines.push(format!("  \"field_{}\": \"value_{}\",", i, i));
        }
        lines.push("}".to_string());

        let input = lines.join("\n");
        let f = curl_json_filter();
        let r = run_pipeline("curl https://api.example.com/data", &input, &f);
        // Should keep HTTP status line
        assert!(
            r.compressed.contains("HTTP/1.1 200 OK"),
            "should keep HTTP status"
        );
        // Should be truncated (original is ~83 lines, should be ~50)
        let output_lines: Vec<&str> = r.compressed.lines().collect();
        assert!(
            output_lines.len() <= 55,
            "should truncate to ~50 lines, got {}",
            output_lines.len()
        );
        // Should contain omission message
        assert!(
            r.compressed.contains("lines omitted"),
            "should contain omission message: {}",
            r.compressed
        );
    }
}
