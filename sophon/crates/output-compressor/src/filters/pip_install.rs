//! pip / pip3 install filters — drop download & install progress,
//! keep errors, warnings, and the "Successfully installed" summary.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `pip install` / `pip3 install`.
pub fn pip_install_filter() -> FilterConfig {
    FilterConfig {
        name: "pip_install",
        command_patterns: vec![rx(r"^\s*pip3?\s+install\b")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![
                    rx(r"^\s*Downloading\b"),
                    rx(r"^\s*Installing collected packages"),
                    rx(r"^\s*Using cached\b"),
                    rx(r"━"),
                    rx(r"^\s*$"),
                    rx(r"^\s*Collecting\b"),
                    rx(r"^\s*Using legacy"),
                    rx(r"^\s*Preparing metadata"),
                    rx(r"^\s*Building wheel"),
                ],
                keep_patterns: vec![
                    rx(r"ERROR"),
                    rx(r"WARNING"),
                    rx(r"Successfully installed"),
                    rx(r"Requirement already satisfied"),
                    rx(r"Could not"),
                ],
            },
            CompressionStrategy::Deduplicate {
                similarity_threshold: 0.85,
                output_format: "{line} (x{count})".to_string(),
            },
        ],
        max_output_tokens: Some(300),
        preserve_head: 0,
        preserve_tail: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn pip_install_drops_download_keeps_errors_and_summary() {
        let input = "\
Collecting requests>=2.28
  Downloading requests-2.31.0-py3-none-any.whl (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 2.1 MB/s eta 0:00:00
Collecting urllib3>=1.21.1
  Downloading urllib3-2.0.4-py3-none-any.whl (123 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 123.0/123.0 kB 4.5 MB/s eta 0:00:00
Using cached certifi-2023.7.22-py3-none-any.whl
Installing collected packages: urllib3, certifi, requests
WARNING: pip is configured with locations that require TLS/SSL
ERROR: Cannot install package 'broken'; missing dependency
Successfully installed requests-2.31.0 urllib3-2.0.4 certifi-2023.7.22";

        let f = pip_install_filter();
        let r = run_pipeline("pip install requests", input, &f);
        // Keeps important lines
        assert!(
            r.compressed.contains("WARNING"),
            "should keep WARNING lines"
        );
        assert!(r.compressed.contains("ERROR"), "should keep ERROR lines");
        assert!(
            r.compressed.contains("Successfully installed"),
            "should keep success summary"
        );
        // Drops download noise
        assert!(
            !r.compressed.contains("Downloading"),
            "should drop Downloading"
        );
        assert!(!r.compressed.contains("━"), "should drop progress bars");
        assert!(
            !r.compressed.contains("Using cached"),
            "should drop Using cached"
        );
    }
}
