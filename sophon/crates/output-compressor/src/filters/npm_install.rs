//! npm / yarn / pnpm install filters — drop progress noise, keep
//! warnings, errors, and the final summary.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `npm install`, `npm i`, `yarn add`, `pnpm add`, `pnpm install`.
pub fn npm_install_filter() -> FilterConfig {
    FilterConfig {
        name: "npm_install",
        command_patterns: vec![
            rx(r"^\s*(npm\s+(install|i)\b)"),
            rx(r"^\s*yarn\s+add\b"),
            rx(r"^\s*pnpm\s+(add|install)\b"),
        ],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![
                    rx(r"^npm http"),
                    rx(r"added \d+ packages?"),
                    rx(r"⸩"),
                    rx(r"idealTree"),
                    rx(r"^\s*$"),
                    rx(r"^npm timing"),
                    rx(r"^npm sill"),
                    rx(r"^npm verb"),
                    rx(r"^\.\.\.$"),
                ],
                keep_patterns: vec![
                    rx(r"npm ERR!"),
                    rx(r"npm WARN"),
                    rx(r"up to date"),
                    rx(r"added \d+ packages? in"),
                    rx(r"^\d+ packages? are looking for funding"),
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
    fn npm_install_drops_progress_keeps_errors_and_summary() {
        let input = "\
npm http fetch GET 200 https://registry.npmjs.org/express 50ms
npm http fetch GET 200 https://registry.npmjs.org/lodash 30ms
npm http fetch GET 200 https://registry.npmjs.org/react 40ms
idealTree: timing idealTree Completed in 500ms
⸩ ░░░░░░░░░░░░░░░░░░ 0/15
⸩ ████░░░░░░░░░░░░░░ 4/15
⸩ ████████░░░░░░░░░░ 8/15
npm WARN deprecated faker@5.5.3: This package is no longer maintained.
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
added 15 packages, and audited 320 packages in 5s
3 packages are looking for funding";

        let f = npm_install_filter();
        let r = run_pipeline("npm install", input, &f);
        // Keeps warnings and errors
        assert!(r.compressed.contains("npm WARN"), "should keep npm WARN");
        assert!(r.compressed.contains("npm ERR!"), "should keep npm ERR!");
        assert!(
            r.compressed.contains("looking for funding"),
            "should keep funding summary"
        );
        // Drops progress noise
        assert!(!r.compressed.contains("npm http"), "should drop http lines");
        assert!(!r.compressed.contains("idealTree"), "should drop idealTree");
        assert!(!r.compressed.contains("⸩"), "should drop progress bars");
    }
}
