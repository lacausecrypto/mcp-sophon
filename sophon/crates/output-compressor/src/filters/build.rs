//! Build / compile filters — `cargo build`, `cargo clippy`, `cargo check`,
//! `tsc`, `eslint`, `biome`, `make`, `ninja`, `bazel`, `go build`, and the
//! JS-runner `… run build` scripts.
//!
//! These were falling through to the generic filter (whose safety floor
//! caps it at the truncation baseline), so the high-volume noise a build
//! emits — `Compiling crate vX` once per dependency, `Downloading`,
//! `Updating`, progress redraws — was never dropped. Strategy: remove that
//! progress noise, keep every diagnostic (errors, warnings, and the
//! `-->`/`|`/`note:`/`help:` context cargo prints under them) plus the
//! final `Finished` line so a clean build still shows it succeeded.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

pub fn build_filter() -> FilterConfig {
    FilterConfig {
        name: "build",
        command_patterns: vec![
            rx(r"^\s*cargo\s+(build|clippy|check|c\b|b\b)"),
            rx(r"^\s*(tsc|eslint|biome|ninja|bazel)\b"),
            rx(r"^\s*make\b"),
            rx(r"^\s*go\s+build\b"),
            rx(r"^\s*(gradle|gradlew|mvn)\b"),
            // JS build scripts: `npm run build`, `pnpm build`, `yarn build`.
            rx(r"^\s*(npm|pnpm|yarn|bun)\s+(run\s+)?build\b"),
        ],
        strategies: vec![
            CompressionStrategy::FoldStackFrames {
                max_frames: 8,
                head: 4,
                tail: 2,
            },
            CompressionStrategy::FilterLines {
                // Progress / bookkeeping noise. A line matching none of these
                // (and none of the keeps) is kept by default — so diagnostics
                // and the final `Finished` line survive untouched.
                remove_patterns: vec![
                    rx(r"^\s*Compiling\s"),
                    rx(r"^\s*Checking\s"),
                    rx(r"^\s*Downloading\s"),
                    rx(r"^\s*Downloaded\s"),
                    rx(r"^\s*Updating\s"),
                    rx(r"^\s*Adding\s"),
                    rx(r"^\s*Building\s\["), // cargo progress bar header
                    rx(r"^\s*Fresh\s"),
                    rx(r"^\s*Installing\s"),
                    rx(r"^\s*Packaging\s"),
                    rx(r"^\s*Documenting\s"),
                    rx(r"^\s*$"),
                ],
                keep_patterns: vec![
                    rx(r"(?i)error"),
                    rx(r"(?i)warning"),
                    rx(r"(?i)\bfailed\b"),
                    rx(r"-->"),    // cargo diagnostic location
                    rx(r"^\s*\|"), // cargo diagnostic source gutter
                    rx(r"^\s*=\s*note"),
                    rx(r"^\s*note:"),
                    rx(r"^\s*help:"),
                    rx(r"^\s*Finished\s"), // success signal
                    rx(r"cannot find"),
                    rx(r"^\s*\^+"), // underline carets under an error span
                ],
            },
        ],
        max_output_tokens: Some(1200),
        preserve_head: 0,
        preserve_tail: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn clean_cargo_build_collapses_to_finished() {
        let filter = build_filter();
        let out = "   Compiling serde v1.0.0\n   Compiling tokio v1.0.0\n   \
                   Compiling myapp v0.1.0\n    Finished dev [unoptimized] target(s) in 4.2s\n";
        let r = run_pipeline("cargo build", out, &filter);
        assert!(
            r.compressed.contains("Finished"),
            "clean build must keep the success line: {:?}",
            r.compressed
        );
        assert!(
            !r.compressed.contains("Compiling serde"),
            "Compiling spam must be dropped: {:?}",
            r.compressed
        );
        assert!(r.compressed_tokens < r.original_tokens);
    }

    #[test]
    fn failing_build_keeps_the_error_and_location() {
        let filter = build_filter();
        let out = "   Compiling myapp v0.1.0\n\
                   error[E0308]: mismatched types\n  \
                   --> src/lib.rs:10:5\n   \
                   |\n10 |     let x: u32 = \"s\";\n   \
                   |                  ^^^ expected `u32`\n\
                   error: could not compile `myapp`\n";
        let r = run_pipeline("cargo build", out, &filter);
        assert!(r.compressed.contains("error[E0308]"));
        assert!(r.compressed.contains("src/lib.rs:10:5"));
        assert!(!r.compressed.contains("Compiling myapp"));
    }
}
