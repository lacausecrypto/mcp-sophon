//! Git filters.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(pat: &str) -> Regex {
    Regex::new(pat).expect("valid regex")
}

/// `git status` — drop the instruction preamble, keep tracked changes.
pub fn git_status_filter() -> FilterConfig {
    FilterConfig {
        name: "git_status",
        command_patterns: vec![rx(r"^\s*git\s+status")],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r#"^\s*\(use "git"#),
                rx(r"^On branch"),
                rx(r"^Your branch is"),
                rx(r"^Changes not staged"),
                rx(r"^Changes to be committed"),
                rx(r"^Untracked files:"),
                rx(r"^no changes added to commit"),
                rx(r"^nothing to commit"),
                rx(r"^\s*$"),
            ],
            // Narrow keep list: only "modified: / new file: / …" and
            // bare filenames under an untracked block. We deliberately
            // do NOT keep arbitrary alphanumeric-starting lines because
            // those include "no changes added to commit (use …)".
            keep_patterns: vec![rx(
                r"^\s*(modified|deleted|new file|renamed|typechange|both modified):",
            )],
        }],
        max_output_tokens: Some(400),
        preserve_head: 0,
        preserve_tail: 0,
    }
}

/// `git log` — short commits only, truncate long histories.
pub fn git_log_filter() -> FilterConfig {
    FilterConfig {
        name: "git_log",
        command_patterns: vec![rx(r"^\s*git\s+log")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![
                    rx(r"^Author:"),
                    rx(r"^AuthorDate:"),
                    rx(r"^Commit:"),
                    rx(r"^CommitDate:"),
                    rx(r"^Date:"),
                    rx(r"^\s*$"),
                    rx(r"^    (Signed-off-by|Co-authored-by):"),
                ],
                keep_patterns: vec![],
            },
            CompressionStrategy::Truncate {
                max_lines: 40,
                omission_message: "... {n} more commits omitted ...".to_string(),
            },
        ],
        max_output_tokens: Some(600),
        preserve_head: 10,
        preserve_tail: 0,
    }
}

/// `git push` / `git pull` — just the outcome line.
pub fn git_push_pull_filter() -> FilterConfig {
    FilterConfig {
        name: "git_push_pull",
        command_patterns: vec![rx(r"^\s*git\s+(push|pull|fetch)\b")],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r"^(Enumerating|Counting|Compressing|Writing|Delta|Total)"),
                rx(r"^remote:"),
                rx(r"^Unpacking"),
                rx(r"^Resolving"),
                rx(r"^\s*$"),
            ],
            keep_patterns: vec![
                rx(r"^\s*[a-f0-9]+\.\.[a-f0-9]+"),
                rx(r"->"),
                rx(r"Fast-forward"),
                rx(r"Already up to date"),
                rx(r"error:"),
                rx(r"fatal:"),
                rx(r"Everything up-to-date"),
                rx(r"To "),
            ],
        }],
        max_output_tokens: Some(100),
        preserve_head: 0,
        preserve_tail: 3,
    }
}

/// `git diff` — keep diff headers + +/- lines, dedupe repeated hunks,
/// then truncate long diffs.
///
/// Repeated boilerplate hunks (same `-old` / `+new` pair applied to
/// every match of a refactor) would otherwise pass through verbatim.
/// `Deduplicate` collapses runs of identical consecutive lines to
/// `<line> (repeated N times)`, which crushes bulk renames /
/// mechanical rewrites without losing the information that the
/// change was uniform.
pub fn git_diff_filter() -> FilterConfig {
    FilterConfig {
        name: "git_diff",
        command_patterns: vec![rx(r"^\s*git\s+diff")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![rx(r"^index [0-9a-f]+\.\.[0-9a-f]+")],
                keep_patterns: vec![rx(r"^(diff|---|\+\+\+|@@)"), rx(r"^[+-][^+-]")],
            },
            // Collapse identical consecutive `+` / `-` lines from
            // mechanical refactors. `similarity_threshold=1.0` means
            // exact match only — we don't want to fuzzy-merge
            // semantically distinct hunks.
            CompressionStrategy::Deduplicate {
                similarity_threshold: 1.0,
                output_format: "{line}  // (× {count})".to_string(),
            },
            CompressionStrategy::Truncate {
                max_lines: 120,
                omission_message: "... diff truncated, {n} lines omitted ...".to_string(),
            },
        ],
        max_output_tokens: Some(1200),
        preserve_head: 10,
        preserve_tail: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn git_status_drops_instructions() {
        let input = r#"On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   src/main.rs
        modified:   src/lib.rs

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        new_file.txt

no changes added to commit (use "git add" and/or "git commit -a")"#;

        let f = git_status_filter();
        let r = run_pipeline("git status", input, &f);
        assert!(
            r.compressed.contains("modified:"),
            "lost modified: {}",
            r.compressed
        );
        assert!(r.compressed.contains("src/main.rs"));
        assert!(!r.compressed.contains("use \"git add"));
        assert!(!r.compressed.contains("On branch main"));
        assert!(r.ratio < 0.7, "too loose: {}", r.ratio);
    }

    #[test]
    fn git_push_keeps_only_outcome() {
        let input = r#"Enumerating objects: 42, done.
Counting objects: 100% (42/42), done.
Delta compression using up to 8 threads
Compressing objects: 100% (20/20), done.
Writing objects: 100% (25/25), 3.42 KiB | 1.71 MiB/s, done.
Total 25 (delta 15), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (15/15), done.
To github.com:user/repo.git
   abc1234..def5678  main -> main"#;

        let f = git_push_pull_filter();
        let r = run_pipeline("git push", input, &f);
        assert!(r.compressed.contains("abc1234..def5678"));
        assert!(r.compressed.contains("main -> main"));
        assert!(!r.compressed.contains("Enumerating"));
        assert!(r.ratio < 0.35, "git push should crush: {}", r.ratio);
    }

    #[test]
    fn git_diff_keeps_hunks() {
        let input = r#"diff --git a/foo.rs b/foo.rs
index abcd1234..efgh5678 100644
--- a/foo.rs
+++ b/foo.rs
@@ -1,3 +1,3 @@
-old line
+new line
 unchanged"#;

        let f = git_diff_filter();
        let r = run_pipeline("git diff", input, &f);
        assert!(r.compressed.contains("@@"));
        assert!(r.compressed.contains("+new line"));
        assert!(r.compressed.contains("-old line"));
        assert!(!r.compressed.contains("index abcd1234"));
    }
}
