//! Command rewriter.
//!
//! The rewriter turns a raw shell command into one of two outcomes:
//!
//! - `Passthrough(cmd)` — the caller should run `cmd` verbatim. Used for
//!   commands that already start with `sophon `, that the user has
//!   excluded, or that don't match any rule.
//!
//! - `Rewritten { original, rewritten }` — the caller should run
//!   `rewritten` instead. `rewritten` is always of the form
//!   `sophon exec -- <original>` so the downstream `sophon exec`
//!   executes the real command and pipes its output through the
//!   `output-compressor` filters before returning to the LLM.
//!
//! The rule set is a flat `Vec<RewriteRule>` matched top-to-bottom.
//! All rules are deterministic regexes — no ML, no heuristics.

use once_cell::sync::Lazy;
use regex::Regex;

/// A single rewrite rule.
#[derive(Debug, Clone)]
pub struct RewriteRule {
    /// Human-readable name (appears in rewriter diagnostics).
    pub name: &'static str,
    /// Regex matched against the full command string.
    pub pattern: Regex,
}

impl RewriteRule {
    fn new(name: &'static str, pattern: &str) -> Self {
        Self {
            name,
            pattern: Regex::new(pattern).expect("valid regex in RewriteRule"),
        }
    }
}

/// Outcome of a rewrite attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteResult {
    /// Run `cmd` unchanged.
    Passthrough(String),
    /// Run `rewritten` (which wraps `original` in `sophon exec -- …`).
    Rewritten { original: String, rewritten: String, rule: &'static str },
}

impl RewriteResult {
    pub fn final_command(&self) -> &str {
        match self {
            Self::Passthrough(c) => c,
            Self::Rewritten { rewritten, .. } => rewritten,
        }
    }

    pub fn is_rewritten(&self) -> bool {
        matches!(self, Self::Rewritten { .. })
    }
}

/// Default rule set — 20 regex patterns covering the commands that
/// benefit most from Sophon's output compression.
static DEFAULT_RULES: Lazy<Vec<RewriteRule>> = Lazy::new(|| {
    vec![
        // Git family ---------------------------------------------------------
        RewriteRule::new("git_status", r"^\s*git\s+status\b"),
        RewriteRule::new("git_diff", r"^\s*git\s+diff\b"),
        RewriteRule::new("git_log", r"^\s*git\s+log\b"),
        RewriteRule::new("git_show", r"^\s*git\s+show\b"),
        RewriteRule::new("git_stash_list", r"^\s*git\s+stash\s+list\b"),
        RewriteRule::new("git_branch", r"^\s*git\s+branch\b"),
        RewriteRule::new("git_push", r"^\s*git\s+push\b"),
        RewriteRule::new("git_pull", r"^\s*git\s+pull\b"),
        RewriteRule::new("git_fetch", r"^\s*git\s+fetch\b"),
        // Test runners -------------------------------------------------------
        RewriteRule::new("cargo_test", r"^\s*cargo\s+test\b"),
        RewriteRule::new("pytest", r"^\s*(pytest|python\s+-m\s+pytest)\b"),
        RewriteRule::new("vitest_jest", r"^\s*(vitest|jest|(npm|pnpm|yarn)\s+test)\b"),
        RewriteRule::new("go_test", r"^\s*go\s+test\b"),
        // Build --------------------------------------------------------------
        RewriteRule::new("cargo_build", r"^\s*cargo\s+(build|clippy|check)\b"),
        // Filesystem ---------------------------------------------------------
        RewriteRule::new("ls", r"^\s*ls(\s|$)"),
        RewriteRule::new("tree", r"^\s*tree(\s|$)"),
        RewriteRule::new("find", r"^\s*find\s"),
        RewriteRule::new("grep", r"^\s*(grep|rg|ripgrep|ag|ack)\b"),
        // Docker / Kubernetes -----------------------------------------------
        RewriteRule::new("docker_ps", r"^\s*(docker|podman)\s+(ps|container\s+ls)\b"),
        RewriteRule::new("docker_logs", r"^\s*(docker|podman|kubectl)\s+logs\b"),
    ]
});

/// Pure rewriter. Holds a list of rules and an exclusion prefix list.
#[derive(Debug, Clone)]
pub struct CommandRewriter {
    rules: Vec<RewriteRule>,
    exclusions: Vec<String>,
}

impl Default for CommandRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandRewriter {
    /// Build a rewriter using the [`DEFAULT_RULES`] set.
    pub fn new() -> Self {
        Self {
            rules: DEFAULT_RULES.clone(),
            exclusions: Vec::new(),
        }
    }

    /// Build a rewriter with an explicit rule list. Used in tests and
    /// when loading a user-provided TOML config.
    pub fn with_rules(rules: Vec<RewriteRule>) -> Self {
        Self { rules, exclusions: Vec::new() }
    }

    /// Add a prefix that short-circuits to passthrough (e.g. `curl`).
    pub fn exclude<S: Into<String>>(mut self, prefix: S) -> Self {
        self.exclusions.push(prefix.into());
        self
    }

    pub fn rules(&self) -> &[RewriteRule] {
        &self.rules
    }

    /// Attempt to rewrite `command`. Never mutates state; deterministic.
    pub fn rewrite(&self, command: &str) -> RewriteResult {
        let trimmed = command.trim();

        // Already a sophon command → never wrap twice.
        if trimmed.starts_with("sophon ") || trimmed == "sophon" {
            return RewriteResult::Passthrough(command.to_string());
        }

        // User-excluded prefix.
        if self
            .exclusions
            .iter()
            .any(|ex| trimmed.starts_with(ex.as_str()))
        {
            return RewriteResult::Passthrough(command.to_string());
        }

        // Heredoc / multi-line bash block: leave it alone. Wrapping these
        // through `sophon exec --` would be fragile because the `<<` body
        // may contain quoting that breaks argv splitting.
        if trimmed.contains("<<") {
            return RewriteResult::Passthrough(command.to_string());
        }

        // Pipelines: rewriting the first stage is risky (the tail of the
        // pipe might depend on the raw shape). For safety, passthrough.
        if trimmed.contains('|') || trimmed.contains("&&") || trimmed.contains("||") {
            return RewriteResult::Passthrough(command.to_string());
        }

        // Match the first rule whose pattern hits.
        for rule in &self.rules {
            if rule.pattern.is_match(trimmed) {
                let rewritten = format!("sophon exec -- {}", trimmed);
                return RewriteResult::Rewritten {
                    original: command.to_string(),
                    rewritten,
                    rule: rule.name,
                };
            }
        }

        RewriteResult::Passthrough(command.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rule_count() {
        let r = CommandRewriter::new();
        // 9 git + 4 tests + 1 build + 4 fs + 2 docker = 20
        assert_eq!(r.rules().len(), 20);
    }

    #[test]
    fn rewrites_git_status() {
        let r = CommandRewriter::new();
        let out = r.rewrite("git status");
        assert!(out.is_rewritten());
        assert_eq!(out.final_command(), "sophon exec -- git status");
    }

    #[test]
    fn rewrites_git_diff_with_args() {
        let r = CommandRewriter::new();
        let out = r.rewrite("git diff HEAD~2 src/main.rs");
        assert!(out.is_rewritten());
        assert_eq!(
            out.final_command(),
            "sophon exec -- git diff HEAD~2 src/main.rs"
        );
    }

    #[test]
    fn rewrites_cargo_test() {
        let r = CommandRewriter::new();
        let out = r.rewrite("cargo test --workspace");
        assert!(out.is_rewritten());
        assert_eq!(
            out.final_command(),
            "sophon exec -- cargo test --workspace"
        );
    }

    #[test]
    fn rewrites_pytest_and_vitest() {
        let r = CommandRewriter::new();
        assert!(r.rewrite("pytest tests/").is_rewritten());
        assert!(r.rewrite("python -m pytest -xvs").is_rewritten());
        assert!(r.rewrite("vitest run").is_rewritten());
        assert!(r.rewrite("npm test").is_rewritten());
        assert!(r.rewrite("go test ./...").is_rewritten());
    }

    #[test]
    fn rewrites_filesystem_and_search() {
        let r = CommandRewriter::new();
        assert!(r.rewrite("ls -la").is_rewritten());
        assert!(r.rewrite("tree -L 2").is_rewritten());
        assert!(r.rewrite("find . -name '*.rs'").is_rewritten());
        assert!(r.rewrite("grep -rn TODO").is_rewritten());
        assert!(r.rewrite("rg --files").is_rewritten());
    }

    #[test]
    fn rewrites_docker() {
        let r = CommandRewriter::new();
        assert!(r.rewrite("docker ps").is_rewritten());
        assert!(r.rewrite("docker container ls").is_rewritten());
        assert!(r.rewrite("kubectl logs pod-foo").is_rewritten());
    }

    #[test]
    fn passthrough_for_sophon_commands() {
        let r = CommandRewriter::new();
        let out = r.rewrite("sophon serve");
        assert!(!out.is_rewritten(), "should NOT double-wrap sophon");
    }

    #[test]
    fn passthrough_for_heredoc() {
        let r = CommandRewriter::new();
        let out = r.rewrite("cat <<EOF\nhello\nEOF");
        assert!(!out.is_rewritten());
    }

    #[test]
    fn passthrough_for_pipelines() {
        let r = CommandRewriter::new();
        let out = r.rewrite("git log | head -5");
        assert!(!out.is_rewritten(), "pipelines should be left alone");
    }

    #[test]
    fn passthrough_for_unknown_command() {
        let r = CommandRewriter::new();
        let out = r.rewrite("totally_unknown_tool --args");
        assert!(!out.is_rewritten());
        assert_eq!(out.final_command(), "totally_unknown_tool --args");
    }

    #[test]
    fn exclusion_list_wins() {
        let r = CommandRewriter::new().exclude("git log");
        let out = r.rewrite("git log --oneline");
        assert!(!out.is_rewritten(), "exclusion should short-circuit rewrite");
    }

    #[test]
    fn determinism() {
        let r = CommandRewriter::new();
        let a = r.rewrite("git status");
        let b = r.rewrite("git status");
        assert_eq!(a, b);
    }
}
