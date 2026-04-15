//! Command family detection. Classifies a raw command string into one
//! of a few high-level families used for reporting. The actual filter
//! selection lives in `FilterRegistry` and uses regex patterns per
//! filter — this module is just for human-readable tags in the result.

use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandFamily {
    Git,
    TestRunner,
    Build,
    Filesystem,
    Search,
    Docker,
    Cloud,
    Generic,
}

impl CommandFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Git => "git",
            Self::TestRunner => "test",
            Self::Build => "build",
            Self::Filesystem => "filesystem",
            Self::Search => "search",
            Self::Docker => "docker",
            Self::Cloud => "cloud",
            Self::Generic => "generic",
        }
    }
}

static GIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*git(\s|$)").unwrap());
static TEST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\s*(cargo\s+test|pytest|python\s+-m\s+pytest|vitest|jest|go\s+test|npm\s+test|pnpm\s+test)")
        .unwrap()
});
static BUILD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\s*(cargo\s+(build|clippy|check)|tsc|eslint|biome|make\b|ninja|bazel)").unwrap()
});
static FS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*(ls|tree|find|du|df|stat)\b").unwrap());
static SEARCH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*(grep|rg|ripgrep|ag|ack)\b").unwrap());
static DOCKER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*(docker|podman|kubectl|helm)\b").unwrap());
static CLOUD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*(aws|gcloud|az)\b").unwrap());

pub fn detect_command_family(cmd: &str) -> CommandFamily {
    if GIT_RE.is_match(cmd) {
        CommandFamily::Git
    } else if TEST_RE.is_match(cmd) {
        CommandFamily::TestRunner
    } else if BUILD_RE.is_match(cmd) {
        CommandFamily::Build
    } else if SEARCH_RE.is_match(cmd) {
        CommandFamily::Search
    } else if FS_RE.is_match(cmd) {
        CommandFamily::Filesystem
    } else if DOCKER_RE.is_match(cmd) {
        CommandFamily::Docker
    } else if CLOUD_RE.is_match(cmd) {
        CommandFamily::Cloud
    } else {
        CommandFamily::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_git() {
        assert_eq!(detect_command_family("git status"), CommandFamily::Git);
        assert_eq!(detect_command_family("git diff HEAD~1"), CommandFamily::Git);
    }

    #[test]
    fn detects_test_runners() {
        assert_eq!(detect_command_family("cargo test -p foo"), CommandFamily::TestRunner);
        assert_eq!(detect_command_family("pytest tests/"), CommandFamily::TestRunner);
        assert_eq!(detect_command_family("go test ./..."), CommandFamily::TestRunner);
        assert_eq!(detect_command_family("vitest run"), CommandFamily::TestRunner);
    }

    #[test]
    fn detects_build() {
        assert_eq!(detect_command_family("cargo build --release"), CommandFamily::Build);
        assert_eq!(detect_command_family("tsc --noEmit"), CommandFamily::Build);
    }

    #[test]
    fn detects_filesystem() {
        assert_eq!(detect_command_family("ls -la"), CommandFamily::Filesystem);
        assert_eq!(detect_command_family("tree -L 2"), CommandFamily::Filesystem);
    }

    #[test]
    fn detects_search() {
        assert_eq!(detect_command_family("grep -rn TODO"), CommandFamily::Search);
        assert_eq!(detect_command_family("rg --files"), CommandFamily::Search);
    }

    #[test]
    fn detects_docker() {
        assert_eq!(detect_command_family("docker ps"), CommandFamily::Docker);
        assert_eq!(detect_command_family("kubectl get pods"), CommandFamily::Docker);
    }

    #[test]
    fn detects_cloud() {
        assert_eq!(detect_command_family("aws s3 ls"), CommandFamily::Cloud);
        assert_eq!(detect_command_family("gcloud projects list"), CommandFamily::Cloud);
    }

    #[test]
    fn fallback_to_generic() {
        assert_eq!(detect_command_family("echo hi"), CommandFamily::Generic);
        assert_eq!(detect_command_family("curl example.com"), CommandFamily::Generic);
    }
}
