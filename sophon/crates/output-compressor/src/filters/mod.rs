//! Filter registry — a list of `FilterConfig`s tried in order, with a
//! generic fallback at the end.

pub mod docker;
pub mod filesystem;
pub mod generic;
pub mod git;
pub mod test_runners;

use crate::strategy::FilterConfig;

#[derive(Debug)]
pub struct FilterRegistry {
    filters: Vec<FilterConfig>,
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterRegistry {
    pub fn new() -> Self {
        Self {
            filters: vec![
                // Git family
                git::git_push_pull_filter(),
                git::git_status_filter(),
                git::git_log_filter(),
                git::git_diff_filter(),
                // Test runners
                test_runners::cargo_test_filter(),
                test_runners::pytest_filter(),
                test_runners::vitest_filter(),
                test_runners::go_test_filter(),
                // Filesystem
                filesystem::ls_filter(),
                filesystem::grep_filter(),
                filesystem::find_filter(),
                // Docker
                docker::docker_ps_filter(),
                docker::docker_logs_filter(),
                // Fallback MUST be last
                generic::generic_filter(),
            ],
        }
    }

    pub fn find_filter(&self, command: &str) -> &FilterConfig {
        for f in &self.filters {
            if f.command_patterns.iter().any(|r| r.is_match(command)) {
                return f;
            }
        }
        // The generic filter has a `.*` pattern so this is unreachable,
        // but guard anyway.
        self.filters.last().expect("at least the generic filter")
    }

    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_filters() {
        let r = FilterRegistry::new();
        // 4 git + 4 tests + 3 fs + 2 docker + 1 generic = 14
        assert_eq!(r.filter_count(), 14);
    }

    #[test]
    fn git_status_routes_to_git_status_filter() {
        let r = FilterRegistry::new();
        let f = r.find_filter("git status");
        assert_eq!(f.name, "git_status");
    }

    #[test]
    fn cargo_test_routes_to_cargo_test_filter() {
        let r = FilterRegistry::new();
        let f = r.find_filter("cargo test");
        assert_eq!(f.name, "cargo_test");
    }

    #[test]
    fn unknown_command_routes_to_generic() {
        let r = FilterRegistry::new();
        let f = r.find_filter("something_weird_no_one_knows");
        assert_eq!(f.name, "generic");
    }
}
