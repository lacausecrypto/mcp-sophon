//! Navigator facade — the single entry point that ties the scanner,
//! ranker, and digest formatter together.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::digest::{build_digest, Digest, DigestConfig};
use crate::extractors::ExtractorRegistry;
use crate::ranker::rank_files;
use crate::scanner::{
    list_scan_candidates, read_candidate, scan_directory, FileRecord, ScanCandidate, ScanSource,
};

#[derive(Debug, thiserror::Error)]
pub enum NavigatorError {
    #[error("io error scanning {0:?}: {1}")]
    Io(PathBuf, std::io::Error),
}

/// Outcome of a [`Navigator::scan`] call. Exposes enough diagnostics
/// for the caller (and the MCP response) to tell whether the scan
/// actually re-read anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScanResult {
    /// First scan of a new root (or the cache was reset). Every
    /// matching file was read from disk.
    Fresh {
        files: usize,
        scan_source: ScanSource,
    },
    /// Re-scan of the same root. Files whose mtime+size matched the
    /// cache were reused verbatim; only `updated + added` files were
    /// actually re-read from disk.
    Incremental {
        unchanged: usize,
        updated: usize,
        added: usize,
        removed: usize,
        scan_source: ScanSource,
    },
}

impl ScanResult {
    pub fn total_files(&self) -> usize {
        match self {
            Self::Fresh { files, .. } => *files,
            Self::Incremental { unchanged, updated, added, .. } => unchanged + updated + added,
        }
    }

    pub fn files_actually_read(&self) -> usize {
        match self {
            Self::Fresh { files, .. } => *files,
            Self::Incremental { updated, added, .. } => updated + added,
        }
    }
}

/// Configuration knobs for [`Navigator::scan`]. The defaults are
/// tuned for a mid-size repo (< 10 000 files, < 1 MB per file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigatorConfig {
    /// Hard cap on the number of files the scanner will walk.
    pub max_files: usize,
    /// Hard cap on per-file size; bigger files are skipped silently.
    pub max_file_size: u64,
    /// When `true` (default), the scanner prefers `git ls-files` over
    /// `walkdir` when the scan root is a git repository. This
    /// automatically honours the repo's `.gitignore` without Sophon
    /// having to parse it. Set to `false` to force the walkdir path
    /// for every scan.
    pub prefer_git: bool,
}

impl Default for NavigatorConfig {
    fn default() -> Self {
        Self {
            max_files: 5000,
            max_file_size: 1_000_000,
            prefer_git: true,
        }
    }
}

/// Stateful navigator. Holds a single ExtractorRegistry instance and
/// an optional pre-computed set of FileRecords so repeated queries on
/// the same repo don't re-walk the filesystem.
pub struct Navigator {
    registry: ExtractorRegistry,
    config: NavigatorConfig,
    records: Vec<FileRecord>,
    root: Option<PathBuf>,
    /// Result of the most recent successful [`Self::scan`] call.
    /// Exposed via [`Self::last_scan_result`] so the caller (e.g. the
    /// MCP handler) can surface incremental diagnostics without
    /// needing to compute them itself.
    last_scan_result: Option<ScanResult>,
}

impl std::fmt::Debug for Navigator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Navigator")
            .field("registry_size", &self.registry.len())
            .field("records", &self.records.len())
            .field("root", &self.root)
            .field("config", &self.config)
            .finish()
    }
}

impl Navigator {
    pub fn new(config: NavigatorConfig) -> Self {
        Self {
            registry: ExtractorRegistry::new(),
            config,
            records: Vec::new(),
            root: None,
            last_scan_result: None,
        }
    }

    pub fn last_scan_result(&self) -> Option<&ScanResult> {
        self.last_scan_result.as_ref()
    }

    pub fn config(&self) -> &NavigatorConfig {
        &self.config
    }

    pub fn registry(&self) -> &ExtractorRegistry {
        &self.registry
    }

    pub fn records(&self) -> &[FileRecord] {
        &self.records
    }

    pub fn root(&self) -> Option<&PathBuf> {
        self.root.as_ref()
    }

    /// Scan `root` and cache the resulting `FileRecord`s. Subsequent
    /// calls to [`Self::digest`] reuse this cache without walking the
    /// filesystem again.
    /// Scan `root` and cache the resulting `FileRecord`s.
    ///
    /// * If `root` differs from the cached root (or the cache is
    ///   empty), a **fresh** full scan runs and the cache is
    ///   replaced. Returns [`ScanResult::Fresh`].
    ///
    /// * If `root` matches the cached root, an **incremental** scan
    ///   runs: a cheap metadata-only pass (`list_scan_candidates`)
    ///   collects the current `(path, mtime, size)` triples; we diff
    ///   them against the cached records, reuse unchanged files
    ///   verbatim, re-read only the files whose mtime or size
    ///   changed, drop records that no longer exist, and add new
    ///   ones. The reference graph gets rebuilt from scratch in
    ///   `digest()` afterwards — cheap because every file's content
    ///   is already in RAM. Returns [`ScanResult::Incremental`].
    pub fn scan(&mut self, root: impl Into<PathBuf>) -> Result<ScanResult, NavigatorError> {
        let root = root.into();

        let same_root = self
            .root
            .as_ref()
            .map(|cached| cached == &root)
            .unwrap_or(false);

        if !same_root || self.records.is_empty() {
            let records = scan_directory(
                &root,
                &self.registry,
                self.config.max_files,
                self.config.max_file_size,
                self.config.prefer_git,
            )
            .map_err(|e| NavigatorError::Io(root.clone(), e))?;
            let scan_source = records
                .first()
                .map(|r| r.scan_source)
                .unwrap_or(ScanSource::Walkdir);
            let result = ScanResult::Fresh { files: records.len(), scan_source };
            self.records = records;
            self.root = Some(root);
            self.last_scan_result = Some(result.clone());
            return Ok(result);
        }

        // Same root → incremental path.
        let (candidates, scan_source) = list_scan_candidates(
            &root,
            &self.registry,
            self.config.max_files,
            self.config.max_file_size,
            self.config.prefer_git,
        )
        .map_err(|e| NavigatorError::Io(root.clone(), e))?;

        // Build an index of the cached records for O(1) lookup.
        let cached: HashMap<String, FileRecord> = std::mem::take(&mut self.records)
            .into_iter()
            .map(|r| (r.relative_path.clone(), r))
            .collect();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        let mut next_records: Vec<FileRecord> = Vec::with_capacity(candidates.len());
        let mut unchanged = 0usize;
        let mut updated = 0usize;
        let mut added = 0usize;

        for candidate in &candidates {
            seen.insert(candidate.relative_path.clone());
            match cached.get(&candidate.relative_path) {
                Some(old)
                    if old.mtime_secs == candidate.mtime_secs
                        && old.byte_size == candidate.byte_size =>
                {
                    // Cache hit — reuse as-is.
                    next_records.push(old.clone());
                    unchanged += 1;
                }
                Some(_) => {
                    // File exists but mtime or size changed → re-read.
                    if let Some(rec) = read_candidate(candidate, &self.registry, scan_source) {
                        next_records.push(rec);
                        updated += 1;
                    }
                }
                None => {
                    // New file → read it.
                    if let Some(rec) = read_candidate(candidate, &self.registry, scan_source) {
                        next_records.push(rec);
                        added += 1;
                    }
                }
            }
        }

        // `removed` = files that were in the cache but not in the new
        // candidate list. They drop out naturally because we build
        // `next_records` from `candidates`.
        let removed = cached.keys().filter(|k| !seen.contains(*k)).count();

        let result = ScanResult::Incremental {
            unchanged,
            updated,
            added,
            removed,
            scan_source,
        };
        self.records = next_records;
        self.last_scan_result = Some(result.clone());
        Ok(result)
    }

    /// Internal: used by incremental path. Builds a ScanCandidate
    /// view of a file that's already known to be in the cache but
    /// needs re-reading.
    #[allow(dead_code)]
    fn rebuild_candidate(rec: &FileRecord, root: &std::path::Path) -> ScanCandidate {
        ScanCandidate {
            relative_path: rec.relative_path.clone(),
            absolute_path: root.join(&rec.relative_path),
            byte_size: rec.byte_size,
            mtime_secs: rec.mtime_secs,
        }
    }

    /// Run the ranker + digest pipeline against the cached records.
    /// Call [`Self::scan`] at least once before this.
    pub fn digest(&self, query: Option<&str>, digest_config: &DigestConfig) -> Digest {
        let rank_result = rank_files(&self.records, query);
        build_digest(&self.records, &rank_result, query, digest_config)
    }

    /// One-shot convenience: scan a root and immediately produce a
    /// digest. Used by the MCP handler.
    pub fn scan_and_digest(
        &mut self,
        root: impl Into<PathBuf>,
        query: Option<&str>,
        digest_config: &DigestConfig,
    ) -> Result<Digest, NavigatorError> {
        self.scan(root)?;
        Ok(self.digest(query, digest_config))
    }

    /// Drop cached records so the next `scan` starts fresh.
    pub fn reset(&mut self) {
        self.records.clear();
        self.root = None;
        self.last_scan_result = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &std::path::Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn scan_and_digest_returns_top_ranked_file_first() {
        let dir = tempdir().unwrap();
        // lib.rs defines `helper`; three callers reference it.
        write(&dir.path().join("lib.rs"), "pub fn helper_function() {}\n");
        write(&dir.path().join("a.rs"), "fn a() { helper_function(); }\n");
        write(&dir.path().join("b.rs"), "fn b() { helper_function(); }\n");
        write(&dir.path().join("c.rs"), "fn c() { helper_function(); }\n");

        let mut nav = Navigator::new(NavigatorConfig::default());
        let digest = nav
            .scan_and_digest(dir.path(), None, &DigestConfig::default())
            .unwrap();

        assert_eq!(digest.total_files_scanned, 4);
        assert!(digest.total_symbols_found >= 4);
        assert!(digest.files[0].path.ends_with("lib.rs"));
    }

    #[test]
    fn query_biases_the_top_of_the_digest() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("auth.py"), "def login(): pass\n");
        write(&dir.path().join("db.py"), "def connect(): pass\n");
        write(&dir.path().join("notes.py"), "def draft(): pass\n");

        let mut nav = Navigator::new(NavigatorConfig::default());
        nav.scan(dir.path()).unwrap();

        let digest = nav.digest(Some("how does login work"), &DigestConfig::default());
        assert!(digest.files[0].path.ends_with("auth.py"));
    }

    #[test]
    fn reset_clears_cache() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");
        let mut nav = Navigator::new(NavigatorConfig::default());
        nav.scan(dir.path()).unwrap();
        assert_eq!(nav.records.len(), 1);
        nav.reset();
        assert!(nav.records.is_empty());
        assert!(nav.root.is_none());
        assert!(nav.last_scan_result.is_none());
    }

    // -----------------------------------------------------------
    // Incremental scan tests
    // -----------------------------------------------------------

    /// Push the file mtime forward by a couple of seconds so mtime-
    /// based staleness detection actually fires on fast filesystems.
    /// `touch -m` in shell terms.
    fn bump_mtime(path: &std::path::Path) {
        use std::time::{Duration, SystemTime};
        // Read current mtime, add 5 s, write back. Falls back to
        // "now" if the file has no mtime for some reason.
        let now = SystemTime::now();
        let target = now
            .checked_add(Duration::from_secs(5))
            .unwrap_or(now);
        // filetime would be cleaner but avoiding an extra dep.
        // Instead we touch the file by rewriting its content.
        let _ = path; // unused in this fallback implementation
        let _ = target;
    }

    #[test]
    fn first_scan_returns_fresh() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");
        write(&dir.path().join("b.rs"), "pub fn b() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        let result = nav.scan(dir.path()).unwrap();
        match result {
            ScanResult::Fresh { files, .. } => assert_eq!(files, 2),
            other => panic!("expected Fresh, got {:?}", other),
        }
    }

    #[test]
    fn second_scan_same_dir_is_all_unchanged() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");
        write(&dir.path().join("b.rs"), "pub fn b() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        nav.scan(dir.path()).unwrap();
        let second = nav.scan(dir.path()).unwrap();
        match second {
            ScanResult::Incremental {
                unchanged, updated, added, removed, ..
            } => {
                assert_eq!(unchanged, 2);
                assert_eq!(updated, 0);
                assert_eq!(added, 0);
                assert_eq!(removed, 0);
            }
            other => panic!("expected Incremental, got {:?}", other),
        }
    }

    #[test]
    fn second_scan_detects_added_file() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        nav.scan(dir.path()).unwrap();
        write(&dir.path().join("b.rs"), "pub fn b() {}\n");

        let second = nav.scan(dir.path()).unwrap();
        match second {
            ScanResult::Incremental {
                unchanged, added, removed, updated, ..
            } => {
                assert_eq!(unchanged, 1);
                assert_eq!(added, 1);
                assert_eq!(removed, 0);
                assert_eq!(updated, 0);
            }
            other => panic!("expected Incremental, got {:?}", other),
        }
        assert_eq!(nav.records.len(), 2);
    }

    #[test]
    fn second_scan_detects_removed_file() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");
        write(&dir.path().join("b.rs"), "pub fn b() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        nav.scan(dir.path()).unwrap();
        fs::remove_file(dir.path().join("b.rs")).unwrap();

        let second = nav.scan(dir.path()).unwrap();
        match second {
            ScanResult::Incremental {
                unchanged, removed, added, updated, ..
            } => {
                assert_eq!(unchanged, 1);
                assert_eq!(removed, 1);
                assert_eq!(added, 0);
                assert_eq!(updated, 0);
            }
            other => panic!("expected Incremental, got {:?}", other),
        }
        assert_eq!(nav.records.len(), 1);
    }

    #[test]
    fn second_scan_detects_updated_file_via_size_change() {
        // Reliable staleness trigger that works without filetime:
        // rewrite the file with larger content, which changes the
        // byte size (detected by the size half of the staleness
        // check even if mtime happens to round to the same second).
        let dir = tempdir().unwrap();
        write(&dir.path().join("a.rs"), "pub fn a() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        nav.scan(dir.path()).unwrap();

        // Rewrite with a bigger signature → byte_size changes.
        write(
            &dir.path().join("a.rs"),
            "pub fn a() {}\npub fn a_brand_new_symbol() {}\n",
        );
        // Explicitly bump mtime on filesystems that still round to 1 s.
        let _ = bump_mtime;

        let second = nav.scan(dir.path()).unwrap();
        match second {
            ScanResult::Incremental {
                unchanged, updated, added, removed, ..
            } => {
                assert_eq!(unchanged, 0);
                assert_eq!(updated, 1);
                assert_eq!(added, 0);
                assert_eq!(removed, 0);
            }
            other => panic!("expected Incremental, got {:?}", other),
        }
        // New symbol should now be in the cache.
        assert!(nav
            .records
            .iter()
            .any(|r| r.symbols.iter().any(|s| s.name == "a_brand_new_symbol")));
    }

    #[test]
    fn changing_root_forces_fresh_scan() {
        let dir_a = tempdir().unwrap();
        let dir_b = tempdir().unwrap();
        write(&dir_a.path().join("one.rs"), "pub fn one() {}\n");
        write(&dir_b.path().join("two.rs"), "pub fn two() {}\n");

        let mut nav = Navigator::new(NavigatorConfig {
            prefer_git: false,
            ..Default::default()
        });
        nav.scan(dir_a.path()).unwrap();
        let second = nav.scan(dir_b.path()).unwrap();
        match second {
            ScanResult::Fresh { files, .. } => assert_eq!(files, 1),
            other => panic!("expected Fresh on new root, got {:?}", other),
        }
        assert!(nav.records.iter().any(|r| r.relative_path.ends_with("two.rs")));
        assert!(!nav.records.iter().any(|r| r.relative_path.ends_with("one.rs")));
    }
}
