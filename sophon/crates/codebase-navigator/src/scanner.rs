//! Directory scanner — walks a repository root, applies a hard
//! exclusion list (build artefacts / VCS / dependencies), and hands
//! the matching files to the extractor registry.
//!
//! Two scan strategies coexist:
//!
//! 1. **Git-aware (preferred)** — when `root` is a git repository and
//!    `prefer_git` is on, we shell out to
//!    `git ls-files -z --cached --others --exclude-standard`. This
//!    respects the repo's `.gitignore`, `.git/info/exclude`, and
//!    global excludes for free, without us having to parse any of it.
//! 2. **`ignore` crate fallback** — used when the root is not a git
//!    repo, when `git` is unavailable, or when the caller explicitly
//!    disables the git path via `prefer_git=false`. We use the
//!    `ignore` crate from BurntSushi (the one `ripgrep` uses), which
//!    parses `.gitignore`, `.ignore`, global excludes and hidden-file
//!    rules natively — so the walkdir fallback is no longer
//!    correctness-lossy on repos with custom ignore patterns.
//!
//! Both strategies converge on the same [`FileRecord`] shape so the
//! rest of the crate is backend-agnostic.
//!
//! File reads and symbol extraction run in parallel via `rayon`. On
//! a large Rust codebase (serde: 208 files) this takes the fresh scan
//! from ~1 500 ms to ~300 ms by saturating all cores during tree-
//! sitter parsing, which is the dominant cost when the feature is on.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use ignore::WalkBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::extractors::ExtractorRegistry;
use crate::types::Symbol;

/// Directories whose whole subtree is skipped unconditionally by the
/// fallback walker, on top of anything `.gitignore` / `.ignore`
/// already filters out. These are patterns that are rarely in a
/// project's ignore file but still never contain source to index.
pub const HARD_EXCLUDE_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".jj",
    "node_modules",
    "target",
    "build",
    "dist",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".next",
    ".nuxt",
    ".turbo",
    ".cache",
    "vendor",
    "Pods",
    "DerivedData",
];

/// Individual files that are always skipped by the fallback walker.
pub const HARD_EXCLUDE_FILES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "go.sum",
];

/// One scanned file: path (relative to the scan root), the symbols we
/// extracted from it, and the original file contents so the ranker
/// can look for symbol mentions later on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub relative_path: String,
    pub extension: String,
    pub symbols: Vec<Symbol>,
    /// Raw file content, used as the search corpus for the reference
    /// graph. Not serialised because it would blow up the digest.
    #[serde(skip)]
    pub content: String,
    pub byte_size: u64,
    pub line_count: u32,
    /// Seconds since `UNIX_EPOCH` of the file's last modification
    /// time when it was read. Paired with `byte_size` this gives
    /// `Navigator::scan` a cheap way to decide whether a file needs
    /// re-reading on a follow-up scan. A `0` value means "unknown",
    /// which forces a re-read.
    #[serde(default)]
    pub mtime_secs: u64,
    /// Which backend picked this file up. Useful for diagnostics.
    #[serde(default)]
    pub scan_source: ScanSource,
}

/// Metadata-only view of a scan candidate. Used by the incremental
/// scan path: we build this list cheaply, diff it against the
/// previous `FileRecord`s, and re-read only files whose mtime or
/// size changed.
#[derive(Debug, Clone)]
pub struct ScanCandidate {
    pub relative_path: String,
    pub absolute_path: PathBuf,
    pub byte_size: u64,
    pub mtime_secs: u64,
}

/// Which scan strategy produced a [`FileRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScanSource {
    /// Listed by `git ls-files`. Respects `.gitignore`.
    GitLsFiles,
    /// Walked by the `ignore` crate with hard exclusions layered on
    /// top of `.gitignore` / `.ignore`. Kept as `Walkdir` in the
    /// enum name for backwards compatibility with persisted caches
    /// from older Sophon versions.
    #[default]
    Walkdir,
}

/// Scan a directory and return one [`FileRecord`] per source file
/// whose extension is recognised by the [`ExtractorRegistry`].
///
/// Uses the git-aware strategy if `prefer_git` is `true` and `root`
/// is a git repository; otherwise falls back to the `ignore` walker.
/// The fallback is silent — callers who want to know which strategy
/// ran can inspect the `scan_source` field of each record.
///
/// `max_files` bounds the scan — the first N matching files are kept,
/// the rest are dropped. This is a cheap safeguard against accidentally
/// pointing Sophon at `/` on a poorly configured box.
pub fn scan_directory<P: AsRef<Path>>(
    root: P,
    registry: &ExtractorRegistry,
    max_files: usize,
    max_file_size: u64,
    prefer_git: bool,
) -> std::io::Result<Vec<FileRecord>> {
    let root = root.as_ref().to_path_buf();
    let root_canonical = root.canonicalize().unwrap_or_else(|_| root.clone());

    if prefer_git {
        if let Some(paths) = git_list_files(&root_canonical) {
            return Ok(scan_explicit_paths(
                &root_canonical,
                paths,
                registry,
                max_files,
                max_file_size,
                ScanSource::GitLsFiles,
            ));
        }
    }

    Ok(scan_with_ignore_walker(
        &root_canonical,
        registry,
        max_files,
        max_file_size,
    ))
}

/// Try `git ls-files` against `root`. Returns `None` on any failure
/// (not a git repo, git binary missing, non-zero exit). Never panics,
/// never logs — the caller just falls back to the ignore walker.
pub fn git_list_files(root: &Path) -> Option<Vec<PathBuf>> {
    // Cheap presence check: `.git` can be a directory (normal clone)
    // or a file (linked worktree). Either way, absence means not a repo.
    if !root.join(".git").exists() {
        return None;
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Parse the NUL-delimited output. Using `-z` means filenames with
    // embedded newlines / quotes / non-UTF-8 bytes are still handled
    // correctly; we only reject non-UTF-8 paths silently.
    let mut out = Vec::new();
    for chunk in output.stdout.split(|b| *b == 0) {
        if chunk.is_empty() {
            continue;
        }
        if let Ok(s) = std::str::from_utf8(chunk) {
            out.push(PathBuf::from(s));
        }
    }
    Some(out)
}

/// Read an explicit list of relative paths (from git-ls-files or any
/// other source) and produce [`FileRecord`]s in parallel. Rayon gives
/// us a ~5× speed-up on large Rust repos where tree-sitter parsing
/// dominates the wall-clock.
fn scan_explicit_paths(
    root: &Path,
    paths: Vec<PathBuf>,
    registry: &ExtractorRegistry,
    max_files: usize,
    max_file_size: u64,
    source: ScanSource,
) -> Vec<FileRecord> {
    // First filter metadata sequentially so we can enforce `max_files`
    // deterministically — otherwise a parallel filter would emit a
    // different subset depending on thread scheduling.
    let mut candidates: Vec<(PathBuf, PathBuf, std::fs::Metadata)> = Vec::new();
    for rel in paths {
        if candidates.len() >= max_files {
            break;
        }
        let absolute = root.join(&rel);
        let Ok(metadata) = std::fs::metadata(&absolute) else {
            continue;
        };
        if !metadata.is_file() || metadata.len() > max_file_size {
            continue;
        }
        if registry.for_path(&absolute).is_none() {
            continue;
        }
        candidates.push((rel, absolute, metadata));
    }

    // Then read + parse in parallel. `rayon`'s work-stealing scheduler
    // keeps every core busy even when file sizes are wildly uneven.
    candidates
        .into_par_iter()
        .filter_map(|(rel, absolute, metadata)| {
            let extractor = registry.for_path(&absolute)?;
            let content = std::fs::read_to_string(&absolute).ok()?;
            let symbols = extractor.extract(&content);
            let ext = absolute
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let line_count = content.lines().count() as u32;
            Some(FileRecord {
                relative_path: rel.to_string_lossy().to_string(),
                extension: ext,
                symbols,
                content,
                byte_size: metadata.len(),
                line_count,
                mtime_secs: mtime_secs(&metadata),
                scan_source: source,
            })
        })
        .collect()
}

/// Metadata-only pass over the scan space. Walks exactly the same
/// files that [`scan_directory`] would read, but stops at the
/// metadata level — no file content is loaded. Used by the
/// incremental scan path in [`crate::navigator::Navigator::scan`] to
/// detect which files need re-reading.
pub fn list_scan_candidates(
    root: &Path,
    registry: &ExtractorRegistry,
    max_files: usize,
    max_file_size: u64,
    prefer_git: bool,
) -> std::io::Result<(Vec<ScanCandidate>, ScanSource)> {
    let root_canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    // Prefer git when available — same decision as scan_directory.
    if prefer_git {
        if let Some(paths) = git_list_files(&root_canonical) {
            let mut out = Vec::new();
            for rel in paths {
                if out.len() >= max_files {
                    break;
                }
                let absolute = root_canonical.join(&rel);
                let Ok(metadata) = std::fs::metadata(&absolute) else {
                    continue;
                };
                if !metadata.is_file() || metadata.len() > max_file_size {
                    continue;
                }
                if registry.for_path(&absolute).is_none() {
                    continue;
                }
                out.push(ScanCandidate {
                    relative_path: rel.to_string_lossy().to_string(),
                    absolute_path: absolute,
                    byte_size: metadata.len(),
                    mtime_secs: mtime_secs(&metadata),
                });
            }
            return Ok((out, ScanSource::GitLsFiles));
        }
    }

    // `ignore` walker fallback.
    let mut out = Vec::new();
    for entry in walk_with_ignore(&root_canonical).flatten() {
        if out.len() >= max_files {
            break;
        }
        if entry.file_type().map(|t| !t.is_file()).unwrap_or(true) {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = std::fs::metadata(path) else {
            continue;
        };
        if metadata.len() > max_file_size {
            continue;
        }
        if registry.for_path(path).is_none() {
            continue;
        }
        out.push(ScanCandidate {
            relative_path: relative_to(path, &root_canonical),
            absolute_path: path.to_path_buf(),
            byte_size: metadata.len(),
            mtime_secs: mtime_secs(&metadata),
        });
    }
    Ok((out, ScanSource::Walkdir))
}

/// Read a single file given its metadata and produce a [`FileRecord`].
/// Used by the incremental scan path for "updated" and "added" files.
pub fn read_candidate(
    candidate: &ScanCandidate,
    registry: &ExtractorRegistry,
    source: ScanSource,
) -> Option<FileRecord> {
    let extractor = registry.for_path(&candidate.absolute_path)?;
    let content = std::fs::read_to_string(&candidate.absolute_path).ok()?;
    let symbols = extractor.extract(&content);
    let ext = candidate
        .absolute_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let line_count = content.lines().count() as u32;
    Some(FileRecord {
        relative_path: candidate.relative_path.clone(),
        extension: ext,
        symbols,
        content,
        byte_size: candidate.byte_size,
        line_count,
        mtime_secs: candidate.mtime_secs,
        scan_source: source,
    })
}

fn mtime_secs(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build the `ignore::Walk` we use in the fallback path. Layered:
///
/// - `.gitignore`, `.ignore`, global git excludes, parent ignore
///   files — all on by default in the `ignore` crate.
/// - Hidden files (`.foo`) — ON (enabled). Sophon treats them as
///   potential config / source; `.gitignore` already catches the
///   noise.
/// - A hard-coded directory skip layer for `target/`, `node_modules/`,
///   etc., for repos that don't list them in `.gitignore` (rare but
///   it happens, and we have benched users who ran Sophon against
///   unchecked-in build trees).
fn walk_with_ignore(root: &Path) -> ignore::Walk {
    let exclude_dirs: HashSet<&'static str> = HARD_EXCLUDE_DIRS.iter().copied().collect();
    let exclude_files: HashSet<&'static str> = HARD_EXCLUDE_FILES.iter().copied().collect();

    WalkBuilder::new(root)
        .hidden(false) // include dot-files; .gitignore handles the noise
        .parents(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        // Honour `.gitignore` even outside a git repo. Without this,
        // the `ignore` crate silently drops ignore files when there's
        // no `.git/` — which defeats half the reason we use this crate.
        .require_git(false)
        .follow_links(false)
        .filter_entry(move |entry| {
            let name = entry.file_name().to_string_lossy();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir && exclude_dirs.contains(name.as_ref()) {
                return false;
            }
            if !is_dir && exclude_files.contains(name.as_ref()) {
                return false;
            }
            true
        })
        .build()
}

/// `ignore`-based scan. Replaces the old walkdir path. Reads and
/// parses files in parallel via rayon.
fn scan_with_ignore_walker(
    root_canonical: &Path,
    registry: &ExtractorRegistry,
    max_files: usize,
    max_file_size: u64,
) -> Vec<FileRecord> {
    // Collect metadata sequentially (single-threaded walk) so we can
    // enforce `max_files` deterministically, then parallelise the
    // expensive read+parse stage.
    let mut candidates: Vec<(PathBuf, std::fs::Metadata)> = Vec::new();
    for entry in walk_with_ignore(root_canonical).flatten() {
        if candidates.len() >= max_files {
            break;
        }
        if entry.file_type().map(|t| !t.is_file()).unwrap_or(true) {
            continue;
        }
        let path = entry.path().to_path_buf();
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if metadata.len() > max_file_size {
            continue;
        }
        if registry.for_path(&path).is_none() {
            continue;
        }
        candidates.push((path, metadata));
    }

    let root = root_canonical.to_path_buf();
    candidates
        .into_par_iter()
        .filter_map(|(path, metadata)| {
            let extractor = registry.for_path(&path)?;
            let content = std::fs::read_to_string(&path).ok()?;
            let symbols = extractor.extract(&content);
            let relative = relative_to(&path, &root);
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let line_count = content.lines().count() as u32;
            Some(FileRecord {
                relative_path: relative,
                extension: ext,
                symbols,
                content,
                byte_size: metadata.len(),
                line_count,
                mtime_secs: mtime_secs(&metadata),
                scan_source: ScanSource::Walkdir,
            })
        })
        .collect()
}

fn relative_to(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    /// Returns `Some(())` if `git` is on PATH and succeeds at a no-op
    /// invocation; returns `None` otherwise so git-dependent tests
    /// can self-skip instead of failing on machines without git.
    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn git_init(root: &Path) {
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(root)
            .output()
            .expect("git init");
        // Some CI environments don't have user.email / user.name set,
        // which makes `git add` fail. We only run ls-files so we
        // don't actually need to commit, but set them anyway for
        // robustness.
        Command::new("git")
            .args(["config", "user.email", "ci@example.com"])
            .current_dir(root)
            .output()
            .ok();
        Command::new("git")
            .args(["config", "user.name", "ci"])
            .current_dir(root)
            .output()
            .ok();
    }

    // ---------- ignore-walker path ----------

    #[test]
    fn scans_rust_and_python_files() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("src/lib.rs"), "pub fn hello() {}");
        write(&dir.path().join("scripts/build.py"), "def build(): pass");

        let records = scan_directory(
            dir.path(),
            &ExtractorRegistry::new(),
            1000,
            1_000_000,
            false,
        )
        .unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|r| r.scan_source == ScanSource::Walkdir));
    }

    #[test]
    fn skips_excluded_directories() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");
        write(
            &dir.path().join("target/debug/should_be_skipped.rs"),
            "pub fn ghost() {}",
        );
        write(
            &dir.path().join("node_modules/foo/index.js"),
            "export function ghost() {}",
        );

        let records = scan_directory(
            dir.path(),
            &ExtractorRegistry::new(),
            1000,
            1_000_000,
            false,
        )
        .unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].relative_path.ends_with("lib.rs"));
    }

    #[test]
    fn skips_files_larger_than_limit() {
        let dir = tempdir().unwrap();
        let big = "pub fn x() {}\n".repeat(1000);
        write(&dir.path().join("huge.rs"), &big);
        write(&dir.path().join("small.rs"), "pub fn small() {}");

        let records =
            scan_directory(dir.path(), &ExtractorRegistry::new(), 1000, 100, false).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].relative_path, "small.rs");
    }

    #[test]
    fn max_files_bounds_the_walk() {
        let dir = tempdir().unwrap();
        for i in 0..20 {
            write(&dir.path().join(format!("file_{}.rs", i)), "pub fn f() {}");
        }
        let records =
            scan_directory(dir.path(), &ExtractorRegistry::new(), 5, 1_000_000, false).unwrap();
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn skips_files_with_unknown_extension() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("notes.txt"), "just text, not code");
        write(&dir.path().join("readme.md"), "# docs");
        write(&dir.path().join("lib.rs"), "pub fn f() {}");
        let records = scan_directory(
            dir.path(),
            &ExtractorRegistry::new(),
            1000,
            1_000_000,
            false,
        )
        .unwrap();
        assert_eq!(records.len(), 1);
    }

    /// `ignore` walker should honour a `.gitignore` even *without* the
    /// prefer_git path — that's the new correctness win over the old
    /// walkdir-based fallback.
    #[test]
    fn walker_honours_gitignore_without_git_path() {
        let dir = tempdir().unwrap();
        write(&dir.path().join(".gitignore"), "secret.rs\ngenerated/\n");
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");
        write(&dir.path().join("src/secret.rs"), "pub fn leak() {}");
        write(&dir.path().join("generated/auto.rs"), "pub fn ghost() {}");

        let records = scan_directory(
            dir.path(),
            &ExtractorRegistry::new(),
            1000,
            1_000_000,
            false,
        )
        .unwrap();
        // The `ignore` crate parses .gitignore even without a .git dir
        // being present. All records tagged Walkdir (fallback path).
        assert!(records.iter().all(|r| r.scan_source == ScanSource::Walkdir));
        assert!(records.iter().any(|r| r.relative_path.ends_with("lib.rs")));
        assert!(!records
            .iter()
            .any(|r| r.relative_path.ends_with("secret.rs")));
        assert!(!records.iter().any(|r| r.relative_path.ends_with("auto.rs")));
    }

    // ---------- git path ----------

    #[test]
    fn git_list_files_none_for_non_repo() {
        let dir = tempdir().unwrap();
        assert!(git_list_files(dir.path()).is_none());
    }

    #[test]
    fn git_list_files_returns_some_for_real_repo() {
        if !git_available() {
            eprintln!("[skip] git not available on this machine");
            return;
        }
        let dir = tempdir().unwrap();
        git_init(dir.path());
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");

        let paths = git_list_files(dir.path()).expect("git ls-files should succeed");
        // Untracked but not ignored → shown via --others --exclude-standard
        assert!(paths.iter().any(|p| p.to_string_lossy().contains("lib.rs")));
    }

    #[test]
    fn prefer_git_respects_gitignore() {
        if !git_available() {
            eprintln!("[skip] git not available on this machine");
            return;
        }
        let dir = tempdir().unwrap();
        git_init(dir.path());
        write(&dir.path().join(".gitignore"), "generated/\nsecret.rs\n");
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");
        write(&dir.path().join("src/secret.rs"), "pub fn leak() {}");
        write(&dir.path().join("generated/auto.rs"), "pub fn ghost() {}");

        let records =
            scan_directory(dir.path(), &ExtractorRegistry::new(), 1000, 1_000_000, true).unwrap();

        // git path should tag every record with GitLsFiles
        assert!(records
            .iter()
            .all(|r| r.scan_source == ScanSource::GitLsFiles));
        // lib.rs is neither tracked nor ignored → included via --others
        assert!(records.iter().any(|r| r.relative_path.ends_with("lib.rs")));
        // secret.rs is in .gitignore → excluded
        assert!(!records
            .iter()
            .any(|r| r.relative_path.ends_with("secret.rs")));
        // generated/ is in .gitignore → excluded
        assert!(!records.iter().any(|r| r.relative_path.ends_with("auto.rs")));
    }

    #[test]
    fn prefer_git_false_forces_ignore_walker_even_in_repo() {
        if !git_available() {
            eprintln!("[skip] git not available on this machine");
            return;
        }
        let dir = tempdir().unwrap();
        git_init(dir.path());
        write(&dir.path().join(".gitignore"), "secret.rs\n");
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");
        write(&dir.path().join("src/secret.rs"), "pub fn leak() {}");

        let records = scan_directory(
            dir.path(),
            &ExtractorRegistry::new(),
            1000,
            1_000_000,
            false,
        )
        .unwrap();
        // `ignore` walker is used and *also* parses .gitignore, so
        // secret.rs should now be excluded even with prefer_git=false.
        // This is the correctness win over the old walkdir fallback.
        assert!(records.iter().all(|r| r.scan_source == ScanSource::Walkdir));
        assert!(!records
            .iter()
            .any(|r| r.relative_path.ends_with("secret.rs")));
        assert!(records.iter().any(|r| r.relative_path.ends_with("lib.rs")));
    }

    #[test]
    fn broken_git_dir_falls_back_silently() {
        // Create a `.git` file (not dir) that is not a valid gitdir
        // pointer. `git ls-files` will fail; we should silently fall
        // back to the ignore walker.
        let dir = tempdir().unwrap();
        write(&dir.path().join(".git"), "gitdir: nowhere/valid\n");
        write(&dir.path().join("src/lib.rs"), "pub fn ok() {}");
        let records =
            scan_directory(dir.path(), &ExtractorRegistry::new(), 1000, 1_000_000, true).unwrap();
        // The important thing is we don't panic and lib.rs still gets found.
        assert!(records.iter().any(|r| r.relative_path.ends_with("lib.rs")));
    }
}
