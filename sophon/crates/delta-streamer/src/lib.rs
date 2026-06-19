pub mod differ;
pub mod hasher;
pub mod patcher;
pub mod protocol;
pub mod state;

use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

use differ::generate_diff;
use protocol::{FileChanges, FileReadResponse, FileWriteRequest};
use sophon_core::{
    error::SophonError,
    hashing::{hash_content, hash_lines},
    tokens::count_tokens,
};
use state::{FileState, StateStore};

/// Delta streamer engine that serves delta-aware file read/write operations.
#[derive(Debug)]
pub struct DeltaStreamer {
    state: StateStore,
    /// When set, every read/write path is confined to this (canonicalized)
    /// root: `..`, absolute paths, and symlinks that escape it are rejected.
    /// `None` disables confinement (the library default, kept for tests and
    /// embedders that do their own sandboxing). The MCP server always sets
    /// it — see [`DeltaStreamer::with_fs_root`].
    fs_root: Option<PathBuf>,
}

impl DeltaStreamer {
    pub fn new(max_files: usize) -> Self {
        Self {
            state: StateStore::new(max_files),
            fs_root: None,
        }
    }

    /// Confine all subsequent read/write paths to `root`. The root must
    /// exist (it is canonicalized up front). Any path that resolves outside
    /// it — via `..`, an absolute path, or a symlink — is rejected with
    /// [`SophonError::path_outside_root`] instead of touching the file.
    ///
    /// This is the F1 fix: without it, a prompt-injected
    /// `../../.ssh/id_rsa` reaches arbitrary files on disk.
    pub fn with_fs_root(mut self, root: impl AsRef<Path>) -> Result<Self, SophonError> {
        let canonical = root.as_ref().canonicalize().map_err(|e| {
            SophonError::Config(format!(
                "SOPHON_FS_ROOT {:?} is not a usable directory: {e}",
                root.as_ref()
            ))
        })?;
        self.fs_root = Some(canonical);
        Ok(self)
    }

    /// Resolve `path` against the configured root, rejecting any escape.
    /// Returns the path unchanged when confinement is disabled.
    fn resolve_confined(&self, path: &Path) -> Result<PathBuf, SophonError> {
        let Some(root) = self.fs_root.as_ref() else {
            return Ok(path.to_path_buf());
        };

        // An absolute input is checked as-is; a relative one is anchored at
        // the root. Either way we strip `.`/`..` lexically so a path like
        // `<root>/../../etc/passwd` collapses to `/etc/passwd`.
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let normalized = normalize_lexical(&candidate);

        // Containment is decided on the *canonical* form, not the lexical
        // one. Canonicalising resolves symlinks, which (a) blocks a symlink
        // inside the root that escapes it, and (b) — just as important for
        // not over-rejecting — lets a symlinked root work: `SOPHON_FS_ROOT=/tmp`
        // canonicalises to `/private/tmp` on macOS, and an input `/tmp/x`
        // must resolve to the same real location rather than be rejected for
        // a lexical `/tmp` vs `/private/tmp` mismatch. `root` was already
        // canonicalised in `with_fs_root`.
        let real = canonical_within(&normalized);
        if !real.starts_with(root) {
            return Err(SophonError::path_outside_root(real, root.clone()));
        }

        Ok(normalized)
    }

    pub fn read_file_delta<P: AsRef<Path>>(
        &mut self,
        path: P,
        known_version: Option<u64>,
        known_hash: Option<&str>,
    ) -> Result<FileReadResponse, SophonError> {
        let path_buf = self.resolve_confined(path.as_ref())?;
        if !path_buf.exists() {
            return Err(SophonError::file_not_found(path_buf));
        }

        let content = fs::read_to_string(&path_buf)?;
        let hash = hash_content(&content);
        let token_count = count_tokens(&content);

        // Stateless-resume: honor client-provided known_hash even if we have
        // no prior state for this path (in-memory store is lost across runs).
        if self.state.get(&path_buf).is_none() {
            if let Some(kh) = known_hash {
                if kh == hash {
                    let version = known_version.unwrap_or(1);
                    self.state.insert(FileState {
                        path: path_buf.clone(),
                        hash: hash.clone(),
                        version,
                        line_hashes: hash_lines(&content),
                        content: content.clone(),
                        token_count,
                    });
                    return Ok(FileReadResponse::Unchanged { version, hash });
                }
            }
        }

        if let Some(existing) = self.state.get(&path_buf).cloned() {
            if known_hash.map(|h| h == existing.hash).unwrap_or(false)
                || known_version
                    .map(|v| v == existing.version)
                    .unwrap_or(false)
            {
                if existing.hash == hash {
                    self.state.touch(&path_buf);
                    return Ok(FileReadResponse::Unchanged {
                        version: existing.version,
                        hash,
                    });
                }

                let operations = generate_diff(&existing.content, &content);
                let new_state = FileState {
                    path: path_buf.clone(),
                    hash: hash.clone(),
                    version: existing.version + 1,
                    line_hashes: hash_lines(&content),
                    content: content.clone(),
                    token_count,
                };
                self.state.insert(new_state.clone());

                let delta_tokens = crate::differ::delta_token_cost(&operations);
                return Ok(FileReadResponse::Delta {
                    base_version: existing.version,
                    new_version: new_state.version,
                    new_hash: hash,
                    operations,
                    token_count: delta_tokens,
                });
            }

            // Known version/hash doesn't match or wasn't provided.
            if existing.hash == hash {
                self.state.touch(&path_buf);
                return Ok(FileReadResponse::Full {
                    content,
                    version: existing.version,
                    hash,
                    token_count,
                });
            }

            let updated = FileState {
                path: path_buf.clone(),
                hash: hash.clone(),
                version: existing.version + 1,
                line_hashes: hash_lines(&content),
                content: content.clone(),
                token_count,
            };
            self.state.insert(updated.clone());
            return Ok(FileReadResponse::Full {
                content,
                version: updated.version,
                hash,
                token_count,
            });
        }

        let state = FileState {
            path: path_buf.clone(),
            hash: hash.clone(),
            version: 1,
            line_hashes: hash_lines(&content),
            content: content.clone(),
            token_count,
        };
        self.state.insert(state.clone());

        Ok(FileReadResponse::Full {
            content,
            version: state.version,
            hash,
            token_count,
        })
    }

    pub fn write_file_delta(
        &mut self,
        request: FileWriteRequest,
    ) -> Result<FileReadResponse, SophonError> {
        use patcher::{apply_diff, apply_structured_edits};

        let path = self.resolve_confined(&request.path)?;
        let existing_content = if path.exists() {
            fs::read_to_string(&path)?
        } else {
            String::new()
        };

        let current_state = self.state.get(&path).cloned();
        let current_version = current_state.as_ref().map(|s| s.version).unwrap_or(0);

        let new_content = match request.changes {
            FileChanges::Full { content } => content,
            FileChanges::Delta {
                base_version,
                operations,
            } => {
                if base_version != current_version {
                    return Err(SophonError::version_mismatch(current_version, base_version));
                }
                apply_diff(&existing_content, &operations)
                    .map_err(|e| SophonError::parse(e.to_string()))?
            }
            FileChanges::Structured { edits } => apply_structured_edits(&existing_content, &edits)
                .map_err(|e| SophonError::parse(e.to_string()))?,
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        atomic_write(&path, new_content.as_bytes())?;

        let hash = hash_content(&new_content);
        let token_count = count_tokens(&new_content);
        let new_version = current_version + 1;

        self.state.insert(FileState {
            path: path.clone(),
            hash: hash.clone(),
            version: new_version,
            line_hashes: hash_lines(&new_content),
            content: new_content.clone(),
            token_count,
        });

        Ok(FileReadResponse::Full {
            content: new_content,
            version: new_version,
            hash,
            token_count,
        })
    }

    pub fn state_store(&self) -> &StateStore {
        &self.state
    }
}

/// Strip `.` and `..` from a path lexically (without touching the
/// filesystem), so containment can be checked before any IO. `..` pops the
/// previous component but never climbs above the filesystem root, so the
/// result, combined with a `starts_with(root)` check, blocks traversal.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(segment) => out.push(segment),
        }
    }
    out
}

/// Resolve `path` to its real location for containment checks: canonicalise
/// the deepest ancestor that actually exists (resolving symlinks) and
/// re-attach the not-yet-existing tail. For a path that fully exists this is
/// just `canonicalize`; for a write to a new file it canonicalises the
/// existing parent directory and appends the new file name. Falls back to
/// the input if nothing along the chain canonicalises.
fn canonical_within(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(real) = existing.canonicalize() {
            let mut out = real;
            for name in tail.iter().rev() {
                out.push(name);
            }
            return out;
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name.to_os_string());
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Atomic write: stage to a sibling temp file, fsync, then rename.
/// `rename` is atomic on the same filesystem, so readers see either the
/// old or the new content — never a partially-written file.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), SophonError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| SophonError::parse("path has no file name".to_string()))?
        .to_string_lossy();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(
        ".{}.sophon-{}-{}.tmp",
        file_name,
        std::process::id(),
        nonce
    ));

    let mut f = fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);

    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}
