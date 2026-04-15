pub mod differ;
pub mod hasher;
pub mod patcher;
pub mod protocol;
pub mod state;

use std::{fs, io::Write, path::Path};

use differ::generate_diff;
use protocol::{FileChanges, FileReadResponse, FileWriteRequest};
use sophon_core::{error::SophonError, hashing::{hash_content, hash_lines}, tokens::count_tokens};
use state::{FileState, StateStore};

/// Delta streamer engine that serves delta-aware file read/write operations.
#[derive(Debug)]
pub struct DeltaStreamer {
    state: StateStore,
}

impl DeltaStreamer {
    pub fn new(max_files: usize) -> Self {
        Self {
            state: StateStore::new(max_files),
        }
    }

    pub fn read_file_delta<P: AsRef<Path>>(
        &mut self,
        path: P,
        known_version: Option<u64>,
        known_hash: Option<&str>,
    ) -> Result<FileReadResponse, SophonError> {
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() {
            return Err(SophonError::FileNotFound(path_buf));
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
                || known_version.map(|v| v == existing.version).unwrap_or(false)
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

                let delta_tokens = count_tokens(&format!("{:?}", operations));
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

    pub fn write_file_delta(&mut self, request: FileWriteRequest) -> Result<FileReadResponse, SophonError> {
        use patcher::{apply_diff, apply_structured_edits};

        let path = request.path;
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
                    return Err(SophonError::VersionMismatch {
                        expected: current_version,
                        actual: base_version,
                    });
                }
                apply_diff(&existing_content, &operations)
                    .map_err(|e| SophonError::ParseError(e.to_string()))?
            }
            FileChanges::Structured { edits } => {
                apply_structured_edits(&existing_content, &edits)
                    .map_err(|e| SophonError::InvalidAnchor(e.to_string()))?
            }
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

/// Atomic write: stage to a sibling temp file, fsync, then rename.
/// `rename` is atomic on the same filesystem, so readers see either the
/// old or the new content — never a partially-written file.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), SophonError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| SophonError::ParseError("path has no file name".into()))?
        .to_string_lossy();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{}.sophon-{}-{}.tmp", file_name, std::process::id(), nonce));

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
