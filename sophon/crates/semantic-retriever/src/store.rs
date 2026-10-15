//! JSONL-backed chunk store.
//!
//! One line of JSON per chunk. Append-only on disk; we keep an in-memory
//! `HashMap<id, Chunk>` for O(1) lookups by id. Re-loading a store rebuilds
//! the map by streaming the file once.
//!
//! Why JSONL instead of SQLite (the original spec): zero new C dependencies,
//! same persistence pattern as `memory_manager::with_persistence`, and the
//! file is `cat`-able / `jq`-able for debugging. For the chunk counts Sophon
//! sees this is plenty fast (writes are append-only, reads happen once at
//! startup or never).

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug)]
pub struct ChunkStore {
    path: PathBuf,
    by_id: HashMap<String, Chunk>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoreEntry {
    chunk: Chunk,
}

/// Borrow the chunk for serialization without cloning potentially large
/// embedding vectors during a persistent batch commit.
#[derive(Serialize)]
struct StoreEntryRef<'a> {
    chunk: &'a Chunk,
}

impl ChunkStore {
    /// Open or create a chunk store at `path`. If the file exists, all
    /// existing entries are loaded into memory.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut by_id: HashMap<String, Chunk> = HashMap::new();
        if path.exists() {
            let f = OpenOptions::new().read(true).open(&path)?;
            for line in BufReader::new(f).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<StoreEntry>(&line) {
                    Ok(entry) => {
                        by_id.insert(entry.chunk.id.clone(), entry.chunk);
                    }
                    Err(_) => {
                        // Skip malformed lines rather than failing the whole
                        // load — a half-written line from a previous crash
                        // shouldn't brick the store.
                        continue;
                    }
                }
            }
        }

        Ok(Self { path, by_id })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&Chunk> {
        self.by_id.get(id)
    }

    /// Return all stored chunks. Used to rebuild the vector index after
    /// a process restart.
    pub fn iter(&self) -> impl Iterator<Item = &Chunk> {
        self.by_id.values()
    }

    /// Commit an entire batch as one durable file replacement. A failure
    /// during serialization, writing, or rename leaves both the old JSONL
    /// snapshot and the in-memory map unchanged. Existing chunk IDs remain
    /// idempotent; imported malformed/truncated rows are repaired because the
    /// snapshot is written from the validated in-memory map.
    pub fn insert_batch(&mut self, chunks: Vec<Chunk>) -> Result<usize, StoreError> {
        let mut new_ids = HashSet::new();
        let staged: Vec<Chunk> = chunks
            .into_iter()
            .filter(|chunk| {
                !self.by_id.contains_key(&chunk.id) && new_ids.insert(chunk.id.clone())
            })
            .collect();
        if staged.is_empty() {
            return Ok(0);
        }

        let mut bytes = Vec::new();
        for chunk in self.by_id.values().chain(staged.iter()) {
            serde_json::to_writer(&mut bytes, &StoreEntryRef { chunk })?;
            bytes.push(b'\n');
        }
        sophon_core::atomic_file::replace(&self.path, &bytes)?;
        let added = staged.len();
        for chunk in staged {
            self.by_id.insert(chunk.id.clone(), chunk);
        }
        Ok(added)
    }

    /// Insert a single chunk (idempotent). Use insert_batch() for multi-chunk
    /// indexing to avoid repeatedly rebuilding the durable snapshot.
    pub fn insert(&mut self, chunk: Chunk) -> Result<bool, StoreError> {
        Ok(self.insert_batch(vec![chunk])? != 0)
    }

    /// Truncate both file and in-memory store. Do not clear RAM unless the
    /// empty durable snapshot was committed successfully.
    pub fn clear(&mut self) -> Result<(), StoreError> {
        if self.path.exists() {
            sophon_core::atomic_file::replace(&self.path, b"")?;
        }
        self.by_id.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkType;
    use chrono::Utc;
    use tempfile::tempdir;

    fn sample_chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            source_message_indices: vec![0],
            content: content.to_string(),
            token_count: 3,
            timestamp: Utc::now(),
            session_id: None,
            chunk_type: ChunkType::UserStatement,
            embedding: None,
        }
    }

    #[test]
    fn open_creates_empty_store() {
        let dir = tempdir().unwrap();
        let store = ChunkStore::open(dir.path().join("chunks.jsonl")).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn insert_and_get() {
        let dir = tempdir().unwrap();
        let mut store = ChunkStore::open(dir.path().join("chunks.jsonl")).unwrap();
        let inserted = store.insert(sample_chunk("a1", "hello")).unwrap();
        assert!(inserted);
        assert!(store.contains("a1"));
        assert_eq!(store.get("a1").unwrap().content, "hello");
    }

    #[test]
    fn duplicate_insert_is_noop() {
        let dir = tempdir().unwrap();
        let mut store = ChunkStore::open(dir.path().join("chunks.jsonl")).unwrap();
        assert!(store.insert(sample_chunk("a1", "hello")).unwrap());
        assert!(!store.insert(sample_chunk("a1", "hello")).unwrap());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn multi_chunk_batch_commits_atomically_and_preserves_old_snapshot_on_failure() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunks.jsonl");
        let backup = dir.path().join("saved.jsonl");
        let mut store = ChunkStore::open(&path).unwrap();
        assert_eq!(
            store.insert_batch(vec![
                sample_chunk("a", "first"),
                sample_chunk("b", "second"),
            ]).unwrap(),
            2
        );
        let committed = fs::read(&path).unwrap();
        assert_eq!(committed.iter().filter(|&&b| b == b'\n').count(), 2);
        assert_eq!(store.insert_batch(vec![sample_chunk("b", "second")]).unwrap(), 0);

        fs::rename(&path, &backup).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(store.insert_batch(vec![
            sample_chunk("c", "new first"),
            sample_chunk("d", "new second"),
        ]).is_err());
        assert_eq!(store.len(), 2);
        assert!(!store.contains("c"));
        assert!(!store.contains("d"));
        assert!(store.clear().is_err());
        assert_eq!(store.len(), 2);
        fs::remove_dir(&path).unwrap();
        fs::rename(&backup, &path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), committed);
        assert_eq!(ChunkStore::open(&path).unwrap().len(), 2);
    }

    #[test]
    fn persistence_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunks.jsonl");
        {
            let mut store = ChunkStore::open(&path).unwrap();
            store.insert(sample_chunk("p1", "first")).unwrap();
            store.insert(sample_chunk("p2", "second")).unwrap();
        }
        let store2 = ChunkStore::open(&path).unwrap();
        assert_eq!(store2.len(), 2);
        assert_eq!(store2.get("p1").unwrap().content, "first");
        assert_eq!(store2.get("p2").unwrap().content, "second");
    }
}
