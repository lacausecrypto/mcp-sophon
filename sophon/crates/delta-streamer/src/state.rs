use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub path: PathBuf,
    pub hash: String,
    pub version: u64,
    pub line_hashes: Vec<String>,
    pub content: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct StateStore {
    pub files: HashMap<PathBuf, FileState>,
    pub max_files: usize,
    pub access_order: Vec<PathBuf>,
}

impl StateStore {
    pub fn new(max_files: usize) -> Self {
        Self {
            files: HashMap::new(),
            max_files: max_files.max(1),
            access_order: Vec::new(),
        }
    }

    pub fn get(&self, path: &PathBuf) -> Option<&FileState> {
        self.files.get(path)
    }

    pub fn insert(&mut self, state: FileState) {
        let path = state.path.clone();
        self.files.insert(path.clone(), state);
        self.touch(&path);
        self.evict_if_needed();
    }

    pub fn touch(&mut self, path: &PathBuf) {
        self.access_order.retain(|p| p != path);
        self.access_order.push(path.clone());
    }

    fn evict_if_needed(&mut self) {
        while self.files.len() > self.max_files {
            if let Some(oldest) = self.access_order.first().cloned() {
                self.access_order.remove(0);
                self.files.remove(&oldest);
            } else {
                break;
            }
        }
    }
}
