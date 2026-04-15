use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sophon_core::{hashing::hash_content, tokens::count_tokens};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    pub id: String,
    pub content: String,
    pub hash: String,
    pub token_count: usize,
    pub use_count: u64,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    pub category: FragmentCategory,
    pub tags: Vec<String>,
}

impl Default for Fragment {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content: String::new(),
            hash: String::new(),
            token_count: 0,
            use_count: 0,
            created_at: now,
            last_used: now,
            category: FragmentCategory::Template,
            tags: vec![],
        }
    }
}

impl Fragment {
    pub fn from_content(content: String, category: FragmentCategory, tags: Vec<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            token_count: count_tokens(&content),
            hash: hash_content(&content),
            content,
            use_count: 0,
            created_at: now,
            last_used: now,
            category,
            tags,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FragmentCategory {
    CodeBoilerplate,
    SystemInstruction,
    DataStructure,
    Template,
    Definition,
}

#[derive(Debug)]
pub struct FragmentStore {
    fragments: HashMap<String, Fragment>,
    hash_index: HashMap<String, String>,
    storage_path: PathBuf,
    max_fragments: usize,
}

impl FragmentStore {
    pub fn new_memory() -> Self {
        Self {
            fragments: HashMap::new(),
            hash_index: HashMap::new(),
            storage_path: PathBuf::new(),
            max_fragments: 1_000,
        }
    }

    pub fn new(storage_path: impl AsRef<Path>, max_fragments: usize) -> Result<Self, std::io::Error> {
        let path = storage_path.as_ref().to_path_buf();
        let mut store = Self {
            fragments: HashMap::new(),
            hash_index: HashMap::new(),
            storage_path: path,
            max_fragments: max_fragments.max(1),
        };
        store.load()?;
        Ok(store)
    }

    pub fn load(&mut self) -> Result<(), std::io::Error> {
        if self.storage_path.as_os_str().is_empty() || !self.storage_path.exists() {
            return Ok(());
        }

        let raw = fs::read_to_string(&self.storage_path)?;
        let list: Vec<Fragment> = serde_json::from_str(&raw).unwrap_or_default();

        self.fragments.clear();
        self.hash_index.clear();

        for fragment in list {
            self.hash_index
                .insert(fragment.hash.clone(), fragment.id.clone());
            self.fragments.insert(fragment.id.clone(), fragment);
        }

        Ok(())
    }

    pub fn save(&self) -> Result<(), std::io::Error> {
        if self.storage_path.as_os_str().is_empty() {
            return Ok(());
        }
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut list = self.fragments.values().cloned().collect::<Vec<_>>();
        list.sort_by(|a, b| b.use_count.cmp(&a.use_count));
        let raw = serde_json::to_string_pretty(&list).unwrap_or_else(|_| "[]".to_string());
        fs::write(&self.storage_path, raw)
    }

    pub fn add(&mut self, mut fragment: Fragment) -> String {
        if let Some(existing_id) = self.hash_index.get(&fragment.hash).cloned() {
            self.touch(&existing_id);
            return existing_id;
        }

        fragment.last_used = Utc::now();
        let id = fragment.id.clone();
        self.hash_index.insert(fragment.hash.clone(), id.clone());
        self.fragments.insert(id.clone(), fragment);
        self.evict_if_needed();
        let _ = self.save();
        id
    }

    pub fn get(&self, id: &str) -> Option<&Fragment> {
        self.fragments.get(id)
    }

    pub fn touch(&mut self, id: &str) {
        if let Some(fragment) = self.fragments.get_mut(id) {
            fragment.use_count += 1;
            fragment.last_used = Utc::now();
        }
    }

    pub fn contains_hash(&self, hash: &str) -> bool {
        self.hash_index.contains_key(hash)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Fragment> {
        self.fragments.values()
    }

    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    fn evict_if_needed(&mut self) {
        if self.fragments.len() <= self.max_fragments {
            return;
        }

        let mut fragments = self.fragments.values().cloned().collect::<Vec<_>>();
        fragments.sort_by(|a, b| a.last_used.cmp(&b.last_used));

        while self.fragments.len() > self.max_fragments {
            if let Some(fragment) = fragments.first() {
                self.hash_index.remove(&fragment.hash);
                self.fragments.remove(&fragment.id);
                fragments.remove(0);
            } else {
                break;
            }
        }
    }
}
