use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};

use crate::compressor::CompressionResult;

#[derive(Debug, Clone)]
struct CacheEntry {
    result: CompressionResult,
    created_at: DateTime<Utc>,
    last_access: DateTime<Utc>,
}

#[derive(Debug)]
pub struct PromptCache {
    max_entries: usize,
    entries: HashMap<String, CacheEntry>,
    order: VecDeque<String>,
}

impl PromptCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn make_key(
        &self,
        content_hash: &str,
        query: &str,
        max_tokens: usize,
        min_tokens: usize,
    ) -> String {
        format!("{content_hash}::{query}::{max_tokens}::{min_tokens}")
    }

    pub fn get(&mut self, key: &str) -> Option<CompressionResult> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_access = Utc::now();
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.to_string());
            return Some(entry.result.clone());
        }
        None
    }

    pub fn insert(&mut self, key: String, result: CompressionResult) {
        let now = Utc::now();
        if self.entries.contains_key(&key) {
            self.entries.insert(
                key.clone(),
                CacheEntry {
                    result,
                    created_at: now,
                    last_access: now,
                },
            );
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
            return;
        }

        self.entries.insert(
            key.clone(),
            CacheEntry {
                result,
                created_at: now,
                last_access: now,
            },
        );
        self.order.push_back(key);

        while self.entries.len() > self.max_entries {
            if let Some(oldest_key) = self.order.pop_front() {
                self.entries.remove(&oldest_key);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn oldest_entry_age_seconds(&self) -> Option<i64> {
        let oldest = self.entries.values().map(|entry| entry.created_at).min()?;
        Some((Utc::now() - oldest).num_seconds())
    }
}
