use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::differ::DiffOperation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReadRequest {
    pub path: PathBuf,
    pub known_version: Option<u64>,
    pub known_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileReadResponse {
    Full {
        content: String,
        version: u64,
        hash: String,
        token_count: usize,
    },
    Delta {
        base_version: u64,
        new_version: u64,
        new_hash: String,
        operations: Vec<DiffOperation>,
        token_count: usize,
    },
    Unchanged {
        version: u64,
        hash: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWriteRequest {
    pub path: PathBuf,
    pub changes: FileChanges,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileChanges {
    Full {
        content: String,
    },
    Delta {
        base_version: u64,
        operations: Vec<DiffOperation>,
    },
    Structured {
        edits: Vec<StructuredEdit>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredEdit {
    pub anchor: EditAnchor,
    pub operation: EditOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditAnchor {
    LineRange { start: usize, end: usize },
    UniqueText { text: String },
    Symbol { name: String, kind: SymbolKind },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Class,
    Struct,
    Enum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditOperation {
    Replace { new_content: String },
    InsertBefore { content: String },
    InsertAfter { content: String },
    Delete,
}
