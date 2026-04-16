//! SQLite-backed persistent storage for Sophon.
//!
//! Replaces the JSONL files used by `memory-manager`, `fragment-cache`,
//! and `semantic-retriever` with a single WAL-mode SQLite database.
//! WAL gives crash-safety without write-locking readers, and the
//! bundled `rusqlite` means no external SQLite install is required.

pub mod error;
pub mod schema;
pub mod sqlite;

pub use error::StorageError;
pub use sqlite::SqliteStorage;
