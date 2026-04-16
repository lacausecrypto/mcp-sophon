use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::error::StorageError;
use crate::schema::SCHEMA_V1;

pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub token_count: usize,
    pub created_at: i64,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    pub id: String,
    pub content: String,
    pub hash: String,
    pub usage_count: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    pub total_original: u64,
    pub total_compressed: u64,
    pub call_count: u64,
}

impl SqliteStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;

        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "cache_size", "-64000")?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;

        conn.execute_batch(SCHEMA_V1)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA_V1)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // === Memory ===

    pub fn save_memory(&self, memory: &Memory) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO memories
             (id, session_id, role, content, token_count, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                memory.id,
                memory.session_id,
                memory.role,
                memory.content,
                memory.token_count as i64,
                memory.created_at,
                memory.metadata.as_ref().map(|m| m.to_string()),
            ],
        )?;
        Ok(())
    }

    pub fn load_memories(
        &self,
        session_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<Memory>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let sql = match limit {
            Some(n) => format!(
                "SELECT id, session_id, role, content, token_count, created_at, metadata
                 FROM memories WHERE session_id = ?1 ORDER BY created_at ASC LIMIT {n}"
            ),
            None => "SELECT id, session_id, role, content, token_count, created_at, metadata
                     FROM memories WHERE session_id = ?1 ORDER BY created_at ASC"
                .to_string(),
        };
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([session_id], |row| {
            Ok(Memory {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                token_count: row.get::<_, i64>(4)? as usize,
                created_at: row.get(5)?,
                metadata: row
                    .get::<_, Option<String>>(6)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn delete_session(&self, session_id: &str) -> Result<usize, StorageError> {
        let conn = self.conn.lock().unwrap();
        let count = conn.execute("DELETE FROM memories WHERE session_id = ?1", [session_id])?;
        Ok(count)
    }

    pub fn list_sessions(&self) -> Result<Vec<String>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT DISTINCT session_id FROM memories ORDER BY session_id")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // === Embeddings cache ===

    pub fn get_embedding(&self, content_hash: &[u8]) -> Result<Option<Vec<f32>>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT embedding FROM embeddings WHERE content_hash = ?1")?;
        match stmt.query_row([content_hash], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(bytes_to_f32(&blob))
        }) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save_embedding(
        &self,
        content_hash: &[u8],
        embedding: &[f32],
        model: &str,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO embeddings
             (content_hash, embedding, model, dimension, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                content_hash,
                f32_to_bytes(embedding),
                model,
                embedding.len() as i64,
                Utc::now().timestamp(),
            ],
        )?;
        Ok(())
    }

    // === Fragments ===

    pub fn get_fragment(&self, hash: &str) -> Result<Option<Fragment>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id, content, hash, usage_count FROM fragments WHERE hash = ?1")?;
        match stmt.query_row([hash], |row| {
            Ok(Fragment {
                id: row.get(0)?,
                content: row.get(1)?,
                hash: row.get(2)?,
                usage_count: row.get::<_, i64>(3)? as u32,
            })
        }) {
            Ok(f) => Ok(Some(f)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save_fragment(&self, fragment: &Fragment) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO fragments
             (id, content, hash, usage_count, last_used_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params![
                fragment.id,
                fragment.content,
                fragment.hash,
                fragment.usage_count as i64,
                now,
            ],
        )?;
        Ok(())
    }

    pub fn increment_fragment_usage(&self, hash: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE fragments SET usage_count = usage_count + 1, last_used_at = ?1 WHERE hash = ?2",
            params![Utc::now().timestamp(), hash],
        )?;
        Ok(())
    }

    // === Token stats ===

    pub fn record_compression(
        &self,
        session_id: &str,
        module: &str,
        original: usize,
        compressed: usize,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO token_stats (session_id, module, original_tokens, compressed_tokens, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_id,
                module,
                original as i64,
                compressed as i64,
                Utc::now().timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn get_stats(&self, session_id: Option<&str>) -> Result<TokenStats, StorageError> {
        let conn = self.conn.lock().unwrap();
        let (sql, param): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match session_id {
            Some(sid) => (
                "SELECT COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0), COUNT(*)
                 FROM token_stats WHERE session_id = ?1",
                vec![Box::new(sid.to_string())],
            ),
            None => (
                "SELECT COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0), COUNT(*)
                 FROM token_stats",
                vec![],
            ),
        };
        let mut stmt = conn.prepare(sql)?;
        let stats = stmt.query_row(rusqlite::params_from_iter(param.iter()), |row| {
            Ok(TokenStats {
                total_original: row.get::<_, i64>(0)? as u64,
                total_compressed: row.get::<_, i64>(1)? as u64,
                call_count: row.get::<_, i64>(2)? as u64,
            })
        })?;
        Ok(stats)
    }
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_memory() {
        let db = SqliteStorage::in_memory().unwrap();
        let mem = Memory {
            id: "m1".into(),
            session_id: "s1".into(),
            role: "user".into(),
            content: "hello".into(),
            token_count: 1,
            created_at: 1000,
            metadata: None,
        };
        db.save_memory(&mem).unwrap();
        let loaded = db.load_memories("s1", None).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "hello");
    }

    #[test]
    fn list_sessions() {
        let db = SqliteStorage::in_memory().unwrap();
        for (sid, id) in [("a", "1"), ("b", "2"), ("a", "3")] {
            db.save_memory(&Memory {
                id: id.into(),
                session_id: sid.into(),
                role: "user".into(),
                content: "x".into(),
                token_count: 1,
                created_at: 0,
                metadata: None,
            })
            .unwrap();
        }
        let sessions = db.list_sessions().unwrap();
        assert_eq!(sessions, vec!["a", "b"]);
    }

    #[test]
    fn delete_session() {
        let db = SqliteStorage::in_memory().unwrap();
        db.save_memory(&Memory {
            id: "m1".into(),
            session_id: "s1".into(),
            role: "user".into(),
            content: "x".into(),
            token_count: 1,
            created_at: 0,
            metadata: None,
        })
        .unwrap();
        let deleted = db.delete_session("s1").unwrap();
        assert_eq!(deleted, 1);
        assert!(db.load_memories("s1", None).unwrap().is_empty());
    }

    #[test]
    fn roundtrip_embedding() {
        let db = SqliteStorage::in_memory().unwrap();
        let hash = b"0123456789abcdef";
        let vec = vec![1.0f32, 2.0, 3.0];
        db.save_embedding(hash, &vec, "test-model").unwrap();
        let loaded = db.get_embedding(hash).unwrap().unwrap();
        assert_eq!(loaded, vec);
    }

    #[test]
    fn embedding_miss() {
        let db = SqliteStorage::in_memory().unwrap();
        assert!(db.get_embedding(b"nonexistent").unwrap().is_none());
    }

    #[test]
    fn roundtrip_fragment() {
        let db = SqliteStorage::in_memory().unwrap();
        let frag = Fragment {
            id: "f1".into(),
            content: "repeated block".into(),
            hash: "abc123".into(),
            usage_count: 0,
        };
        db.save_fragment(&frag).unwrap();
        let loaded = db.get_fragment("abc123").unwrap().unwrap();
        assert_eq!(loaded.content, "repeated block");

        db.increment_fragment_usage("abc123").unwrap();
        let updated = db.get_fragment("abc123").unwrap().unwrap();
        assert_eq!(updated.usage_count, 1);
    }

    #[test]
    fn token_stats() {
        let db = SqliteStorage::in_memory().unwrap();
        db.record_compression("s1", "prompt", 1000, 400).unwrap();
        db.record_compression("s1", "memory", 500, 200).unwrap();
        db.record_compression("s2", "prompt", 800, 300).unwrap();

        let s1 = db.get_stats(Some("s1")).unwrap();
        assert_eq!(s1.total_original, 1500);
        assert_eq!(s1.total_compressed, 600);
        assert_eq!(s1.call_count, 2);

        let all = db.get_stats(None).unwrap();
        assert_eq!(all.total_original, 2300);
        assert_eq!(all.call_count, 3);
    }

    #[test]
    fn file_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sophon.db");

        {
            let db = SqliteStorage::open(&path).unwrap();
            db.save_memory(&Memory {
                id: "m1".into(),
                session_id: "s1".into(),
                role: "user".into(),
                content: "persisted".into(),
                token_count: 1,
                created_at: 42,
                metadata: None,
            })
            .unwrap();
        }

        let db2 = SqliteStorage::open(&path).unwrap();
        let loaded = db2.load_memories("s1", None).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "persisted");
    }
}
