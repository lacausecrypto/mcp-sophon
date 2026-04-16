#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("migration failed from v{from} to v{to}: {reason}")]
    Migration { from: u32, to: u32, reason: String },

    #[error("session not found: {0}")]
    SessionNotFound(String),
}
