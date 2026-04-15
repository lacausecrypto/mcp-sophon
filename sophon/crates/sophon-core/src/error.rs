use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SophonError {
    #[error("Token limit exceeded: {current} > {max}")]
    TokenLimitExceeded { current: usize, max: usize },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u64, actual: u64 },

    #[error("Fragment not found: {0}")]
    FragmentNotFound(String),

    #[error("Invalid anchor: {0}")]
    InvalidAnchor(String),

    #[error("Multimodal error: {0}")]
    MultimodalError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
