use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum SophonError {
    #[error("compression error: {0}")]
    Compression(#[from] CompressionError),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("embedder error: {0}")]
    Embedder(String),

    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("navigation error: {0}")]
    Navigation(#[from] NavigationError),

    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("configuration error: {0}")]
    Config(String),
}

#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("prompt too large: {size} tokens exceeds maximum {max}")]
    PromptTooLarge { size: usize, max: usize },

    #[error("no sections found in prompt")]
    NoSections,

    #[error("token budget exhausted after removing all optional sections")]
    BudgetExhausted,

    #[error("output compression failed for command '{command}': {reason}")]
    OutputCompression { command: String, reason: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid XML at line {line}: {message}")]
    InvalidXml { line: usize, message: String },

    #[error("unclosed tag: <{0}>")]
    UnclosedTag(String),

    #[error("mismatched tag: </{found}> (expected </{expected}>)")]
    MismatchedTag { found: String, expected: String },

    #[error("regex error: {0}")]
    Regex(String),

    #[error("parse error: {0}")]
    Other(String),
}

#[derive(Debug, thiserror::Error)]
pub enum NavigationError {
    #[error("root directory not found: {0}")]
    RootNotFound(PathBuf),

    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),

    #[error("git error: {0}")]
    Git(String),

    #[error("extractor failed for {path}: {reason}")]
    Extractor { path: PathBuf, reason: String },

    #[error("scan timeout after {0:?}")]
    Timeout(Duration),

    #[error("too many files: {count} exceeds limit {limit}")]
    TooManyFiles { count: usize, limit: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer '{0}' not available")]
    NotAvailable(String),

    #[error("encoding failed: {0}")]
    Encoding(String),

    #[error("decoding failed: {0}")]
    Decoding(String),
}

// Backwards-compat re-exports for crates that still use the old variants
impl SophonError {
    pub fn token_limit(current: usize, max: usize) -> Self {
        Self::Compression(CompressionError::PromptTooLarge {
            size: current,
            max,
        })
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(ParseError::Other(msg.into()))
    }

    pub fn file_not_found(path: PathBuf) -> Self {
        Self::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("file not found: {}", path.display()),
        ))
    }

    pub fn version_mismatch(expected: u64, actual: u64) -> Self {
        Self::Config(format!(
            "version mismatch: expected {expected}, got {actual}"
        ))
    }

    pub fn fragment_not_found(id: impl Into<String>) -> Self {
        Self::Config(format!("fragment not found: {}", id.into()))
    }
}
