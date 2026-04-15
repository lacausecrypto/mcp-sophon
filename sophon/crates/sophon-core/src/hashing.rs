use sha2::{Digest, Sha256};

/// Hash full content with SHA-256.
pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Hash lines with truncated SHA-256 (16 chars) for lightweight diffing.
pub fn hash_lines(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|line| {
            let mut hasher = Sha256::new();
            hasher.update(line.as_bytes());
            format!("{:x}", hasher.finalize())[..16].to_string()
        })
        .collect()
}
