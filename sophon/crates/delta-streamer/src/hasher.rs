use sophon_core::hashing::{hash_content as core_hash_content, hash_lines as core_hash_lines};

/// Hash file content for comparison.
pub fn hash_content(content: &str) -> String {
    core_hash_content(content)
}

/// Hash each line for efficient diff.
pub fn hash_lines(content: &str) -> Vec<String> {
    core_hash_lines(content)
}
