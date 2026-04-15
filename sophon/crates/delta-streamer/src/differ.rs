use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffOperation {
    Keep { start: usize, count: usize },
    Delete { start: usize, count: usize },
    Insert { at: usize, lines: Vec<String> },
    Replace {
        start: usize,
        delete_count: usize,
        new_lines: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct DiffStats {
    pub full_tokens: usize,
    pub diff_tokens: usize,
    pub savings_percent: f32,
}

/// Generate a compact diff between two text versions.
///
/// This implementation uses prefix/suffix anchoring to produce a minimal
/// contiguous replacement region, which is very efficient for iterative edits.
pub fn generate_diff(old: &str, new: &str) -> Vec<DiffOperation> {
    let old_lines = to_lines(old);
    let new_lines = to_lines(new);

    let mut prefix = 0usize;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }

    let mut old_suffix = old_lines.len();
    let mut new_suffix = new_lines.len();
    while old_suffix > prefix
        && new_suffix > prefix
        && old_lines[old_suffix - 1] == new_lines[new_suffix - 1]
    {
        old_suffix -= 1;
        new_suffix -= 1;
    }

    let mut ops = Vec::new();
    if prefix > 0 {
        ops.push(DiffOperation::Keep {
            start: 1,
            count: prefix,
        });
    }

    let old_mid = &old_lines[prefix..old_suffix];
    let new_mid = &new_lines[prefix..new_suffix];

    if !old_mid.is_empty() && new_mid.is_empty() {
        ops.push(DiffOperation::Delete {
            start: prefix + 1,
            count: old_mid.len(),
        });
    } else if old_mid.is_empty() && !new_mid.is_empty() {
        ops.push(DiffOperation::Insert {
            at: prefix + 1,
            lines: new_mid.to_vec(),
        });
    } else if !old_mid.is_empty() || !new_mid.is_empty() {
        ops.push(DiffOperation::Replace {
            start: prefix + 1,
            delete_count: old_mid.len(),
            new_lines: new_mid.to_vec(),
        });
    }

    let suffix_count = old_lines.len().saturating_sub(old_suffix);
    if suffix_count > 0 {
        ops.push(DiffOperation::Keep {
            start: old_suffix + 1,
            count: suffix_count,
        });
    }

    ops
}

pub fn calculate_savings(_old: &str, new: &str, diff: &[DiffOperation]) -> DiffStats {
    let full_tokens = count_tokens(new);
    let diff_tokens = count_tokens(&format!("{:?}", diff));

    let savings_percent = if full_tokens == 0 {
        0.0
    } else {
        (1.0 - diff_tokens as f32 / full_tokens as f32) * 100.0
    };

    DiffStats {
        full_tokens,
        diff_tokens,
        savings_percent,
    }
}

fn to_lines(content: &str) -> Vec<String> {
    content.lines().map(|line| line.to_string()).collect()
}
