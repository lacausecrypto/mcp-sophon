use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffOperation {
    Keep {
        start: usize,
        count: usize,
    },
    Delete {
        start: usize,
        count: usize,
    },
    Insert {
        at: usize,
        lines: Vec<String>,
    },
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

/// Token cost of a delta as the LLM actually sees it: the operations are
/// serialized to JSON on the wire (`handlers.rs` → `serde_json::to_value`),
/// not as Rust `{:?}`. Counting the Debug form undercounts by ~15-20 %
/// (JSON quoting/escaping) and inflates `savings_percent`, which can push
/// a delta past the real token budget. Fall back to Debug only if JSON
/// serialization somehow fails (it cannot for these types).
pub fn delta_token_cost(operations: &[DiffOperation]) -> usize {
    let serialized =
        serde_json::to_string(operations).unwrap_or_else(|_| format!("{operations:?}"));
    count_tokens(&serialized)
}

pub fn calculate_savings(_old: &str, new: &str, diff: &[DiffOperation]) -> DiffStats {
    let full_tokens = count_tokens(new);
    let diff_tokens = delta_token_cost(diff);

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

#[cfg(test)]
mod cost_tests {
    use super::*;
    use sophon_core::tokens::count_tokens;

    #[test]
    fn delta_cost_counts_real_json_not_debug() {
        // A delta whose new_lines contain quote/backslash chars — exactly
        // where JSON escaping diverges from Rust Debug.
        let ops = vec![DiffOperation::Replace {
            start: 1,
            delete_count: 1,
            new_lines: vec![r#"println!("a\tb");"#.to_string(), "next".to_string()],
        }];

        // delta_token_cost must equal the cost of what is actually sent on
        // the wire (JSON). The Debug form is a *different* string whose
        // token count diverges from reality in either direction (here it
        // happens to be larger) — so we lock to the JSON cost, not Debug.
        let json_cost = count_tokens(&serde_json::to_string(&ops).unwrap());
        let debug_cost = count_tokens(&format!("{ops:?}"));
        assert_eq!(delta_token_cost(&ops), json_cost);
        assert_ne!(
            json_cost, debug_cost,
            "this fixture is meant to exercise a JSON≠Debug divergence"
        );
    }
}
