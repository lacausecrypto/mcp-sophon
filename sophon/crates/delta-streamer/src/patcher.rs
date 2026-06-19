use regex::Regex;

use crate::{
    differ::DiffOperation,
    protocol::{EditAnchor, EditOperation, StructuredEdit, SymbolKind},
};

#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error("Invalid line range: start={start}, count={count}, total={total}")]
    InvalidRange {
        start: usize,
        count: usize,
        total: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum EditError {
    #[error("Anchor not found: {0}")]
    AnchorNotFound(String),

    #[error("Anchor not unique: {0}")]
    AnchorNotUnique(String),

    #[error("Invalid line range: {0}-{1}")]
    InvalidLineRange(usize, usize),

    #[error(
        "Overlapping edits: ranges {0}-{1} and {2}-{3} touch the same lines; \
         apply them in separate calls or merge them into one edit"
    )]
    OverlappingEdits(usize, usize, usize, usize),
}

/// Apply diff operations to content.
pub fn apply_diff(base: &str, operations: &[DiffOperation]) -> Result<String, PatchError> {
    let (mut lines, had_trailing_newline) = split_lines(base);
    let eol = detect_eol(base);

    // Apply non-keep operations from bottom to top to avoid index-shift issues.
    let mut indexed_ops = operations
        .iter()
        .enumerate()
        .filter(|(_, op)| !matches!(op, DiffOperation::Keep { .. }))
        .collect::<Vec<_>>();

    indexed_ops.sort_by(|(_, a), (_, b)| op_position(b).cmp(&op_position(a)));

    for (_, op) in indexed_ops {
        match op {
            DiffOperation::Keep { .. } => {}
            DiffOperation::Delete { start, count } => {
                let idx = start.saturating_sub(1);
                if idx + count > lines.len() {
                    return Err(PatchError::InvalidRange {
                        start: *start,
                        count: *count,
                        total: lines.len(),
                    });
                }
                lines.drain(idx..idx + count);
            }
            DiffOperation::Insert {
                at,
                lines: new_lines,
            } => {
                let idx = at.saturating_sub(1).min(lines.len());
                for (offset, line) in new_lines.iter().enumerate() {
                    lines.insert(idx + offset, line.clone());
                }
            }
            DiffOperation::Replace {
                start,
                delete_count,
                new_lines,
            } => {
                let idx = start.saturating_sub(1);
                if idx + delete_count > lines.len() {
                    return Err(PatchError::InvalidRange {
                        start: *start,
                        count: *delete_count,
                        total: lines.len(),
                    });
                }

                lines.splice(idx..idx + delete_count, new_lines.iter().cloned());
            }
        }
    }

    Ok(join_lines(&lines, had_trailing_newline, eol))
}

/// Apply structured edits (safer than raw diff).
pub fn apply_structured_edits(
    content: &str,
    edits: &[StructuredEdit],
) -> Result<String, EditError> {
    let (mut lines, had_trailing_newline) = split_lines(content);
    let eol = detect_eol(content);

    let mut resolved = Vec::new();
    for edit in edits {
        let (start, end) = resolve_anchor(&lines, &edit.anchor)?;
        resolved.push((start, end, edit.operation.clone()));
    }

    // Reject overlapping edits *before* touching anything. The edits are
    // applied bottom-to-top, which is only correct when no two edits touch
    // the same lines: two edits whose anchor ranges intersect (e.g. a
    // Replace on lines 1-2 and another on 2-3) splice over each other and
    // silently destroy data. A file-writing tool must never corrupt — so
    // we fail loudly instead and let the caller split or merge the edits.
    // O(n^2) is fine: edit batches are tiny.
    for i in 0..resolved.len() {
        for j in (i + 1)..resolved.len() {
            let (a_start, a_end, _) = &resolved[i];
            let (b_start, b_end, _) = &resolved[j];
            if ranges_overlap(*a_start, *a_end, *b_start, *b_end) {
                return Err(EditError::OverlappingEdits(
                    *a_start, *a_end, *b_start, *b_end,
                ));
            }
        }
    }

    resolved.sort_by(|(a_start, _, _), (b_start, _, _)| b_start.cmp(a_start));

    for (start, end, operation) in resolved {
        let start_idx = start.saturating_sub(1);
        let end_idx = end.min(lines.len());

        match operation {
            EditOperation::Replace { new_content } => {
                let replacement = new_content
                    .lines()
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>();
                if start_idx > end_idx || end_idx > lines.len() {
                    return Err(EditError::InvalidLineRange(start, end));
                }
                lines.splice(start_idx..end_idx, replacement);
            }
            EditOperation::InsertBefore { content } => {
                let insert = content
                    .lines()
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>();
                if start_idx > lines.len() {
                    return Err(EditError::InvalidLineRange(start, end));
                }
                for (offset, line) in insert.into_iter().enumerate() {
                    lines.insert(start_idx + offset, line);
                }
            }
            EditOperation::InsertAfter { content } => {
                let insert = content
                    .lines()
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>();
                let at = end_idx.min(lines.len());
                for (offset, line) in insert.into_iter().enumerate() {
                    lines.insert(at + offset, line);
                }
            }
            EditOperation::Delete => {
                if start_idx > end_idx || end_idx > lines.len() {
                    return Err(EditError::InvalidLineRange(start, end));
                }
                lines.drain(start_idx..end_idx);
            }
        }
    }

    Ok(join_lines(&lines, had_trailing_newline, eol))
}

/// Whether two 1-based inclusive line ranges intersect. Point anchors
/// (resolved to `start == end`) are treated as a single occupied line, so
/// two edits targeting the same anchor are correctly flagged as
/// overlapping rather than silently clobbering each other.
fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn resolve_anchor(lines: &[String], anchor: &EditAnchor) -> Result<(usize, usize), EditError> {
    match anchor {
        EditAnchor::LineRange { start, end } => {
            if *start == 0 || *start > *end || *end > lines.len() {
                return Err(EditError::InvalidLineRange(*start, *end));
            }
            Ok((*start, *end))
        }
        EditAnchor::UniqueText { text } => {
            let matches = lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line.contains(text))
                .map(|(idx, _)| idx + 1)
                .collect::<Vec<_>>();

            match matches.as_slice() {
                [single] => Ok((*single, *single)),
                [] => Err(EditError::AnchorNotFound(text.clone())),
                _ => Err(EditError::AnchorNotUnique(text.clone())),
            }
        }
        EditAnchor::Symbol { name, kind } => resolve_symbol(lines, name, kind),
    }
}

fn resolve_symbol(
    lines: &[String],
    name: &str,
    kind: &SymbolKind,
) -> Result<(usize, usize), EditError> {
    let pattern = match kind {
        SymbolKind::Function => format!(
            r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+{}\b",
            regex::escape(name)
        ),
        SymbolKind::Class => format!(r"^\s*class\s+{}\b", regex::escape(name)),
        SymbolKind::Struct => format!(r"^\s*(?:pub\s+)?struct\s+{}\b", regex::escape(name)),
        SymbolKind::Enum => format!(r"^\s*(?:pub\s+)?enum\s+{}\b", regex::escape(name)),
    };

    let re = Regex::new(&pattern).map_err(|_| EditError::AnchorNotFound(name.to_string()))?;
    for (idx, line) in lines.iter().enumerate() {
        if re.is_match(line) {
            let end = find_block_end(lines, idx + 1);
            return Ok((idx + 1, end));
        }
    }

    Err(EditError::AnchorNotFound(name.to_string()))
}

fn find_block_end(lines: &[String], start_line: usize) -> usize {
    let mut balance = 0i32;
    let mut seen_brace = false;

    for (idx, line) in lines.iter().enumerate().skip(start_line - 1) {
        for c in line.chars() {
            if c == '{' {
                balance += 1;
                seen_brace = true;
            } else if c == '}' {
                balance -= 1;
            }
        }

        if seen_brace && balance <= 0 {
            return idx + 1;
        }
    }

    start_line
}

fn split_lines(content: &str) -> (Vec<String>, bool) {
    let trailing = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    (lines, trailing)
}

/// The dominant line ending of `content`. `str::lines()` discards `\r`, so
/// without restoring it on join a CRLF (Windows) file gets silently
/// rewritten as LF — F3. We treat the file as CRLF if it contains any
/// `\r\n`, LF otherwise.
fn detect_eol(content: &str) -> &'static str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

fn join_lines(lines: &[String], trailing_newline: bool, eol: &str) -> String {
    if lines.is_empty() {
        return if trailing_newline {
            eol.to_string()
        } else {
            String::new()
        };
    }

    let mut out = lines.join(eol);
    if trailing_newline {
        out.push_str(eol);
    }
    out
}

fn op_position(op: &DiffOperation) -> usize {
    match op {
        DiffOperation::Keep { start, .. } => *start,
        DiffOperation::Delete { start, .. } => *start,
        DiffOperation::Insert { at, .. } => *at,
        DiffOperation::Replace { start, .. } => *start,
    }
}
