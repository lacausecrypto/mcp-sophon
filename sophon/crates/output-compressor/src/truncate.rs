//! Middle-truncation helpers — keep the interesting head and tail,
//! replace the middle with a one-line elision marker.

/// Truncate `input` to at most `max_lines` lines. If it's already small
/// enough the input is returned unchanged. Otherwise we keep `max_lines / 2`
/// lines from the head and `max_lines / 2` from the tail.
///
/// `omission_message` supports a `{n}` placeholder replaced with the
/// elided line count.
pub fn middle_truncate_lines(input: &str, max_lines: usize, omission_message: &str) -> String {
    if max_lines == 0 {
        return input.to_string();
    }
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() <= max_lines {
        return input.to_string();
    }
    let half = max_lines / 2;
    let elided = lines.len() - (half * 2);
    let mut out: Vec<String> = lines[..half].iter().map(|s| s.to_string()).collect();
    out.push(omission_message.replace("{n}", &elided.to_string()));
    out.extend(lines[lines.len() - half..].iter().map(|s| s.to_string()));
    out.join("\n")
}

/// Character-based middle truncation. Used by the budget cap pass
/// because we target a token (~= char/4) budget there.
///
/// `head_lines` and `tail_lines` are always preserved verbatim.
pub fn middle_truncate_chars(
    input: &str,
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
    omission_message: &str,
) -> String {
    if input.len() <= max_chars {
        return input.to_string();
    }

    let lines: Vec<&str> = input.lines().collect();
    let head = if head_lines >= lines.len() {
        lines.as_slice()
    } else {
        &lines[..head_lines]
    };
    let tail = if tail_lines >= lines.len().saturating_sub(head_lines) {
        &lines[head_lines..]
    } else {
        &lines[lines.len() - tail_lines..]
    };

    let head_str = head.join("\n");
    let tail_str = tail.join("\n");

    let used = head_str.len() + tail_str.len() + omission_message.len() + 8;
    if used >= max_chars {
        // Even the head+tail exceed the budget. Chop the tail from the head,
        // prefer keeping the head.
        let cutoff = max_chars.saturating_sub(omission_message.len() + 8);
        let head_cut: String = input.chars().take(cutoff).collect();
        return format!(
            "{}\n{}",
            head_cut,
            omission_message.replace("{n}", &input.len().saturating_sub(cutoff).to_string())
        );
    }

    let elided = input.len() - head_str.len() - tail_str.len();
    format!(
        "{}\n{}\n{}",
        head_str,
        omission_message.replace("{n}", &elided.to_string()),
        tail_str
    )
}

/// Head-truncate `input` to at most `max_tokens` tokens, cutting on line
/// boundaries. This is the *baseline* the safety floor compares against —
/// the same "keep the start" behaviour a naive caller would fall back to.
///
/// The token budget is enforced accurately (line-by-line, monotone) rather
/// than via a char estimate: the safety-floor invariant "compressed ≥
/// truncation-to-same-budget" only holds if this never overshoots the
/// budget on dense text (grep rows, JSON), which a chars/4 guess does.
pub fn head_truncate_tokens(input: &str, max_tokens: usize) -> String {
    use sophon_core::tokens::count_tokens;
    if max_tokens == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut total = 0usize;
    for line in input.lines() {
        // +1 approximates the newline joining this line to the previous.
        let cost = count_tokens(line) + 1;
        if total + cost > max_tokens && !out.is_empty() {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        total += cost;
    }
    // Degenerate case: even the first line blows the budget. Hard-cut by a
    // char estimate so we still return *something* from the head.
    if out.is_empty() {
        let approx = max_tokens.saturating_mul(4);
        out = input.chars().take(approx).collect();
    }
    out
}

/// Does `compressed` drop any non-blank line that the truncation
/// `baseline` keeps? Lines are matched by substring (so a dedup-collapsed
/// `foo (x3)` still counts as keeping `foo`). This is the safety-floor
/// predicate: if truncation preserves a line the compression lost, the
/// compression is *worse than truncation on that line* — and since
/// command output often buries the answer in an early line that an
/// aggregating filter collapses, we fall back to truncation. A pure
/// line-count comparison misses this (same count, different lines).
pub fn drops_lines_kept_by(compressed: &str, baseline: &str) -> bool {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for line in baseline.lines() {
        let t = line.trim();
        if t.is_empty() || !seen.insert(t) {
            continue;
        }
        if seen.len() > 4000 {
            break;
        }
        if !compressed.contains(t) {
            return true;
        }
    }
    false
}

/// How many *distinct* non-blank lines of `original` survive (as a
/// substring) in `candidate`. Substring rather than exact-line so a
/// dedup-collapsed `foo (x3)` still counts as covering `foo`, while a
/// fuzzily-merged-away distinct line does not. Used by tests to assert
/// the safety-floor guarantee (compressed ≥ truncation).
pub fn unique_line_coverage(original: &str, candidate: &str) -> usize {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    let mut covered = 0usize;
    for line in original.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if !seen.insert(t) {
            continue;
        }
        // Bound worst-case cost on pathological inputs.
        if seen.len() > 4000 {
            break;
        }
        if candidate.contains(t) {
            covered += 1;
        }
    }
    covered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_truncate_short_input_unchanged() {
        let input = "one\ntwo\nthree";
        assert_eq!(middle_truncate_lines(input, 10, "..."), input);
    }

    #[test]
    fn middle_truncate_long_input_keeps_head_and_tail() {
        let input = (1..=20)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let out = middle_truncate_lines(&input, 4, "... {n} skipped ...");
        assert!(out.contains("line 1"));
        assert!(out.contains("line 2"));
        assert!(out.contains("line 19"));
        assert!(out.contains("line 20"));
        assert!(out.contains("... 16 skipped ..."));
        // Interior lines must be gone
        assert!(!out.contains("line 10"));
    }

    #[test]
    fn middle_truncate_chars_respects_budget() {
        let input = "a".repeat(5000);
        let out = middle_truncate_chars(&input, 1000, 0, 0, "... {n} omitted ...");
        assert!(out.len() <= 1100);
        assert!(out.contains("omitted"));
    }
}
