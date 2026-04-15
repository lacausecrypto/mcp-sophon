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
