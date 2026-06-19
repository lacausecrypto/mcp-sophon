//! Strip ANSI / terminal escape sequences from command output.
//!
//! Real-world tool output (`cargo test`, `pytest`, `npm`, `vitest`,
//! `docker`) is almost always colored. Every line then begins with a
//! `\x1b[..m` SGR sequence, which defeats the *anchored* regexes the
//! command filters rely on (`^test … ok$`, `^✓`, `^--- PASS`): the
//! anchor never matches the escape byte, so the filter degenerates into
//! a no-op and `compress_output` collapses to ~truncation. Stripping the
//! escapes once, up-front, lets every downstream filter see the text the
//! way its patterns expect — and removing the escape bytes is itself a
//! real token saving the model never needed.

use std::sync::OnceLock;

use regex::Regex;

/// Matches the escape sequences that show up in terminal output:
/// CSI (`ESC [ … final`, covers SGR colors, cursor moves, erase),
/// OSC (`ESC ] … BEL|ST`, window titles / hyperlinks), and lone
/// two-byte escapes (`ESC <0x40-0x5F>`). Deliberately conservative:
/// it does not touch ordinary control chars (tabs, newlines) that
/// carry layout meaning.
fn ansi_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            \x1b\[ [0-9;?]* [ -/]* [@-~]        # CSI: ESC [ params interm final
          | \x1b\] [^\x07\x1b]* (?:\x07|\x1b\\) # OSC: ESC ] … (BEL | ST)
          | \x1b [@-Z\\-_]                      # two-byte escape (ESC <Fe>)
            ",
        )
        .expect("ANSI escape regex is a valid literal")
    })
}

/// Remove ANSI/terminal escape sequences. Returns the input borrowed
/// unchanged when there is nothing to strip (the common case for plain
/// piped output), so the hot path allocates nothing.
pub fn strip_ansi(text: &str) -> std::borrow::Cow<'_, str> {
    // Fast path: no ESC byte at all → no escapes to strip.
    if !text.contains('\x1b') {
        return std::borrow::Cow::Borrowed(text);
    }
    ansi_re().replace_all(text, "")
}

/// Collapse carriage-return progress overwrites. Progress UIs (`pip`,
/// `cargo`, `docker pull`, download bars) redraw a line in place with a
/// bare `\r`, so a single logical line can carry dozens of intermediate
/// states the model never needs — only the final one. We keep the segment
/// after the last in-line `\r`. A trailing `\r` immediately before `\n`
/// (CRLF) is a *line ending*, not an overwrite, so it is dropped without
/// eating the line (it would otherwise blank every Windows line).
///
/// Returns the input borrowed unchanged when there is no `\r` at all.
pub fn collapse_carriage_returns(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('\r') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut first = true;
    for line in text.split('\n') {
        if !first {
            out.push('\n');
        }
        first = false;
        // Drop the CR of a CRLF pair (it sits at the end of this split).
        let line = line.strip_suffix('\r').unwrap_or(line);
        // Any remaining CR is an in-line overwrite: keep the last segment.
        match line.rfind('\r') {
            Some(pos) => out.push_str(&line[pos + 1..]),
            None => out.push_str(line),
        }
    }
    std::borrow::Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_progress_bar_to_final_state() {
        let s = "Downloading\r 10%\r 50%\r 100% done\nNext line";
        assert_eq!(collapse_carriage_returns(s), " 100% done\nNext line");
    }

    #[test]
    fn crlf_line_endings_are_preserved_as_lines() {
        let s = "line one\r\nline two\r\n";
        // CRLF → LF, no content eaten.
        assert_eq!(collapse_carriage_returns(s), "line one\nline two\n");
    }

    #[test]
    fn no_cr_is_borrowed_untouched() {
        let s = "plain\nlines";
        assert!(matches!(
            collapse_carriage_returns(s),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn strips_sgr_color_codes() {
        let colored = "\x1b[32mtest result\x1b[0m: \x1b[1mok\x1b[0m";
        assert_eq!(strip_ansi(colored), "test result: ok");
    }

    #[test]
    fn plain_text_is_borrowed_untouched() {
        let plain = "test foo ... ok\nrunning 3 tests";
        let out = strip_ansi(plain);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out, plain);
    }

    #[test]
    fn preserves_newlines_and_tabs() {
        let s = "\x1b[31mline1\x1b[0m\n\tindented";
        assert_eq!(strip_ansi(s), "line1\n\tindented");
    }

    #[test]
    fn strips_cursor_and_erase_sequences() {
        // `\x1b[2K` erase line, `\x1b[1G` cursor column — common in progress UIs.
        let s = "\x1b[2K\x1b[1GBuilding…";
        assert_eq!(strip_ansi(s), "Building…");
    }

    #[test]
    fn strips_osc_hyperlink() {
        let s = "\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\";
        assert_eq!(strip_ansi(s), "link");
    }

    #[test]
    fn unanchors_filter_patterns() {
        // The motivating case: an anchored `^test` pattern only matches
        // once the leading color escape is gone.
        let colored = "\x1b[32mtest tokens::works ... ok\x1b[0m";
        let stripped = strip_ansi(colored);
        assert!(stripped.starts_with("test "));
    }
}
