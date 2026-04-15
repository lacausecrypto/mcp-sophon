//! Test runner filters — cargo test, pytest, vitest/jest, go test.
//! Strategy: keep only failure signal, drop passing output.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

pub fn cargo_test_filter() -> FilterConfig {
    FilterConfig {
        name: "cargo_test",
        command_patterns: vec![rx(r"^\s*cargo\s+test")],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r"^test .+ \.\.\. ok$"),
                rx(r"^test .+ \.\.\. ignored"),
                rx(r"^running \d+ tests?$"),
                rx(r"^\s*Running "),
                rx(r"^\s*Compiling "),
                rx(r"^\s*Finished "),
                rx(r"^\s*Doc-tests "),
                rx(r"^\s*$"),
            ],
            keep_patterns: vec![
                rx(r"FAILED"),
                rx(r"^test .+ \.\.\. FAILED"),
                rx(r"^failures:"),
                rx(r"^---- .+ ----"),
                rx(r"thread .+ panicked"),
                rx(r"^test result:"),
                rx(r"^error"),
                rx(r"^warning"),
                rx(r"^note:"),
                rx(r"assertion"),
            ],
        }],
        max_output_tokens: Some(800),
        preserve_head: 0,
        preserve_tail: 5,
    }
}

pub fn pytest_filter() -> FilterConfig {
    FilterConfig {
        name: "pytest",
        command_patterns: vec![rx(r"^\s*pytest"), rx(r"^\s*python\s+-m\s+pytest")],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r"^=+ test session starts =+"),
                rx(r"^platform "),
                rx(r"^plugins:"),
                rx(r"^collected \d+ items"),
                rx(r"^rootdir:"),
                // Pytest prints "tests/foo.py::test_x PASSED [ 20% ]" —
                // the PASSED token sits mid-line, not anchored. Match
                // anywhere on the line.
                rx(r"\bPASSED\b"),
                rx(r"^\s*\.+\s*$"),
                rx(r"^$"),
            ],
            keep_patterns: vec![
                // These override the PASSED match above on the summary
                // line (which contains both).
                rx(r"FAILED"),
                rx(r"ERROR"),
                rx(r"^E\s+"),
                rx(r"^=+ FAILURES =+"),
                rx(r"^=+ ERRORS =+"),
                rx(r"^=+ short test summary"),
                rx(r"^_+ "),
                rx(r"AssertionError"),
            ],
        }],
        max_output_tokens: Some(800),
        preserve_head: 0,
        preserve_tail: 5,
    }
}

pub fn vitest_filter() -> FilterConfig {
    FilterConfig {
        name: "vitest_jest",
        command_patterns: vec![
            rx(r"^\s*vitest"),
            rx(r"^\s*jest"),
            rx(r"^\s*(npm|pnpm|yarn)\s+test"),
        ],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r"^✓"),
                rx(r"^\s+✓"),
                rx(r"^PASS "),
                rx(r"^\s*$"),
                rx(r"^\s+[\d.]+ms"),
            ],
            keep_patterns: vec![
                rx(r"^✗"),
                rx(r"^×"),
                rx(r"^FAIL "),
                rx(r"Error:"),
                rx(r"Expected"),
                rx(r"Received"),
                rx(r"Test Files"),
                rx(r"Tests\s+\d+"),
            ],
        }],
        max_output_tokens: Some(800),
        preserve_head: 0,
        preserve_tail: 5,
    }
}

pub fn go_test_filter() -> FilterConfig {
    FilterConfig {
        name: "go_test",
        command_patterns: vec![rx(r"^\s*go\s+test")],
        strategies: vec![CompressionStrategy::FilterLines {
            remove_patterns: vec![
                rx(r"^ok\s+"),
                rx(r"^=== RUN"),
                rx(r"^--- PASS:"),
                rx(r"^PASS$"),
                rx(r"^\s*$"),
                rx(r"^\?\s+"),
            ],
            keep_patterns: vec![
                rx(r"^--- FAIL:"),
                rx(r"^FAIL\s+"),
                rx(r"^FAIL$"),
                rx(r"panic:"),
                rx(r"Error"),
            ],
        }],
        max_output_tokens: Some(800),
        preserve_head: 0,
        preserve_tail: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn cargo_test_keeps_failures_only() {
        let input = r#"running 10 tests
test tests::test_add ... ok
test tests::test_sub ... ok
test tests::test_mul ... ok
test tests::test_div ... ok
test tests::test_edge ... FAILED
test tests::test_overflow ... FAILED

failures:

---- tests::test_edge ----
thread 'tests::test_edge' panicked at 'assertion failed: 1 == 2'

---- tests::test_overflow ----
thread 'tests::test_overflow' panicked at 'overflow in add'

test result: FAILED. 8 passed; 2 failed; 0 ignored"#;

        let f = cargo_test_filter();
        let r = run_pipeline("cargo test", input, &f);
        assert!(r.compressed.contains("FAILED"));
        assert!(r.compressed.contains("test_edge"));
        assert!(r.compressed.contains("panicked"));
        assert!(r.compressed.contains("test result:"));
        assert!(!r.compressed.contains("test_add ... ok"));
        assert!(r.ratio < 0.8, "ratio = {}", r.ratio);
    }

    #[test]
    fn pytest_keeps_failures() {
        let input = r#"============================= test session starts ==============================
platform darwin -- Python 3.11.0, pytest-7.4.0
collected 5 items

tests/test_a.py::test_one PASSED                                         [ 20%]
tests/test_a.py::test_two PASSED                                         [ 40%]
tests/test_a.py::test_three FAILED                                       [ 60%]
tests/test_a.py::test_four PASSED                                        [ 80%]
tests/test_a.py::test_five PASSED                                        [100%]

=================================== FAILURES ===================================
____________________________ test_three ______________________________
    def test_three():
>       assert 1 == 2
E       AssertionError: assert 1 == 2
=========================== short test summary info ============================
FAILED tests/test_a.py::test_three
======================== 1 failed, 4 passed in 0.05s =========================="#;

        let f = pytest_filter();
        let r = run_pipeline("pytest", input, &f);
        assert!(r.compressed.contains("FAILED"));
        assert!(r.compressed.contains("test_three"));
        assert!(r.compressed.contains("AssertionError"));
        assert!(!r.compressed.contains("PASSED"));
    }

    #[test]
    fn go_test_keeps_failures() {
        let input = "=== RUN   TestOne\n--- PASS: TestOne (0.00s)\n=== RUN   TestTwo\n--- FAIL: TestTwo (0.01s)\n    foo_test.go:10: expected 2, got 1\nFAIL\nFAIL    github.com/foo/bar    0.012s";
        let f = go_test_filter();
        let r = run_pipeline("go test ./...", input, &f);
        assert!(r.compressed.contains("--- FAIL: TestTwo"));
        assert!(r.compressed.contains("FAIL"));
        assert!(!r.compressed.contains("=== RUN"));
    }
}
