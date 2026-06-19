//! End-to-end regression for the ANSI pre-strip (plan T1.1).
//!
//! Before the fix, colored tool output defeated the anchored per-filter
//! regexes (`^test … ok$`), so `compress_output` degenerated to a no-op
//! (ratio ~0.98) on the exact output users actually pipe in. This test
//! feeds a *colored* `cargo test` run through the full facade and asserts
//! the filter now fires — the only thing the strip is supposed to unlock.

use output_compressor::OutputCompressor;

/// Build a realistic colored `cargo test` output: every status token is
/// wrapped in SGR color the way a TTY-attached `cargo test` emits it.
fn colored_cargo_test(n_pass: usize) -> String {
    let green = "\x1b[32m";
    let reset = "\x1b[0m";
    let mut s = String::new();
    s.push_str(&format!(
        "{green}    Finished{reset} test [unoptimized + debuginfo] target(s)\n"
    ));
    s.push_str(&format!(
        "{green}     Running{reset} unittests src/lib.rs\n\n"
    ));
    s.push_str(&format!("running {n_pass} tests\n"));
    for i in 0..n_pass {
        s.push_str(&format!("test module::case_{i} ... {green}ok{reset}\n"));
    }
    s.push_str(&format!(
        "\ntest result: {green}ok{reset}. {n_pass} passed; 0 failed; 0 ignored\n"
    ));
    s
}

#[test]
fn colored_cargo_test_is_actually_compressed() {
    let compressor = OutputCompressor::default();
    let colored = colored_cargo_test(40);

    let result = compressor.compress("cargo test", &colored);

    // The strip must have run and the cargo filter must have fired.
    assert!(
        result.strategies_applied.iter().any(|s| s == "strip_ansi"),
        "strip_ansi should run on colored output, got {:?}",
        result.strategies_applied
    );

    // The whole point: a passing run collapses hard instead of passing
    // through. Pre-fix this sat at ~0.98 (no-op). We assert a generous
    // ceiling so the test tracks "the filter fires" not an exact number.
    assert!(
        result.ratio < 0.5,
        "colored passing cargo test should compress well, got ratio {:.3} \
         (compressed {} / original {} tokens)",
        result.ratio,
        result.compressed_tokens,
        result.original_tokens,
    );
}

#[test]
fn plain_output_unaffected_by_strip() {
    // Plain (already-stripped) output must not gain a spurious strip_ansi
    // step and must compress at least as well as before.
    let compressor = OutputCompressor::default();
    let plain = "running 3 tests\ntest a ... ok\ntest b ... ok\ntest c ... ok\n\
                 \ntest result: ok. 3 passed; 0 failed; 0 ignored\n";

    let result = compressor.compress("cargo test", plain);

    assert!(
        !result.strategies_applied.iter().any(|s| s == "strip_ansi"),
        "plain output should not record a strip_ansi step"
    );
    assert!(result.ratio <= 1.0);
}
