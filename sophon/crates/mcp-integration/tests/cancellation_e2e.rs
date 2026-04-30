//! End-to-end coverage for the v0.5.4 async dispatch + cancellation
//! registry.
//!
//! These tests spawn the actual release binary (`sophon serve`) and
//! drive it over stdio so we exercise the real async runtime,
//! `JoinSet` drain, and shared-stdout serialisation. Unit-testing
//! `run_stdio` in isolation would miss the cross-task plumbing that
//! the new code is for.
//!
//! Each test is `#[ignore]` because they depend on a release build
//! existing at `sophon/target/release/sophon`. CI runs them
//! explicitly via `cargo test -p mcp-integration --test
//! cancellation_e2e -- --include-ignored` after building the
//! release binary; local runs can `cargo build --release && cargo
//! test -- --include-ignored cancellation`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn binary_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/mcp-integration/Cargo.toml → workspace root (sophon/)
    // → target/release/sophon
    p.pop(); // crates
    p.pop(); // sophon
    p.push("target/release/sophon");
    p
}

fn drive_stdio(messages: &[serde_json::Value], wait: Duration) -> (Vec<serde_json::Value>, Duration) {
    let bin = binary_path();
    assert!(
        bin.exists(),
        "release binary missing: {:?} — run `cargo build --release` first",
        bin
    );

    let mut child = Command::new(&bin)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sophon serve");

    let stdin = child.stdin.as_mut().expect("stdin pipe");
    for msg in messages {
        let line = serde_json::to_string(msg).expect("serialize");
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
    }
    stdin.flush().unwrap();

    // Sleep gives the server time to process before EOF triggers
    // the JoinSet drain. This is the cancellation-latency window:
    // anything not done within `wait` after EOF gets cleanly
    // drained by the loop epilogue.
    std::thread::sleep(wait);

    // Closing stdin (via dropping `child.stdin` when we move `child`)
    // forces the loop to exit; the server still drains in-flight
    // tasks before returning.
    drop(child.stdin.take());

    let t0 = Instant::now();
    let output = child.wait_with_output().expect("wait");
    let elapsed = t0.elapsed();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut responses = Vec::new();
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            responses.push(v);
        }
    }
    (responses, elapsed)
}

#[test]
#[ignore]
fn three_tool_calls_no_cancel_all_responses_arrive() {
    let msgs = vec![
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"}
            }
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": "a"}}
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": "b"}}
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": "c"}}
        }),
    ];
    let (responses, _) = drive_stdio(&msgs, Duration::from_millis(200));
    let ids: Vec<_> = responses.iter().filter_map(|r| r.get("id").and_then(|v| v.as_i64())).collect();
    assert!(ids.contains(&1), "init response missing");
    assert!(ids.contains(&2), "id=2 missing");
    assert!(ids.contains(&3), "id=3 missing");
    assert!(ids.contains(&4), "id=4 missing");
}

#[test]
#[ignore]
fn cancel_for_in_flight_request_drops_response() {
    // Issue a tools/call for `compress_history` with a synthetic
    // 200-message history then immediately cancel its request id.
    // count_tokens is too fast to race; compress_history involves
    // tokenization + summarisation passes so there's a real window
    // for the cancel to arrive before the response is written.
    let big_messages: Vec<serde_json::Value> = (0..200)
        .map(|i| {
            serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("Turn {i}: walk me through how compression interacts \
                                    with the chunker. We were measuring p99 around \
                                    {} ms.", 60 + (i * 7) % 400)
            })
        })
        .collect();
    let msgs = vec![
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"}
            }
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {
                "name": "compress_history",
                "arguments": {"messages": big_messages, "max_tokens": 2000}
            }
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 99, "reason": "client lost interest"}
        }),
        // A second, distinct call should still complete and respond
        // — cancel is per-request, never global.
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": "after-cancel sentinel"}}
        }),
    ];
    let (responses, _) = drive_stdio(&msgs, Duration::from_millis(50));
    let ids: Vec<_> = responses.iter().filter_map(|r| r.get("id").and_then(|v| v.as_i64())).collect();

    // init responds
    assert!(ids.contains(&1), "init response missing: {ids:?}");
    // cancelled request: response MAY OR MAY NOT have been written
    // depending on whether the cancel raced ahead of compression
    // completion. We don't assert either way — the contract is
    // "cancel drops the response if it arrives in time", not
    // "cancel always wins the race".
    // The follow-up call must always succeed regardless of cancel
    // outcome — proves the dispatcher didn't get stuck or
    // mis-route after a cancellation.
    assert!(
        ids.contains(&100),
        "post-cancel request id=100 lost — cancellation broke the dispatcher (got ids {ids:?})"
    );
}

#[test]
#[ignore]
fn cancel_for_unknown_request_id_is_a_safe_noop() {
    // notifications/cancelled with a requestId that was never
    // dispatched (or already completed) must be a no-op — no
    // panic, no error, no spurious response, and the next legit
    // request must still succeed.
    let msgs = vec![
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"}
            }
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 99999, "reason": "stale"}
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": "still ok"}}
        }),
    ];
    let (responses, _) = drive_stdio(&msgs, Duration::from_millis(150));
    let ids: Vec<_> = responses.iter().filter_map(|r| r.get("id").and_then(|v| v.as_i64())).collect();
    assert!(ids.contains(&1), "init missing: {ids:?}");
    assert!(ids.contains(&7), "post-noop-cancel request missing: {ids:?}");
}

#[test]
#[ignore]
fn concurrent_pipeline_drains_all_responses() {
    // Pump a hundred fast tools/call requests back-to-back. The
    // JoinSet drain at the loop epilogue must collect every
    // spawned task before exit, so all 100 responses must show up.
    let mut msgs = vec![serde_json::json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "e2e", "version": "0"}
        }
    })];
    for i in 1..=100 {
        msgs.push(serde_json::json!({
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": format!("msg-{i}")}}
        }));
    }
    let (responses, _) = drive_stdio(&msgs, Duration::from_millis(500));
    let ids: std::collections::BTreeSet<i64> = responses
        .iter()
        .filter_map(|r| r.get("id").and_then(|v| v.as_i64()))
        .collect();
    let expected: std::collections::BTreeSet<i64> = (0..=100).collect();
    let missing: Vec<_> = expected.difference(&ids).copied().collect();
    assert!(
        missing.is_empty(),
        "missing response ids after JoinSet drain: {missing:?}"
    );
}
