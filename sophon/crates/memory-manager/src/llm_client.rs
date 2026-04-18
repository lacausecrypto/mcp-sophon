//! Shared LLM shell-out used by the block-based summariser and the multi-hop
//! query decomposer. Command is configurable via `SOPHON_LLM_CMD` (default
//! `claude -p --model haiku`). Returns `None` on any failure so callers can
//! fall back to heuristics without panicking.

use std::io::Write;
use std::process::{Command, Stdio};

/// Default LLM command. Same default as the original inline summariser so
/// existing deployments keep working.
pub const DEFAULT_LLM_CMD: &str = "claude -p --model haiku";

/// Execute an LLM call with `prompt` on stdin. Returns the trimmed stdout as
/// a `String`, unwrapping `{"result": "..."}` JSON envelopes when present.
/// Returns `None` if the command, spawn, or wait fails, or if stdout is empty.
///
/// When `SOPHON_DEBUG_LLM=1` is set, all failure paths (spawn fail, non-zero
/// exit, empty stdout, JSON parse fail) emit a diagnostic line to stderr
/// with a tag, status code, and first 200 bytes of the child's stderr. This
/// is how we audit silent failures during benches — rate-limit cascades,
/// broken CLI, network hiccups all show up here instead of being masked as
/// "fallback to heuristic".
pub fn call_llm(prompt: &str) -> Option<String> {
    let debug = std::env::var("SOPHON_DEBUG_LLM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let cmd_str = std::env::var("SOPHON_LLM_CMD").unwrap_or_else(|_| DEFAULT_LLM_CMD.to_string());
    let parts: Vec<&str> = cmd_str.split_whitespace().collect();
    if parts.is_empty() {
        if debug {
            eprintln!("[sophon-llm] FAIL: empty SOPHON_LLM_CMD");
        }
        return None;
    }

    let child = match Command::new(parts[0])
        .args(&parts[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if debug { Stdio::piped() } else { Stdio::null() })
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            if debug {
                eprintln!("[sophon-llm] FAIL: spawn {:?}: {}", parts[0], e);
            }
            return None;
        }
    };

    let output = {
        let mut child = child;
        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(prompt.as_bytes());
        }
        match child.wait_with_output() {
            Ok(o) => o,
            Err(e) => {
                if debug {
                    eprintln!("[sophon-llm] FAIL: wait_with_output: {}", e);
                }
                return None;
            }
        }
    };

    if !output.status.success() {
        if debug {
            let stderr_head: String = String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(200)
                .collect();
            eprintln!(
                "[sophon-llm] FAIL: exit={:?} stderr={:?}",
                output.status.code(),
                stderr_head
            );
        }
        return None;
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        if debug {
            let stderr_head: String = String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(200)
                .collect();
            eprintln!("[sophon-llm] FAIL: empty stdout, stderr={:?}", stderr_head);
        }
        return None;
    }

    if raw.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
                return Some(result.to_string());
            }
            if debug {
                eprintln!(
                    "[sophon-llm] WARN: JSON without 'result' key, raw head: {:?}",
                    raw.chars().take(120).collect::<String>()
                );
            }
        } else if debug {
            eprintln!(
                "[sophon-llm] WARN: non-JSON starting with '{{', raw head: {:?}",
                raw.chars().take(120).collect::<String>()
            );
        }
    }

    Some(raw)
}

/// Whether an LLM command is available — used by callers to pick between
/// the LLM path and the heuristic fallback without spawning a process.
pub fn llm_cmd_is_configured() -> bool {
    std::env::var("SOPHON_LLM_CMD").is_ok()
}
