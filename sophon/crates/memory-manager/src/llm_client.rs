//! Shared LLM shell-out used by the block-based summariser and the multi-hop
//! query decomposer. Command is configurable via `SOPHON_LLM_CMD` (default
//! `claude -p --model haiku`). Returns `None` on any failure so callers can
//! fall back to heuristics without panicking.

use std::io::Write;
use std::process::{Command, Stdio};

use tracing::{debug, warn};

/// Default LLM command. Same default as the original inline summariser so
/// existing deployments keep working.
pub const DEFAULT_LLM_CMD: &str = "claude -p --model haiku";

/// Execute an LLM call with `prompt` on stdin. Returns the trimmed stdout as
/// a `String`, unwrapping `{"result": "..."}` JSON envelopes when present.
/// Returns `None` if the command, spawn, or wait fails, or if stdout is empty.
///
/// Failure paths (spawn fail, non-zero exit, empty stdout, JSON parse fail)
/// emit tracing records at WARN (failures) or DEBUG (parse quirks) level —
/// control visibility via `RUST_LOG=memory_manager::llm_client=debug`.
/// `SOPHON_DEBUG_LLM=1` remains supported as a legacy gate that also routes
/// the child's stderr into the captured diagnostics (off by default, cheaper).
pub fn call_llm(prompt: &str) -> Option<String> {
    let capture_stderr = std::env::var("SOPHON_DEBUG_LLM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let cmd_str = std::env::var("SOPHON_LLM_CMD").unwrap_or_else(|_| DEFAULT_LLM_CMD.to_string());
    let parts: Vec<&str> = cmd_str.split_whitespace().collect();
    if parts.is_empty() {
        warn!("empty SOPHON_LLM_CMD");
        return None;
    }

    let child = match Command::new(parts[0])
        .args(&parts[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if capture_stderr {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(program = ?parts[0], error = %e, "spawn failed");
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
                warn!(error = %e, "wait_with_output failed");
                return None;
            }
        }
    };

    if !output.status.success() {
        let stderr_head: String = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(200)
            .collect();
        warn!(exit = ?output.status.code(), stderr = %stderr_head, "LLM exited non-zero");
        return None;
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        let stderr_head: String = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(200)
            .collect();
        warn!(stderr = %stderr_head, "LLM emitted empty stdout");
        return None;
    }

    if raw.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
                return Some(result.to_string());
            }
            debug!(head = %raw.chars().take(120).collect::<String>(), "JSON without 'result' key");
        } else {
            debug!(head = %raw.chars().take(120).collect::<String>(), "non-JSON starting with '{{'");
        }
    }

    Some(raw)
}

/// Whether an LLM command is available — used by callers to pick between
/// the LLM path and the heuristic fallback without spawning a process.
pub fn llm_cmd_is_configured() -> bool {
    std::env::var("SOPHON_LLM_CMD").is_ok()
}
