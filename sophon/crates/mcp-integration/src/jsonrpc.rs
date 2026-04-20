//! JSON-RPC 2.0 error helpers for the MCP stdio server.
//!
//! The MCP spec (2025-06-18) inherits JSON-RPC 2.0's error model. We map
//! internal failure modes to structured `{code, message, data?}` objects
//! instead of bubbling raw `anyhow::Error::to_string()` strings, so
//! clients can branch on the code rather than regex-matching English.
//!
//! ## Code ranges
//!
//! | Range              | Source                            |
//! |--------------------|-----------------------------------|
//! | -32700             | Parse error (JSON-RPC reserved)   |
//! | -32600             | Invalid Request                   |
//! | -32601             | Method not found                  |
//! | -32602             | Invalid params                    |
//! | -32603             | Internal error                    |
//! | -32000 .. -32099   | Server-defined (Sophon)           |
//!
//! Only the `-32000..-32099` range is ours to define. See
//! <https://www.jsonrpc.org/specification#error_object>.

use serde_json::{json, Value};

/// Standard JSON-RPC 2.0 error codes.
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

/// Sophon-defined server error codes (JSON-RPC 2.0 reserves -32000..-32099
/// for implementation-specific server errors). These are stable — clients
/// can match on them. New codes go at the end; never renumber existing
/// ones without a protocol version bump.
pub const SOPHON_TOOL_NOT_FOUND: i32 = -32001;
pub const SOPHON_TOOL_ARGUMENTS_INVALID: i32 = -32002;
pub const SOPHON_TOOL_EXECUTION_FAILED: i32 = -32003;
pub const SOPHON_RETRIEVER_UNAVAILABLE: i32 = -32004;
pub const SOPHON_IO_ERROR: i32 = -32005;
pub const SOPHON_CONFIG_ERROR: i32 = -32006;

/// Build a JSON-RPC error response envelope.
///
/// `id` may be `Null` when the server received a request without an `id`
/// field (a notification) — the JSON-RPC spec still requires the `id`
/// key to be present in error responses so clients can reconcile.
pub fn error_response(id: Value, code: i32, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

/// Error response with an extra `data` field (arbitrary JSON payload
/// attached by the server to help the client diagnose the failure).
pub fn error_response_with_data(
    id: Value,
    code: i32,
    message: impl Into<String>,
    data: Value,
) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
            "data": data,
        }
    })
}

/// Classify an `anyhow::Error` chain and map it to the closest Sophon
/// server error code. Falls back to `INTERNAL_ERROR` (-32603) when no
/// specific category matches.
///
/// We walk the error chain with `downcast_ref` for known types we care
/// about, otherwise inspect the display string. The chain is kept short
/// and the display strings are the existing `anyhow::anyhow!` messages
/// — this is classification, not parsing.
pub fn classify_anyhow(err: &anyhow::Error) -> i32 {
    for cause in err.chain() {
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return SOPHON_IO_ERROR;
        }
        if cause.downcast_ref::<serde_json::Error>().is_some() {
            return SOPHON_TOOL_ARGUMENTS_INVALID;
        }
    }
    let msg = err.to_string().to_ascii_lowercase();
    if msg.contains("unknown tool") || msg.contains("tool not found") {
        SOPHON_TOOL_NOT_FOUND
    } else if msg.contains("invalid")
        || msg.contains("missing")
        || msg.contains("parse")
        || msg.contains("deserialize")
    {
        SOPHON_TOOL_ARGUMENTS_INVALID
    } else if msg.contains("retriever") && msg.contains("not configured") {
        SOPHON_RETRIEVER_UNAVAILABLE
    } else if msg.contains("config") {
        SOPHON_CONFIG_ERROR
    } else {
        SOPHON_TOOL_EXECUTION_FAILED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_response_has_spec_shape() {
        let r = error_response(json!(42), METHOD_NOT_FOUND, "method not found: foo");
        assert_eq!(r["jsonrpc"], "2.0");
        assert_eq!(r["id"], 42);
        assert_eq!(r["error"]["code"], METHOD_NOT_FOUND);
        assert_eq!(r["error"]["message"], "method not found: foo");
        assert!(r["error"].get("data").is_none());
    }

    #[test]
    fn error_response_with_data_preserves_payload() {
        let r = error_response_with_data(
            Value::Null,
            INVALID_PARAMS,
            "missing field",
            json!({"field": "name"}),
        );
        assert_eq!(r["error"]["data"]["field"], "name");
    }

    #[test]
    fn classify_io_error() {
        let err: anyhow::Error =
            std::io::Error::new(std::io::ErrorKind::NotFound, "boom").into();
        assert_eq!(classify_anyhow(&err), SOPHON_IO_ERROR);
    }

    #[test]
    fn classify_unknown_tool() {
        let err = anyhow::anyhow!("unknown tool: frobnicate");
        assert_eq!(classify_anyhow(&err), SOPHON_TOOL_NOT_FOUND);
    }

    #[test]
    fn classify_missing_argument() {
        let err = anyhow::anyhow!("missing required field `query`");
        assert_eq!(classify_anyhow(&err), SOPHON_TOOL_ARGUMENTS_INVALID);
    }

    #[test]
    fn classify_generic_failure_falls_back_to_execution() {
        let err = anyhow::anyhow!("the LLM shelled out and died");
        assert_eq!(classify_anyhow(&err), SOPHON_TOOL_EXECUTION_FAILED);
    }
}
