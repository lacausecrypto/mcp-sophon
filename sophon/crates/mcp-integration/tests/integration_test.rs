use mcp_integration::SophonServer;
use serde_json::json;

#[test]
fn test_tool_definitions_present() {
    let tools = mcp_integration::get_tool_definitions();
    assert!(tools.iter().any(|t| t.name == "compress_prompt"));
    assert!(tools.iter().any(|t| t.name == "compress_history"));
    assert!(tools.iter().any(|t| t.name == "update_memory"));
    assert!(tools.iter().any(|t| t.name == "read_file_delta"));
}

#[test]
fn test_update_memory_accumulates_across_calls() {
    let mut server = SophonServer::new();

    // First call: append two messages, no snapshot
    let r1 = server
        .handle_tool_call(
            "update_memory",
            json!({
                "messages": [
                    {"role": "user", "content": "I am building a Rust compression toolkit."},
                    {"role": "assistant", "content": "Great, what modules do you need?"}
                ],
                "return_snapshot": false
            }),
        )
        .expect("append should succeed");
    assert_eq!(r1["history_len"], 2);

    // Second call: append one more, ask for snapshot
    let r2 = server
        .handle_tool_call(
            "update_memory",
            json!({
                "messages": [
                    {"role": "user", "content": "The memory module needs stateful accumulation."}
                ]
            }),
        )
        .expect("snapshot should succeed");
    assert_eq!(r2["history_len"], 3);
    assert_eq!(r2["original_message_count"], 3);
    assert!(r2["summary"].as_str().is_some());

    // Third call: reset + snapshot of empty history
    let r3 = server
        .handle_tool_call("update_memory", json!({"reset": true}))
        .expect("reset should succeed");
    assert_eq!(r3["history_len"], 0);
    assert_eq!(r3["original_message_count"], 0);
}

#[test]
fn test_compress_prompt_handler() {
    let mut server = SophonServer::new();

    let result = server
        .handle_tool_call(
            "compress_prompt",
            json!({
                "prompt": "<core_identity>You are an assistant.</core_identity><code_formatting>Use code blocks.</code_formatting>",
                "query": "Write Python code",
                "max_tokens": 200
            }),
        )
        .expect("tool call should succeed");

    assert!(result.get("compressed_prompt").is_some());
}

#[test]
fn test_mcp_jsonrpc_tools_list_and_call() {
    let mut server = SophonServer::new();

    let init = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 1,
            "method":"initialize",
            "params":{}
        }))
        .expect("initialize should be handled")
        .expect("initialize should return response");
    assert!(init.get("result").is_some());

    let list = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 2,
            "method":"tools/list",
            "params":{}
        }))
        .expect("tools/list should be handled")
        .expect("tools/list should return response");
    let tools = list
        .get("result")
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array())
        .expect("tools array");
    assert!(tools
        .iter()
        .any(|t| t.get("name") == Some(&json!("compress_prompt"))));

    let call = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 3,
            "method":"tools/call",
            "params":{
                "name":"compress_prompt",
                "arguments":{
                    "prompt":"<core_identity>You are an assistant.</core_identity><code_formatting>Use fenced code blocks.</code_formatting>",
                    "query":"Write Python code"
                }
            }
        }))
        .expect("tools/call should be handled")
        .expect("tools/call should return response");

    assert_eq!(call.get("jsonrpc"), Some(&json!("2.0")));
    assert_eq!(
        call.get("result")
            .and_then(|r| r.get("isError"))
            .and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[test]
fn initialize_advertises_2025_06_18_protocol_version() {
    let mut server = SophonServer::new();
    let init = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 1,
            "method":"initialize",
            "params":{}
        }))
        .expect("initialize should be handled")
        .expect("initialize should return response");
    assert_eq!(
        init.pointer("/result/protocolVersion"),
        Some(&json!("2025-06-18")),
        "server must advertise MCP 2025-06-18"
    );
    // Capabilities declared for 2025-06-18 tools+logging.
    assert!(init.pointer("/result/capabilities/tools").is_some());
    assert!(init.pointer("/result/capabilities/logging").is_some());
}

#[test]
fn unknown_tool_returns_structured_error_code() {
    // Per jsonrpc::SOPHON_TOOL_NOT_FOUND = -32001. The response MUST be
    // a JSON-RPC error (not a tool result with isError) so clients can
    // branch on the code without regex-matching the message text.
    let mut server = SophonServer::new();
    let resp = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 42,
            "method":"tools/call",
            "params":{
                "name":"this_tool_does_not_exist",
                "arguments":{}
            }
        }))
        .expect("dispatch should be infallible")
        .expect("error response should be produced");
    assert_eq!(resp.get("jsonrpc"), Some(&json!("2.0")));
    assert_eq!(resp.get("id"), Some(&json!(42)));
    assert_eq!(resp.pointer("/error/code"), Some(&json!(-32001)));
    assert!(resp
        .pointer("/error/message")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("this_tool_does_not_exist"));
    assert!(resp.get("result").is_none());
}

#[test]
fn tools_call_missing_name_returns_invalid_params() {
    // -32602 INVALID_PARAMS. Previously this bubbled up through `?` and
    // could kill the stdio loop; now it's reported as a spec-shaped
    // error response.
    let mut server = SophonServer::new();
    let resp = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 7,
            "method":"tools/call",
            "params":{}
        }))
        .expect("dispatch should be infallible")
        .expect("error response should be produced");
    assert_eq!(resp.pointer("/error/code"), Some(&json!(-32602)));
}

#[test]
fn notifications_cancelled_is_acknowledged_silently() {
    // Cancellations are no-op for the synchronous stdio dispatcher, but
    // they must not produce a JSON-RPC response (it's a notification,
    // no id) or be treated as an unknown method.
    let mut server = SophonServer::new();
    let resp = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "method":"notifications/cancelled",
            "params":{"requestId": 3, "reason": "user aborted"}
        }))
        .expect("dispatch should be infallible");
    assert!(
        resp.is_none(),
        "cancellation must not produce a response (it's a notification)"
    );
}

#[test]
fn tool_error_produces_is_error_with_structured_code() {
    // When a tool executes but returns an error, MCP requires the
    // response to be a `result` with `isError: true`. Sophon adds a
    // machine-readable error code in `structuredContent.error.code`.
    let mut server = SophonServer::new();
    // read_file_delta on a nonexistent path is the simplest forced
    // failure; the tool exists and accepts the arguments, then fails
    // during execution.
    let resp = server
        .handle_json_rpc_message(&json!({
            "jsonrpc":"2.0",
            "id": 9,
            "method":"tools/call",
            "params":{
                "name":"read_file_delta",
                "arguments":{"path":"/tmp/sophon_definitely_does_not_exist_xyz123"}
            }
        }))
        .expect("dispatch should be infallible")
        .expect("response should be produced");
    assert_eq!(resp.pointer("/result/isError"), Some(&json!(true)));
    let code = resp
        .pointer("/result/structuredContent/error/code")
        .and_then(|v| v.as_i64())
        .expect("structured error code should be present");
    // Must fall in the Sophon server-defined range (-32000..-32099).
    assert!(
        (-32099..=-32000).contains(&(code as i32)),
        "expected Sophon error code, got {code}"
    );
}
