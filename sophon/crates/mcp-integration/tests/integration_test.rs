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
    assert!(tools.iter().any(|t| t.get("name") == Some(&json!("compress_prompt"))));

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
