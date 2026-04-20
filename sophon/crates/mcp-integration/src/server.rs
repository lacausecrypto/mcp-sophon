use std::collections::HashMap;

use codebase_navigator::{Navigator, NavigatorConfig};
use delta_streamer::DeltaStreamer;
use fragment_cache::FragmentCache;
use memory_manager::graph::GraphStore;
use memory_manager::MemoryManager;
use output_compressor::OutputCompressor;
use prompt_compressor::PromptCompressor;
use semantic_retriever::{Retriever, RetrieverConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{config::SophonConfig, handlers::handle_tool_call, tools::get_tool_definitions};

#[derive(Debug)]
pub struct SophonServer {
    pub prompt_compressor: PromptCompressor,
    pub memory_manager: MemoryManager,
    pub delta_streamer: DeltaStreamer,
    pub fragment_cache: FragmentCache,
    /// Always-on output compressor. Stateless; no persistence; no
    /// external deps. Exposed via the `compress_output` MCP tool.
    pub output_compressor: OutputCompressor,
    /// Always-on codebase navigator. Stateful cache of the last
    /// scanned repo lives inside. Exposed via `navigate_codebase`.
    pub codebase_navigator: Navigator,
    /// Optional semantic retriever. Activated only when the
    /// `SOPHON_RETRIEVER_PATH` environment variable points at a directory
    /// where the JSONL chunk store should live. Default off so existing
    /// `compress_history` callers see no behaviour change.
    pub retriever: Option<Retriever>,
    /// Optional graph memory (Path A architecture). Activated by
    /// `SOPHON_GRAPH_MEMORY=1`. When present, `update_memory` extracts
    /// triples from new messages at ingestion time, and
    /// `compress_history` queries the graph for relevant facts at
    /// query time (pure Rust, zero LLM calls per query).
    /// `SOPHON_GRAPH_MEMORY_PATH` selects the JSON snapshot location
    /// for persistence across restarts.
    pub graph_memory: Option<GraphStore>,
    pub stats: StatsCollector,
}

impl Default for SophonServer {
    fn default() -> Self {
        Self::new()
    }
}

impl SophonServer {
    pub fn new() -> Self {
        Self::with_config(SophonConfig::default())
    }

    pub fn with_config(cfg: SophonConfig) -> Self {
        // Opt-in persistent memory: set SOPHON_MEMORY_PATH to a JSONL file
        // path (e.g. ~/.sophon/memory/default.jsonl). When present, the
        // update_memory tool appends to that file and a later `sophon serve`
        // run resumes from it.
        let mut memory_manager = MemoryManager::new(cfg.memory);
        if let Ok(path) = std::env::var("SOPHON_MEMORY_PATH") {
            let expanded = expand_tilde(&path);
            match memory_manager.clone().with_persistence(&expanded) {
                Ok(mgr) => memory_manager = mgr,
                Err(e) => {
                    tracing::warn!(
                        path = ?expanded,
                        error = %e,
                        "could not open memory persistence file"
                    );
                }
            }
        }

        // Opt-in semantic retriever: activated by SOPHON_RETRIEVER_PATH.
        // The path points at a directory where the JSONL chunk store and
        // related metadata live. Errors are logged but don't crash startup —
        // the rest of Sophon should keep working even if the retriever is
        // misconfigured.
        let retriever = match std::env::var("SOPHON_RETRIEVER_PATH") {
            Ok(raw) => {
                let dir = expand_tilde(&raw);
                let mut rcfg = RetrieverConfig::default();
                rcfg.storage_path = dir.join("chunks.jsonl");
                // Multi-hop retrieval: activated via SOPHON_MULTIHOP=1
                rcfg.multihop = std::env::var("SOPHON_MULTIHOP")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                // Hybrid (vector + BM25 sparse-lexical, fused via RRF):
                // activated via SOPHON_HYBRID=1. Closes the rare-term /
                // out-of-vocabulary gap of the pure HashEmbedder baseline.
                rcfg.hybrid = std::env::var("SOPHON_HYBRID")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                // Chunk sizing overrides — Rec #1 from the bench audit
                // showed that 128/256-token chunks often split an answer
                // across boundaries. Doubling to 500/700 keeps related
                // sentences in one chunk and makes verbatim top-K much
                // more informative.
                if let Ok(v) = std::env::var("SOPHON_CHUNK_TARGET") {
                    if let Ok(n) = v.parse::<usize>() {
                        if n >= 32 && n <= 4000 {
                            rcfg.chunk_config.target_size = n;
                        }
                    }
                }
                if let Ok(v) = std::env::var("SOPHON_CHUNK_MAX") {
                    if let Ok(n) = v.parse::<usize>() {
                        if n >= rcfg.chunk_config.target_size && n <= 8000 {
                            rcfg.chunk_config.max_size = n;
                        }
                    }
                }

                let embedder_choice = std::env::var("SOPHON_EMBEDDER")
                    .unwrap_or_default()
                    .to_lowercase();

                let result = match embedder_choice.as_str() {
                    #[cfg(feature = "bge")]
                    "bge" => {
                        tracing::info!("Using BGE-small-en-v1.5 embedder (semantic)");
                        match semantic_retriever::BgeEmbedder::new() {
                            Ok(bge) => Retriever::with_embedder(rcfg, std::sync::Arc::new(bge)),
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "BGE embedder failed to load; falling back to HashEmbedder"
                                );
                                Retriever::open(rcfg)
                            }
                        }
                    }
                    _ => Retriever::open(rcfg),
                };

                match result {
                    Ok(r) => Some(r),
                    Err(e) => {
                        tracing::warn!(
                            path = ?dir,
                            error = %e,
                            "could not open semantic retriever"
                        );
                        None
                    }
                }
            }
            Err(_) => None,
        };

        // Opt-in graph memory (Path A). Activated by SOPHON_GRAPH_MEMORY=1.
        // Persistence path via SOPHON_GRAPH_MEMORY_PATH (JSON snapshot).
        // Falls back to in-memory-only when no path is set.
        let graph_memory = {
            let enabled = std::env::var("SOPHON_GRAPH_MEMORY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !enabled {
                None
            } else {
                match std::env::var("SOPHON_GRAPH_MEMORY_PATH") {
                    Ok(raw) => {
                        let path = expand_tilde(&raw);
                        match GraphStore::open(&path) {
                            Ok(g) => Some(g),
                            Err(e) => {
                                tracing::warn!(
                                    path = ?path,
                                    error = %e,
                                    "could not open graph memory; using in-memory only"
                                );
                                Some(GraphStore::new())
                            }
                        }
                    }
                    Err(_) => Some(GraphStore::new()),
                }
            }
        };

        Self {
            prompt_compressor: PromptCompressor::with_config(cfg.prompt),
            memory_manager,
            delta_streamer: DeltaStreamer::new(cfg.delta.max_files),
            fragment_cache: FragmentCache::with_config(cfg.fragment),
            output_compressor: OutputCompressor::default(),
            codebase_navigator: Navigator::new(NavigatorConfig::default()),
            retriever,
            graph_memory,
            stats: StatsCollector::default(),
        }
    }

    pub fn handle_tool_call(&mut self, name: &str, arguments: Value) -> anyhow::Result<Value> {
        handle_tool_call(self, name, arguments)
    }

    /// Handle one JSON-RPC message for MCP stdio transport.
    ///
    /// Never returns `Err` — every failure mode is reported as a proper
    /// JSON-RPC error response so the stdio loop is not killed by a
    /// single malformed request. Returns `Ok(None)` only for incoming
    /// notifications (no `id`) that require no reply.
    ///
    /// MCP protocol version advertised: `2025-06-18`. Cancellations are
    /// acknowledged but not propagated — tool execution is synchronous
    /// inside `handle_tool_call`, so there is no mid-flight interrupt
    /// hook yet (see tracking note in the `notifications/cancelled`
    /// arm).
    pub fn handle_json_rpc_message(&mut self, message: &Value) -> anyhow::Result<Option<Value>> {
        use crate::jsonrpc::{
            classify_anyhow, error_response, INVALID_PARAMS, METHOD_NOT_FOUND,
            SOPHON_TOOL_NOT_FOUND,
        };

        let method = message
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or_default();
        let raw_id = message.get("id").cloned();
        let id = raw_id.clone().unwrap_or(Value::Null);
        let is_notification = raw_id.is_none();
        let params = message.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {
                                "listChanged": false
                            },
                            "logging": {}
                        },
                        "serverInfo": {
                            "name": "sophon",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
                });
                Ok(Some(response))
            }
            "notifications/initialized" => Ok(None),
            "notifications/cancelled" => {
                // MCP 2025-06-18 cancellation: client asks the server to
                // stop processing a previously-issued request. Our tool
                // dispatch is synchronous per message on the stdio loop,
                // so by the time this notification is parsed, the
                // targeted request has either finished or is in flight
                // on the same thread with no interrupt point. We log and
                // ACK — restoring the thread for truly async tool
                // execution is out of scope for the initial 2025-06-18
                // bump.
                let req_id = params.get("requestId").cloned().unwrap_or(Value::Null);
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no reason)");
                tracing::debug!(
                    request_id = %req_id,
                    reason = %reason,
                    "received cancellation notification (no-op: tool dispatch is synchronous)"
                );
                Ok(None)
            }
            "ping" => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {}
                });
                Ok(Some(response))
            }
            "tools/list" => {
                let tools = get_tool_definitions()
                    .into_iter()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.input_schema,
                        })
                    })
                    .collect::<Vec<_>>();

                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": tools
                    }
                });
                Ok(Some(response))
            }
            "tools/call" => {
                let Some(name) = params.get("name").and_then(|v| v.as_str()) else {
                    return Ok(Some(error_response(
                        id,
                        INVALID_PARAMS,
                        "tools/call missing params.name",
                    )));
                };
                // Reject unknown tool names as a proper JSON-RPC error so
                // clients can branch on the code (rather than parsing a
                // free-form "unknown tool: frobnicate" string).
                if !get_tool_definitions().iter().any(|t| t.name == name) {
                    return Ok(Some(error_response(
                        id,
                        SOPHON_TOOL_NOT_FOUND,
                        format!("unknown tool: {name}"),
                    )));
                }
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));

                match self.handle_tool_call(name, arguments) {
                    Ok(result) => {
                        let result_text = serde_json::to_string(&result)
                            .unwrap_or_else(|_| "{}".to_string());
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    {"type": "text", "text": result_text}
                                ],
                                "structuredContent": result,
                                "isError": false
                            }
                        });
                        Ok(Some(response))
                    }
                    Err(err) => {
                        let code = classify_anyhow(&err);
                        let msg = err.to_string();
                        tracing::warn!(tool = %name, code, error = %msg, "tool execution failed");
                        // MCP tool-level error: returned inside `result`
                        // with `isError: true`, NOT as a JSON-RPC error.
                        // `structuredContent.error` mirrors the text
                        // message with a machine-readable Sophon code so
                        // clients can branch without regex.
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    {"type": "text", "text": &msg}
                                ],
                                "structuredContent": {
                                    "error": {
                                        "code": code,
                                        "message": &msg,
                                    }
                                },
                                "isError": true
                            }
                        });
                        Ok(Some(response))
                    }
                }
            }
            _ => {
                // Backward-compatible legacy mode:
                // {"tool":"name","arguments":{...}}
                if let Some(tool) = message.get("tool").and_then(|v| v.as_str()) {
                    let args = message.get("arguments").cloned().unwrap_or(Value::Null);
                    let legacy = match self.handle_tool_call(tool, args) {
                        Ok(result) => json!({"ok": true, "result": result}),
                        Err(err) => {
                            let code = classify_anyhow(&err);
                            json!({
                                "ok": false,
                                "error": {
                                    "code": code,
                                    "message": err.to_string(),
                                }
                            })
                        }
                    };
                    return Ok(Some(legacy));
                }

                if is_notification {
                    tracing::debug!(method = %method, "ignoring unknown notification");
                    Ok(None)
                } else {
                    Ok(Some(error_response(
                        id,
                        METHOD_NOT_FOUND,
                        format!("method not found: {}", method),
                    )))
                }
            }
        }
    }

    pub async fn run_stdio(mut self) -> anyhow::Result<()> {
        use crate::jsonrpc::{error_response, PARSE_ERROR};
        use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut lines = BufReader::new(stdin).lines();
        let mut out = io::BufWriter::new(stdout);

        // MCP JSON-RPC over stdio (line-delimited JSON objects).
        while let Some(line) = lines.next_line().await? {
            let parsed: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(err) => {
                    let error = error_response(
                        Value::Null,
                        PARSE_ERROR,
                        format!("parse error: {}", err),
                    );
                    let serialized = serde_json::to_string(&error).unwrap_or_else(|_| {
                        String::from(
                            r#"{"jsonrpc":"2.0","error":{"code":-32700,"message":"parse error"}}"#,
                        )
                    });
                    out.write_all(format!("{}\n", serialized).as_bytes())
                        .await?;
                    out.flush().await?;
                    continue;
                }
            };

            // handle_json_rpc_message is infallible-by-design: every error
            // path becomes a JSON-RPC error response. The `?` below only
            // fires on unexpected panic-adjacent conditions; keep it so
            // bugs surface instead of being silently swallowed.
            if let Some(response) = self.handle_json_rpc_message(&parsed)? {
                let serialized = serde_json::to_string(&response).unwrap_or_default();
                out.write_all(format!("{}\n", serialized).as_bytes())
                    .await?;
                out.flush().await?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StatsCollector {
    per_module: HashMap<String, ModuleStats>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModuleStats {
    pub calls: u64,
    pub original_tokens: u64,
    pub compressed_tokens: u64,
}

fn expand_tilde(path: &str) -> std::path::PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(rest);
        }
    }
    std::path::PathBuf::from(path)
}

impl StatsCollector {
    pub fn record(&mut self, module: &str, original_tokens: usize, compressed_tokens: usize) {
        let entry = self
            .per_module
            .entry(module.to_string())
            .or_insert_with(ModuleStats::default);
        entry.calls += 1;
        entry.original_tokens += original_tokens as u64;
        entry.compressed_tokens += compressed_tokens as u64;
    }

    pub fn snapshot(&self) -> HashMap<String, ModuleStats> {
        self.per_module.clone()
    }

    pub fn totals(&self) -> ModuleStats {
        self.per_module
            .values()
            .fold(ModuleStats::default(), |mut acc, st| {
                acc.calls += st.calls;
                acc.original_tokens += st.original_tokens;
                acc.compressed_tokens += st.compressed_tokens;
                acc
            })
    }
}
