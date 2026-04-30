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
    /// MCP protocol version advertised: `2025-06-18`. As of v0.5.4,
    /// `tools/call` requests dispatched through `run_stdio` are run on
    /// a dedicated `spawn_blocking` task; `notifications/cancelled` is
    /// intercepted by `run_stdio` and triggers a `CancellationToken`
    /// that drops the in-flight response if it arrives before the
    /// tool finishes. The in-flight CPU work itself is **not yet**
    /// interrupted mid-flight — cooperative interrupt hooks for the
    /// long ops (`compress_history`, `retrieve`, …) are deferred to
    /// v0.5.5.
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
                        let result_text =
                            serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());
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

    pub async fn run_stdio(self) -> anyhow::Result<()> {
        use crate::jsonrpc::{error_response, PARSE_ERROR};
        use std::sync::Arc;
        use tokio::io::{self, AsyncBufReadExt, BufReader};
        use tokio::sync::Mutex;
        use tokio_util::sync::CancellationToken;

        // Shared state under tokio::sync::Mutex so the server can be
        // borrowed by both the inline path (initialize, ping,
        // tools/list, notifications/cancelled) and the spawned tool
        // tasks. Tools acquire the lock just for the duration of their
        // sync `handle_tool_call` body — readers and other inline
        // requests are blocked while a tool runs (intentional for now;
        // v0.5.5 will split RO/RW dispatch to allow concurrent reads).
        let server = Arc::new(Mutex::new(self));

        // Cancellation registry: maps an in-flight tools/call request_id
        // to its CancellationToken. notifications/cancelled looks up by
        // id and triggers cancel; the spawned task removes itself when
        // it finishes (or is cancelled).
        let pending: Arc<Mutex<HashMap<String, CancellationToken>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Stdout is shared across spawned tool tasks. Wrap in a Mutex
        // so concurrent tasks can't interleave their JSON-RPC responses
        // mid-line. Buffer flush happens per-write to keep responses
        // visible immediately to the client.
        let out: Arc<Mutex<io::BufWriter<io::Stdout>>> =
            Arc::new(Mutex::new(io::BufWriter::new(io::stdout())));

        let stdin = io::stdin();
        let mut lines = BufReader::new(stdin).lines();
        // Track every spawned tool task so we can drain them on
        // stdin EOF — without this, a client that closes the pipe
        // immediately after issuing a request would race the
        // spawn/response and miss the reply.
        let mut tool_tasks: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();

        while let Some(line) = lines.next_line().await? {
            let parsed: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(err) => {
                    let error =
                        error_response(Value::Null, PARSE_ERROR, format!("parse error: {}", err));
                    write_response(&out, &error).await?;
                    continue;
                }
            };

            let method = parsed
                .get("method")
                .and_then(|m| m.as_str())
                .unwrap_or_default();

            // Hot path: tools/call gets dispatched as a spawned task so
            // a long-running tool can be cancelled by a subsequent
            // `notifications/cancelled`. Everything else is cheap and
            // stays inline on the loop thread to keep dispatch latency
            // low.
            if method == "tools/call" {
                let id = parsed.get("id").cloned().unwrap_or(Value::Null);
                let id_key = canonical_id_key(&id);
                let token = CancellationToken::new();
                pending.lock().await.insert(id_key.clone(), token.clone());

                let server_for_task = Arc::clone(&server);
                let pending_for_task = Arc::clone(&pending);
                let out_for_task = Arc::clone(&out);
                let parsed_for_task = parsed.clone();

                tool_tasks.spawn(async move {
                    let work = {
                        let server = Arc::clone(&server_for_task);
                        let parsed = parsed_for_task.clone();
                        tokio::task::spawn_blocking(move || {
                            // blocking_lock is OK inside spawn_blocking —
                            // we're on a dedicated blocking-IO thread,
                            // not on a runtime worker.
                            let mut srv = server.blocking_lock();
                            srv.handle_json_rpc_message(&parsed)
                        })
                    };

                    let resp_opt: Option<Value> = tokio::select! {
                        biased;
                        // Cancellation wins ties so a fast-arriving cancel
                        // for a fast-completing tool is still honoured.
                        _ = token.cancelled() => {
                            tracing::debug!(
                                request_id = %id,
                                "tools/call cancelled before completion — response dropped"
                            );
                            None
                        }
                        result = work => match result {
                            Ok(Ok(opt)) => opt,
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    request_id = %id,
                                    error = %e,
                                    "tools/call dispatcher returned Err — surfacing as JSON-RPC error"
                                );
                                Some(error_response(
                                    id.clone(),
                                    crate::jsonrpc::INTERNAL_ERROR,
                                    format!("internal error: {}", e),
                                ))
                            }
                            Err(join_err) => {
                                tracing::error!(
                                    request_id = %id,
                                    error = %join_err,
                                    "tools/call task panicked"
                                );
                                Some(error_response(
                                    id.clone(),
                                    crate::jsonrpc::INTERNAL_ERROR,
                                    "internal error: tool task panicked".to_string(),
                                ))
                            }
                        }
                    };

                    pending_for_task.lock().await.remove(&id_key);

                    if let Some(response) = resp_opt {
                        if let Err(e) = write_response(&out_for_task, &response).await {
                            tracing::error!(error = %e, "failed to write tool response");
                        }
                    }
                });
                continue;
            }

            // Inline path: initialize / ping / tools/list / notifications/*.
            // notifications/cancelled gets special handling — it must
            // look up the cancellation token and trigger it before
            // returning.
            if method == "notifications/cancelled" {
                let target_id = parsed
                    .get("params")
                    .and_then(|p| p.get("requestId"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let target_key = canonical_id_key(&target_id);
                let cancelled = {
                    let mut map = pending.lock().await;
                    if let Some(token) = map.remove(&target_key) {
                        token.cancel();
                        true
                    } else {
                        false
                    }
                };
                tracing::debug!(
                    request_id = %target_id,
                    cancelled,
                    "notifications/cancelled processed"
                );
                continue;
            }

            // Default inline dispatch.
            let response_opt = {
                let mut srv = server.lock().await;
                srv.handle_json_rpc_message(&parsed)?
            };
            if let Some(response) = response_opt {
                write_response(&out, &response).await?;
            }
        }

        // Drain in-flight tool tasks before we let stdout drop. A
        // client that closes the pipe immediately after issuing a
        // request would otherwise race the spawn/response and miss
        // the reply.
        while let Some(joined) = tool_tasks.join_next().await {
            if let Err(e) = joined {
                tracing::warn!(error = %e, "tool task did not complete cleanly");
            }
        }

        Ok(())
    }
}

/// Stable string key for a JSON-RPC `id` value. The MCP spec allows
/// id to be a number, string, or null; we want all three to dedup
/// correctly in the cancellation registry.
fn canonical_id_key(id: &Value) -> String {
    match id {
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("s:{s}"),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_else(|_| String::from("?")),
    }
}

async fn write_response(
    out: &std::sync::Arc<tokio::sync::Mutex<tokio::io::BufWriter<tokio::io::Stdout>>>,
    response: &Value,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let serialized = serde_json::to_string(response).unwrap_or_default();
    let mut guard = out.lock().await;
    guard
        .write_all(format!("{serialized}\n").as_bytes())
        .await?;
    guard.flush().await?;
    Ok(())
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
