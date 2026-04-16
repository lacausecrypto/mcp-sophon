use std::collections::HashMap;

use codebase_navigator::{Navigator, NavigatorConfig};
use delta_streamer::DeltaStreamer;
use fragment_cache::FragmentCache;
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
                    eprintln!(
                        "[sophon] WARNING: could not open memory persistence file {:?}: {}",
                        expanded, e
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

                let embedder_choice = std::env::var("SOPHON_EMBEDDER")
                    .unwrap_or_default()
                    .to_lowercase();

                let result = match embedder_choice.as_str() {
                    #[cfg(feature = "bge")]
                    "bge" => {
                        eprintln!("[sophon] Using BGE-small-en-v1.5 embedder (semantic)");
                        match semantic_retriever::BgeEmbedder::new() {
                            Ok(bge) => Retriever::with_embedder(
                                rcfg,
                                std::sync::Arc::new(bge),
                            ),
                            Err(e) => {
                                eprintln!(
                                    "[sophon] WARNING: BGE embedder failed to load: {e}. Falling back to HashEmbedder."
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
                        eprintln!(
                            "[sophon] WARNING: could not open semantic retriever at {:?}: {}",
                            dir, e
                        );
                        None
                    }
                }
            }
            Err(_) => None,
        };

        Self {
            prompt_compressor: PromptCompressor::with_config(cfg.prompt),
            memory_manager,
            delta_streamer: DeltaStreamer::new(cfg.delta.max_files),
            fragment_cache: FragmentCache::with_config(cfg.fragment),
            output_compressor: OutputCompressor::default(),
            codebase_navigator: Navigator::new(NavigatorConfig::default()),
            retriever,
            stats: StatsCollector::default(),
        }
    }

    pub fn handle_tool_call(&mut self, name: &str, arguments: Value) -> anyhow::Result<Value> {
        handle_tool_call(self, name, arguments)
    }

    /// Handle one JSON-RPC message for MCP stdio transport.
    /// Returns `Ok(None)` for notifications that require no reply.
    pub fn handle_json_rpc_message(&mut self, message: &Value) -> anyhow::Result<Option<Value>> {
        let method = message
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or_default();
        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id.unwrap_or(Value::Null),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "sophon",
                            "version": "1.0.0"
                        }
                    }
                });
                Ok(Some(response))
            }
            "notifications/initialized" => Ok(None),
            "ping" => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id.unwrap_or(Value::Null),
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
                    "id": id.unwrap_or(Value::Null),
                    "result": {
                        "tools": tools
                    }
                });
                Ok(Some(response))
            }
            "tools/call" => {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("tools/call missing params.name"))?;
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));

                match self.handle_tool_call(name, arguments) {
                    Ok(result) => {
                        let result_text = serde_json::to_string(&result)?;
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": id.unwrap_or(Value::Null),
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
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": id.unwrap_or(Value::Null),
                            "result": {
                                "content": [
                                    {"type": "text", "text": err.to_string()}
                                ],
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
                        Err(err) => json!({"ok": false, "error": err.to_string()}),
                    };
                    return Ok(Some(legacy));
                }

                if id.is_some() {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "error": {
                            "code": -32601,
                            "message": format!("method not found: {}", method)
                        }
                    });
                    Ok(Some(response))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub async fn run_stdio(mut self) -> anyhow::Result<()> {
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
                    let error = json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {}", err)
                        }
                    });
                    out.write_all(format!("{}\n", serde_json::to_string(&error)?).as_bytes())
                        .await?;
                    out.flush().await?;
                    continue;
                }
            };

            if let Some(response) = self.handle_json_rpc_message(&parsed)? {
                out.write_all(format!("{}\n", serde_json::to_string(&response)?).as_bytes())
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
