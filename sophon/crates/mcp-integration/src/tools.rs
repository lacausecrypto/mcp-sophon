use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// All Sophon tools exposed via MCP.
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "compress_prompt".to_string(),
            description: "Compress a system prompt based on the current query context. Returns minimized prompt containing only relevant sections.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The full system prompt to compress"},
                    "query": {"type": "string", "description": "The current user query"},
                    "max_tokens": {"type": "integer", "default": 2000}
                },
                "required": ["prompt", "query"]
            }),
        },
        ToolDefinition {
            name: "compress_history".to_string(),
            description: "Compress conversation history while preserving semantic meaning. \
                When the optional `query` parameter is provided AND the semantic retriever \
                is activated (set the SOPHON_RETRIEVER_PATH environment variable), the \
                response also includes top-k retrieved chunks relevant to that query — \
                this closes the recall gap on long conversations where summary alone is \
                not enough. The retriever uses a deterministic hash embedder by default \
                (no ML, no model download); build the workspace with --features bert for \
                semantic embeddings via candle.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": { "type": "string", "enum": ["user", "assistant", "system"] },
                                "content": { "type": "string" }
                            }
                        }
                    },
                    "max_tokens": {"type": "integer", "default": 2000},
                    "recent_window": {"type": "integer", "default": 5},
                    "include_index": {"type": "boolean", "default": false, "description": "Include dense semantic index (embeddings) in the response. Off by default to keep output small."},
                    "query": {"type": "string", "description": "Optional retrieval query. Activates the semantic retriever if SOPHON_RETRIEVER_PATH is set."},
                    "retrieval_top_k": {"type": "integer", "description": "Override the number of retrieved chunks for this call."}
                },
                "required": ["messages"]
            }),
        },
        ToolDefinition {
            name: "update_memory".to_string(),
            description: "Stateful session memory: append new messages to the server-side history and/or return a compressed snapshot. Avoids re-sending the full history on every call.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "New messages to append (optional).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": { "type": "string", "enum": ["user", "assistant", "system"] },
                                "content": { "type": "string" }
                            }
                        }
                    },
                    "reset": {"type": "boolean", "default": false, "description": "Clear the session history before appending."},
                    "return_snapshot": {"type": "boolean", "default": true, "description": "Return the compressed snapshot of the accumulated history."},
                    "include_index": {"type": "boolean", "default": false}
                }
            }),
        },
        ToolDefinition {
            name: "read_file_delta".to_string(),
            description: "Read file, returning delta from known version/hash when possible.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "known_version": {"type": "integer"},
                    "known_hash": {"type": "string"}
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "write_file_delta".to_string(),
            description: "Write file changes using delta operations or structured edits.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "changes": {"type": "object"}
                },
                "required": ["path", "changes"]
            }),
        },
        ToolDefinition {
            name: "encode_fragments".to_string(),
            description: "Replace known fragments in content with references.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "auto_detect": {"type": "boolean", "default": true}
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: "decode_fragments".to_string(),
            description: "Expand fragment references back to full content.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"}
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: "count_tokens".to_string(),
            description: "Count cl100k_base tokens in a given text.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }),
        },
        ToolDefinition {
            name: "compress_output".to_string(),
            description: "Compress the stdout/stderr of a shell command before it enters \
                the LLM context. Applies command-aware filters (git, cargo test, pytest, \
                vitest/jest, go test, ls, grep, find, docker ps, docker logs) or a generic \
                fallback. Preserves errors, modified files, and test failures; drops \
                boilerplate (OK lines, progress counters, instruction text). \
                Deterministic, zero ML, zero network.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command that produced the output (e.g. 'git status', 'cargo test')."
                    },
                    "output": {
                        "type": "string",
                        "description": "The raw stdout/stderr text of the command."
                    }
                },
                "required": ["command", "output"]
            }),
        },
        ToolDefinition {
            name: "navigate_codebase".to_string(),
            description: "Build a compact, token-budgeted map of a repository without reading \
                every file. Walks the given root, extracts top-level symbols (functions, classes, \
                structs, traits, interfaces, …) from Rust / Python / JS+TS / Go files, ranks \
                files via personalised PageRank over a symbol-reference graph, and returns a \
                ranked digest. With an optional `query`, the ranker biases toward files whose \
                path or symbol names match the query keywords, making this a fast 'where should I \
                look?' tool. Scan results are cached server-side; subsequent calls without a \
                `root` reuse the cache. Deterministic, no ML, regex-based extractors for 4 \
                languages (tree-sitter backend reserved behind a Cargo feature).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Repository root to scan. If omitted, uses the cached root from the previous call."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional free-form question. Keywords boost matching files in the ranking."
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 1500,
                        "description": "Soft cap on digest output tokens."
                    },
                    "force_rescan": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force a fresh scan even if a cached root exists."
                    }
                }
            }),
        },
        ToolDefinition {
            name: "get_token_stats".to_string(),
            description: "Get token savings statistics across modules.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "period": {"type": "string", "enum": ["session", "day", "week", "all"], "default": "session"}
                }
            }),
        },
    ]
}
