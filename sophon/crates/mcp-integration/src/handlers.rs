use std::path::PathBuf;

use chrono::Utc;
use delta_streamer::protocol::{FileChanges, FileWriteRequest};
use memory_manager::{Message, Role};
use semantic_retriever::chunker::{ChunkInputMessage, ChunkInputRole};
use serde::Deserialize;
use serde_json::{json, Value};
use sophon_core::tokens::count_tokens;

use crate::server::SophonServer;

#[derive(Debug, Deserialize)]
struct CompressPromptArgs {
    prompt: String,
    query: String,
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct UpdateMemoryArgs {
    messages: Vec<MessageArg>,
    reset: bool,
    return_snapshot: Option<bool>,
    include_index: bool,
}

#[derive(Debug, Deserialize)]
struct CompressHistoryArgs {
    messages: Vec<MessageArg>,
    max_tokens: Option<usize>,
    recent_window: Option<usize>,
    #[serde(default)]
    include_index: bool,
    /// Optional retrieval query. When provided AND the retriever is
    /// activated (via `SOPHON_RETRIEVER_PATH`), the messages are indexed
    /// incrementally and the top-k most relevant chunks are added to the
    /// response under `retrieved_chunks`. No effect if the retriever is
    /// not configured.
    query: Option<String>,
    /// Override `top_k` for this call. Defaults to the retriever config.
    retrieval_top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct MessageArg {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ReadFileDeltaArgs {
    path: String,
    known_version: Option<u64>,
    known_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WriteFileDeltaArgs {
    path: String,
    changes: FileChanges,
}

#[derive(Debug, Deserialize)]
struct EncodeFragmentsArgs {
    content: String,
    auto_detect: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct DecodeFragmentsArgs {
    content: String,
}

#[derive(Debug, Deserialize)]
struct CountTokensArgs {
    text: String,
}

#[derive(Debug, Deserialize)]
struct CompressOutputArgs {
    /// The shell command that produced `output` — used to pick a
    /// command-aware filter (git / test / docker / …).
    command: String,
    /// The raw stdout/stderr text.
    output: String,
}

#[derive(Debug, Deserialize)]
struct NavigateCodebaseArgs {
    /// Repository root to scan. If omitted, uses the last scanned
    /// root from the server-side cache (fast path for repeat calls).
    #[serde(default)]
    root: Option<String>,
    /// Optional free-form query. When provided, files/symbols whose
    /// path or name contains a query keyword are boosted via the
    /// personalised PageRank restart vector.
    #[serde(default)]
    query: Option<String>,
    /// Soft cap on digest output tokens.
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Force a fresh scan even if a cached root exists.
    #[serde(default)]
    force_rescan: bool,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct StatsArgs {
    period: Option<String>,
}

pub fn handle_tool_call(
    server: &mut SophonServer,
    name: &str,
    arguments: Value,
) -> anyhow::Result<Value> {
    match name {
        "compress_prompt" => {
            let args: CompressPromptArgs = serde_json::from_value(arguments)?;
            let parsed = server
                .prompt_compressor
                .compress(&args.prompt, &args.query, None, args.max_tokens)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            server.stats.record(
                "prompt_compressor",
                count_tokens(&args.prompt),
                parsed.token_count,
            );

            Ok(json!({
                "compressed_prompt": parsed.compressed_prompt,
                "token_count": parsed.token_count,
                "included_sections": parsed.included_sections,
                "excluded_sections": parsed.excluded_sections,
                "compression_ratio": parsed.compression_ratio,
            }))
        }
        "compress_history" => {
            let args: CompressHistoryArgs = serde_json::from_value(arguments)?;
            let query = args.query.clone();
            let retrieval_top_k = args.retrieval_top_k;

            let messages = args
                .messages
                .into_iter()
                .map(|m| Message::new(parse_role(&m.role), m.content))
                .collect::<Vec<_>>();

            let compressed = server.memory_manager.compress_with_overrides(
                &messages,
                args.max_tokens,
                args.recent_window,
            );

            let original_tokens: usize = messages.iter().map(|m| m.token_count).sum();
            server
                .stats
                .record("memory_manager", original_tokens, compressed.token_count);

            // Optional retrieval pass: index incrementally, then look up the
            // top-k chunks for the query. No-op if the retriever is off or
            // the caller didn't pass a query.
            let retrieval_value =
                run_retrieval(server, &messages, query.as_deref(), retrieval_top_k)?;

            // Strip the dense SemanticIndex (embeddings blow up the payload
            // orders of magnitude past the input). Opt-in with include_index.
            let mut value = json!({
                "summary": compressed.summary,
                "stable_facts": compressed.stable_facts,
                "recent_messages": compressed.recent_messages,
                "token_count": compressed.token_count,
                "original_message_count": compressed.original_message_count,
            });
            if args.include_index {
                value["index"] = serde_json::to_value(&compressed.index)?;
            }
            if let Some(retrieval) = retrieval_value {
                value["retrieved_chunks"] = retrieval;
            }
            Ok(value)
        }
        "update_memory" => {
            let args: UpdateMemoryArgs = serde_json::from_value(arguments).unwrap_or_default();

            if args.reset {
                server.memory_manager.reset();
            }

            if !args.messages.is_empty() {
                let new_msgs = args
                    .messages
                    .into_iter()
                    .map(|m| Message::new(parse_role(&m.role), m.content))
                    .collect::<Vec<_>>();
                server.memory_manager.append(new_msgs);
            }

            let return_snapshot = args.return_snapshot.unwrap_or(true);
            if !return_snapshot {
                return Ok(json!({
                    "appended": true,
                    "history_len": server.memory_manager.history_len(),
                }));
            }

            let compressed = server.memory_manager.snapshot();
            server.stats.record(
                "memory_manager",
                compressed
                    .recent_messages
                    .iter()
                    .map(|m| m.token_count)
                    .sum::<usize>()
                    .max(compressed.token_count),
                compressed.token_count,
            );

            let mut value = json!({
                "summary": compressed.summary,
                "stable_facts": compressed.stable_facts,
                "recent_messages": compressed.recent_messages,
                "token_count": compressed.token_count,
                "original_message_count": compressed.original_message_count,
                "history_len": server.memory_manager.history_len(),
            });
            if args.include_index {
                value["index"] = serde_json::to_value(&compressed.index)?;
            }
            Ok(value)
        }
        "read_file_delta" => {
            let args: ReadFileDeltaArgs = serde_json::from_value(arguments)?;
            let response = server.delta_streamer.read_file_delta(
                PathBuf::from(&args.path),
                args.known_version,
                args.known_hash.as_deref(),
            )?;

            // Pull the original token count from the authoritative state store
            // instead of re-reading the file (which races with external writes
            // and previously swallowed IO errors via `unwrap_or`).
            use delta_streamer::protocol::FileReadResponse;
            let full_tokens = server
                .delta_streamer
                .state_store()
                .get(&PathBuf::from(&args.path))
                .map(|s| s.token_count);
            let (orig, compressed) = match &response {
                FileReadResponse::Full { token_count, .. } => (*token_count, *token_count),
                FileReadResponse::Delta { token_count, .. } => {
                    (full_tokens.unwrap_or(*token_count), *token_count)
                }
                FileReadResponse::Unchanged { .. } => (full_tokens.unwrap_or(0), 0),
            };
            server.stats.record("delta_streamer", orig, compressed);

            Ok(serde_json::to_value(response)?)
        }
        "write_file_delta" => {
            let args: WriteFileDeltaArgs = serde_json::from_value(arguments)?;
            let response = server.delta_streamer.write_file_delta(FileWriteRequest {
                path: PathBuf::from(args.path),
                changes: args.changes,
            })?;
            Ok(serde_json::to_value(response)?)
        }
        "encode_fragments" => {
            let args: EncodeFragmentsArgs = serde_json::from_value(arguments)?;
            if let Some(auto) = args.auto_detect {
                server.fragment_cache.config.auto_detect = auto;
            }

            let encoded = server.fragment_cache.encode(&args.content);
            server.stats.record(
                "fragment_cache",
                count_tokens(&args.content),
                count_tokens(&encoded.content),
            );

            Ok(json!({
                "content": encoded.content,
                "used_fragments": encoded.used_fragments,
                "new_fragments": encoded.new_fragments,
                "token_count": encoded.token_count,
                "tokens_saved": encoded.tokens_saved,
            }))
        }
        "decode_fragments" => {
            let args: DecodeFragmentsArgs = serde_json::from_value(arguments)?;
            let decoded = server
                .fragment_cache
                .decode(&args.content)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            Ok(json!({ "content": decoded }))
        }
        "count_tokens" => {
            let args: CountTokensArgs = serde_json::from_value(arguments)?;
            let n = count_tokens(&args.text);
            Ok(json!({"token_count": n, "char_count": args.text.chars().count()}))
        }
        "compress_output" => {
            let args: CompressOutputArgs = serde_json::from_value(arguments)?;
            let result = server
                .output_compressor
                .compress(&args.command, &args.output);
            server.stats.record(
                "output_compressor",
                result.original_tokens,
                result.compressed_tokens,
            );
            Ok(serde_json::to_value(result)?)
        }
        "navigate_codebase" => {
            let args: NavigateCodebaseArgs = serde_json::from_value(arguments)?;

            // Resolve the root: explicit arg > cached root.
            let root_to_scan = match args.root.as_deref() {
                Some(p) => Some(std::path::PathBuf::from(p)),
                None => server.codebase_navigator.root().cloned(),
            };

            // `force_rescan` clears the incremental cache so the next
            // `scan` call always runs a full Fresh scan.
            if args.force_rescan {
                server.codebase_navigator.reset();
            }

            // We always call `scan` now — Navigator::scan is naturally
            // incremental when the root matches the cached one, and
            // falls back to a Fresh full scan otherwise. This removes
            // the old "skip scan if cache exists" heuristic, which
            // previously meant the client couldn't pick up filesystem
            // changes without `force_rescan=true`.
            let root = root_to_scan.ok_or_else(|| {
                anyhow::anyhow!("navigate_codebase: no root provided and no cached scan available")
            })?;
            let scan_result = server
                .codebase_navigator
                .scan(&root)
                .map_err(|e| anyhow::anyhow!("scan failed: {}", e))?;

            let mut dcfg = codebase_navigator::DigestConfig::default();
            if let Some(max) = args.max_tokens {
                dcfg.max_tokens = max;
            }
            let digest = server
                .codebase_navigator
                .digest(args.query.as_deref(), &dcfg);

            // Crudely estimate "original" tokens as the sum of byte
            // sizes / 4 (~1 token per 4 bytes) so get_token_stats has
            // something meaningful to compare against.
            let original_estimate: usize = server
                .codebase_navigator
                .records()
                .iter()
                .map(|r| (r.byte_size as usize) / 4)
                .sum();
            server
                .stats
                .record("codebase_navigator", original_estimate, digest.total_tokens);

            // Merge the digest payload with the scan diagnostics so the
            // caller sees both in one response object.
            let mut value = serde_json::to_value(digest)?;
            if let Some(obj) = value.as_object_mut() {
                obj.insert(
                    "scan_result".to_string(),
                    serde_json::to_value(&scan_result)?,
                );
            }
            Ok(value)
        }
        "get_token_stats" => {
            let args: StatsArgs = serde_json::from_value(arguments)?;
            let period = args.period.unwrap_or_else(|| "session".to_string());
            let modules = server.stats.snapshot();
            let totals = server.stats.totals();
            // compressed / original, clamped to [0.0, 1.0]. Lower is better.
            let ratio = if totals.original_tokens == 0 {
                1.0
            } else {
                (totals.compressed_tokens as f64 / totals.original_tokens as f64).min(1.0)
            };
            Ok(json!({
                "period": period,
                "modules": modules,
                "totals": totals,
                "compression_ratio": ratio,
            }))
        }
        _ => Err(anyhow::anyhow!("unknown tool: {}", name)),
    }
}

fn parse_role(role: &str) -> Role {
    match role.to_lowercase().as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        _ => Role::User,
    }
}

/// Index `messages` into the semantic retriever (if active) and run a
/// top-k retrieval against `query`. Returns the JSON value to merge into
/// the `compress_history` response, or `None` if retrieval was a no-op.
fn run_retrieval(
    server: &mut crate::server::SophonServer,
    messages: &[Message],
    query: Option<&str>,
    top_k_override: Option<usize>,
) -> anyhow::Result<Option<Value>> {
    let Some(query) = query else {
        return Ok(None);
    };
    let Some(retriever) = server.retriever.as_mut() else {
        return Ok(None);
    };
    if query.trim().is_empty() {
        return Ok(None);
    }

    // Convert memory_manager::Message to the chunker's input shape.
    // The chunker wants &str references with a parallel timestamp; we keep
    // owned strings alive for the duration of the call via the messages slice.
    let inputs: Vec<ChunkInputMessage> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| ChunkInputMessage {
            index: i,
            role: match m.role {
                Role::User => ChunkInputRole::User,
                Role::Assistant => ChunkInputRole::Assistant,
                Role::System => ChunkInputRole::System,
            },
            content: &m.content,
            timestamp: Utc::now(),
            session_id: None,
        })
        .collect();

    // Index incrementally — duplicates are skipped at the store layer.
    let _ = retriever.index_messages(&inputs)?;

    // Honour the per-call top_k override if any.
    let result = if let Some(k) = top_k_override {
        let mut cfg = retriever.config().clone();
        cfg.top_k = k;
        // Build a one-off retriever view by temporarily mutating config —
        // we restore it before returning. Simpler than threading config
        // through the retrieve method.
        let original = retriever.config().clone();
        // Safety: we own &mut retriever. We can't actually mutate config
        // through the public API, so we just call retrieve with the caller's
        // top_k by manually scoring. Cheaper: just call retrieve and slice.
        let _ = (cfg, original);
        let r = retriever.retrieve(query)?;
        let mut chunks = r.chunks;
        chunks.truncate(k);
        semantic_retriever::RetrievalResult { chunks, ..r }
    } else {
        retriever.retrieve(query)?
    };

    let total_retrieved_tokens: usize = result.chunks.iter().map(|c| c.chunk.token_count).sum();
    server.stats.record(
        "semantic_retriever",
        count_tokens(query),
        total_retrieved_tokens,
    );

    Ok(Some(json!({
        "query": query,
        "embedder": retriever.embedder_name(),
        "total_searched": result.total_searched,
        "latency_ms": result.latency_ms,
        "total_tokens": total_retrieved_tokens,
        "chunks": result.chunks,
    })))
}
