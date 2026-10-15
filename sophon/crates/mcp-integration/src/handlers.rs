use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use delta_streamer::protocol::{FileChanges, FileWriteRequest};
use memory_manager::graph::{ingest_messages_batched, query_graph, render_facts};
use memory_manager::{
    classify_question, decompose_query, extract_fact_cards, hyde_rewrite_query, is_likely_multihop,
    react_decide, rerank_chunks, summarise_tail, Message, QuestionMode, ReactDecision, Role,
};
use semantic_retriever::chunker::{ChunkInputMessage, ChunkInputRole};
use semantic_retriever::{rrf_fuse, ScoredChunk, RRF_K};
use serde::Deserialize;
use serde_json::{json, Value};
use sophon_core::tokens::count_tokens;
use tokio_util::sync::CancellationToken;

use crate::server::{SophonServer, StatsCollector};

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

/// Restrict filesystem tools to an explicitly configured workspace.
/// The opt-in boundary preserves existing trusted local usage.
fn authorize_path(path: &std::path::Path) -> anyhow::Result<PathBuf> {
    let Ok(root) = std::env::var("SOPHON_ALLOWED_ROOT") else {
        return Ok(path.to_path_buf());
    };
    authorize_path_with_root(path, std::path::Path::new(&root))
}

fn authorize_path_with_root(
    path: &std::path::Path,
    root: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    let root = std::fs::canonicalize(root)?;
    // Path::exists() returns false for dangling symlinks. Treating one as a
    // new file could let a later write follow the link outside the workspace.
    let candidate = match std::fs::symlink_metadata(path) {
        Ok(_) => std::fs::canonicalize(path)?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            let parent = std::fs::canonicalize(parent)?;
            parent.join(
                path.file_name()
                    .ok_or_else(|| anyhow::anyhow!("missing filename"))?,
            )
        }
        Err(err) => return Err(err.into()),
    };
    anyhow::ensure!(
        candidate.starts_with(&root),
        "path outside SOPHON_ALLOWED_ROOT"
    );
    Ok(candidate)
}

/// Pure tools may be computed without acquiring the shared mutable server
/// lock. Their result is returned with optional compression statistics, which
/// the transport commits after the computation has finished.
type IndependentToolResult = anyhow::Result<(Value, Option<(usize, usize)>)>;

pub(crate) fn independent_tool_call(
    name: &str,
    arguments: Value,
    output_compressor: &output_compressor::OutputCompressor,
    stats: &StatsCollector,
) -> Option<IndependentToolResult> {
    match name {
        "count_tokens" => Some((|| {
            let args: CountTokensArgs = serde_json::from_value(arguments)?;
            let n = count_tokens(&args.text);
            Ok((
                json!({"token_count": n, "char_count": args.text.chars().count()}),
                None,
            ))
        })()),
        "compress_output" => Some((|| {
            let args: CompressOutputArgs = serde_json::from_value(arguments)?;
            let result = output_compressor.compress(&args.command, &args.output);
            let accounting = (result.original_tokens, result.compressed_tokens);
            Ok((serde_json::to_value(result)?, Some(accounting)))
        })()),
        "get_token_stats" => Some((|| {
            let value = get_token_stats_value(stats, arguments)?;
            Ok((value, None))
        })()),
        _ => None,
    }
}

fn check_cancel(cancellation: Option<&CancellationToken>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cancellation.is_some_and(CancellationToken::is_cancelled),
        "tool call cancelled"
    );
    Ok(())
}

pub fn handle_tool_call(
    server: &mut SophonServer,
    name: &str,
    arguments: Value,
) -> anyhow::Result<Value> {
    handle_tool_call_with_cancel(server, name, arguments, None)
}

/// Cancellation is cooperative: state mutations remain serialized and run
/// to completion once started, while expensive multi-stage reads may stop
/// between stages. Never interrupt a persistent state mutation halfway.
pub(crate) fn handle_tool_call_with_cancel(
    server: &mut SophonServer,
    name: &str,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    match name {
        "compress_prompt" => run_prompt_tool(
            &mut server.prompt_compressor,
            &server.retriever,
            &server.stats,
            arguments,
            cancellation,
        ),
        "compress_history" => run_history_tool(
            &server.memory_manager,
            &mut server.retriever,
            &server.graph_memory,
            &server.stats,
            arguments,
            cancellation,
        ),
        "update_memory" => run_update_memory_tool(
            &mut server.memory_manager,
            &mut server.graph_memory,
            &server.stats,
            arguments,
            cancellation,
        ),
        "read_file_delta" | "write_file_delta" => run_delta_tool(
            &mut server.delta_streamer,
            &server.stats,
            name,
            arguments,
            cancellation,
        ),
        "encode_fragments" | "decode_fragments" => run_fragment_tool(
            &mut server.fragment_cache,
            &server.stats,
            name,
            arguments,
            cancellation,
        ),
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
            let (value, original, compressed) =
                run_navigate_codebase(&mut server.codebase_navigator, arguments, cancellation)?;
            server
                .stats
                .record("codebase_navigator", original, compressed);
            Ok(value)
        }
        "get_token_stats" => get_token_stats_value(&server.stats, arguments),
        _ => Err(anyhow::anyhow!("unknown tool: {}", name)),
    }
}

/// Module-local handlers serve both synchronous and stdio dispatch.
/// Memory and graph form one consistency boundary; the retriever keeps its
/// own state and is explicitly passed into history/prompt operations.
pub(crate) fn run_prompt_tool(
    compressor: &mut prompt_compressor::PromptCompressor,
    retriever: &Option<semantic_retriever::Retriever>,
    stats: &StatsCollector,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    let args: CompressPromptArgs = serde_json::from_value(arguments)?;

    // When a semantic embedder is available (via the retriever),
    // compute per-section cosine similarity scores so the compressor
    // can include sections that match semantically even without a
    // keyword hit (e.g. "loop" ≈ "iteration").
    let section_scores = compute_section_scores(compressor, retriever, &args.prompt, &args.query);

    check_cancel(cancellation)?;
    let parsed = compressor
        .compress_with_scores(
            &args.prompt,
            &args.query,
            None,
            args.max_tokens,
            section_scores.as_ref(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    check_cancel(cancellation)?;
    stats.record(
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

pub(crate) fn run_history_tool(
    memory: &memory_manager::MemoryManager,
    retriever: &mut Option<semantic_retriever::Retriever>,
    graph: &Option<memory_manager::graph::GraphStore>,
    stats: &StatsCollector,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    let args: CompressHistoryArgs = serde_json::from_value(arguments)?;
    let query = args.query.clone();
    let retrieval_top_k = args.retrieval_top_k;

    let messages = args
        .messages
        .into_iter()
        .map(|m| Message::new(parse_role(&m.role), m.content))
        .collect::<Vec<_>>();

    let compressed = memory.compress_with_overrides(
        &messages,
        args.max_tokens,
        args.recent_window,
    );

    check_cancel(cancellation)?;
    let original_tokens: usize = messages.iter().map(|m| m.token_count).sum();
    stats
        .record("memory_manager", original_tokens, compressed.token_count);

    // Optional retrieval pass: index incrementally, then look up the
    // top-k chunks for the query. No-op if the retriever is off or
    // the caller didn't pass a query.
    let retrieval_value = run_retrieval(
        retriever,
        stats,
        &messages,
        query.as_deref(),
        retrieval_top_k,
        cancellation,
    )?;

    // Path A: pure-Rust graph query. Runs when graph memory is
    // active AND a query is provided. Zero LLM calls at this
    // step — all the extraction work was done at update_memory.
    let graph_facts_value =
        if let (Some(graph), Some(q)) = (graph.as_ref(), query.as_deref()) {
            let top_k = retrieval_top_k.unwrap_or(8);
            let hits = query_graph(graph, q, top_k);
            if hits.is_empty() {
                None
            } else {
                let rendered = render_facts(&hits);
                Some(json!({
                    "rendered": rendered,
                    "fact_count": hits.len(),
                    "top_score": hits.first().map(|s| s.score).unwrap_or(0.0),
                }))
            }
        } else {
            None
        };

    // Optional fact-card extraction (SOPHON_FACT_CARDS=1). Entity-
    // indexed timeline produced via one extra Haiku call. Rendered
    // into the response as a structured block the downstream LLM
    // can pattern-match on for temporal / single-hop questions.
    let fact_cards_on = std::env::var("SOPHON_FACT_CARDS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    check_cancel(cancellation)?;
    let fact_cards_value = if fact_cards_on {
        extract_fact_cards(&messages).and_then(|fc| {
            if fc.is_empty() {
                None
            } else {
                let rendered = fc.render();
                Some(json!({
                    "rendered": rendered,
                    "entity_count": fc.entities.len(),
                    "event_count": fc.entities.values().map(|v| v.len()).sum::<usize>(),
                }))
            }
        })
    } else {
        None
    };

    check_cancel(cancellation)?;
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
        // Add a retrieval_confidence signal so the caller can adjust
        // its prompt: "low" means the retrieved chunks are weakly
        // relevant (all scores < 0.3) and the LLM should be told to
        // say "Not answerable" rather than guess. This fixes the
        // adversarial question failure mode from the N=40 bench.
        let confidence = if let Some(chunks) = retrieval.get("chunks") {
            let scores: Vec<f32> = chunks
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|c| c.get("score").and_then(|s| s.as_f64()))
                .map(|s| s as f32)
                .collect();
            let max_score = scores.iter().copied().fold(0.0f32, f32::max);
            if max_score >= 0.5 {
                "high"
            } else if max_score >= 0.3 {
                "medium"
            } else {
                "low"
            }
        } else {
            "none"
        };
        value["retrieved_chunks"] = retrieval;
        value["retrieval_confidence"] = json!(confidence);
    }
    if let Some(fc) = fact_cards_value {
        value["fact_cards"] = fc;
    }
    if let Some(gf) = graph_facts_value {
        value["graph_facts"] = gf;
    }
    Ok(value)
}

pub(crate) fn run_update_memory_tool(
    memory: &mut memory_manager::MemoryManager,
    graph: &mut Option<memory_manager::graph::GraphStore>,
    stats: &StatsCollector,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    let args: UpdateMemoryArgs = serde_json::from_value(arguments).unwrap_or_default();

    if args.reset {
        memory.reset();
    }

    let mut graph_ingest_summary: Option<Value> = None;
    if !args.messages.is_empty() {
        let new_msgs = args
            .messages
            .into_iter()
            .map(|m| Message::new(parse_role(&m.role), m.content))
            .collect::<Vec<_>>();

        // Path A: if graph memory is active, extract triples
        // from the NEW batch before appending. One Haiku call
        // per update_memory — the only LLM cost of the graph
        // path; query time stays pure Rust.
        if let Some(graph) = graph.as_mut() {
            let chunk_id = format!("update-{}", Utc::now().timestamp_millis());
            let now = Utc::now().to_rfc3339();
            // Parallelised extraction via rayon — up to N×
            // speed-up on long batches (one Haiku call per
            // ~30-message slice, run concurrently).
            let report = ingest_messages_batched(graph, &new_msgs, &chunk_id, &now, None);
            let _ = graph.save(); // best-effort persistence
            graph_ingest_summary = Some(json!({
                "triples_seen": report.triples_seen,
                "new_facts": report.new_facts,
                "merged_facts": report.merged_facts,
                "new_entities": report.new_entities,
                "fact_total": graph.fact_count(),
                "entity_total": graph.entity_count(),
            }));
        }

        memory.append(new_msgs);
    }

    let return_snapshot = args.return_snapshot.unwrap_or(true);
    if !return_snapshot {
        let mut resp = json!({
            "appended": true,
            "history_len": memory.history_len(),
        });
        if let Some(g) = graph_ingest_summary.clone() {
            resp["graph_ingest"] = g;
        }
        return Ok(resp);
    }

    let compressed = memory.snapshot();
    stats.record(
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
        "history_len": memory.history_len(),
    });
    if args.include_index {
        value["index"] = serde_json::to_value(&compressed.index)?;
    }
    if let Some(g) = graph_ingest_summary {
        value["graph_ingest"] = g;
    }
    Ok(value)
}

/// One delta-streamer transaction: the file state store and the filesystem
/// operation stay under the same module mutex. Cancellation is checked before
/// entry, never in the middle of a persistent write or store update.
pub(crate) fn run_delta_tool(
    delta: &mut delta_streamer::DeltaStreamer,
    stats: &StatsCollector,
    name: &str,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    match name {
        "read_file_delta" => {
            let args: ReadFileDeltaArgs = serde_json::from_value(arguments)?;
            let safe_path = authorize_path(std::path::Path::new(&args.path))?;
            check_cancel(cancellation)?;
            let response = delta.read_file_delta(
                safe_path.clone(),
                args.known_version,
                args.known_hash.as_deref(),
            )?;

            // Read authoritative version/token count while holding the
            // same module lock as the file read and cache update.
            use delta_streamer::protocol::FileReadResponse;
            let full_tokens = delta.state_store().get(&safe_path).map(|s| s.token_count);
            let (original, compressed) = match &response {
                FileReadResponse::Full { token_count, .. } => (*token_count, *token_count),
                FileReadResponse::Delta { token_count, .. } => {
                    (full_tokens.unwrap_or(*token_count), *token_count)
                }
                FileReadResponse::Unchanged { .. } => (full_tokens.unwrap_or(0), 0),
            };
            stats.record("delta_streamer", original, compressed);
            Ok(serde_json::to_value(response)?)
        }
        "write_file_delta" => {
            let args: WriteFileDeltaArgs = serde_json::from_value(arguments)?;
            let safe_path = authorize_path(std::path::Path::new(&args.path))?;
            check_cancel(cancellation)?;
            let response = delta.write_file_delta(FileWriteRequest {
                path: safe_path,
                changes: args.changes,
            })?;
            Ok(serde_json::to_value(response)?)
        }
        _ => Err(anyhow::anyhow!("unknown delta tool: {}", name)),
    }
}

/// Fragment encoding inserts references into a mutable store. Hold its
/// dedicated lock throughout encoding and store commit so any subsequent
/// decode observes every fragment returned by an earlier encoder response.
pub(crate) fn run_fragment_tool(
    fragments: &mut fragment_cache::FragmentCache,
    stats: &StatsCollector,
    name: &str,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Value> {
    check_cancel(cancellation)?;
    match name {
        "encode_fragments" => {
            let args: EncodeFragmentsArgs = serde_json::from_value(arguments)?;
            check_cancel(cancellation)?;
            if let Some(auto_detect) = args.auto_detect {
                fragments.config.auto_detect = auto_detect;
            }

            let encoded = fragments.encode(&args.content);
            stats.record(
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
            check_cancel(cancellation)?;
            let decoded = fragments
                .decode(&args.content)
                .map_err(|err| anyhow::anyhow!(err.to_string()))?;
            Ok(json!({ "content": decoded }))
        }
        _ => Err(anyhow::anyhow!("unknown fragment tool: {}", name)),
    }
}

/// Filesystem navigation owns only its own mutable cache. This same
/// implementation is called from the public synchronous handler and the
/// dedicated stdio worker, retaining all existing scan/digest semantics.
pub(crate) fn run_navigate_codebase(
    navigator: &mut codebase_navigator::Navigator,
    arguments: Value,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<(Value, usize, usize)> {
    check_cancel(cancellation)?;
    let args: NavigateCodebaseArgs = serde_json::from_value(arguments)?;

    // Resolve explicit root first, otherwise use the navigator's cached root.
    let root_to_scan = match args.root.as_deref() {
        Some(path) => Some(authorize_path(std::path::Path::new(path))?),
        None => navigator
            .root()
            .map(|root| authorize_path(root.as_path()))
            .transpose()?,
    };
    let root = root_to_scan.ok_or_else(|| {
        anyhow::anyhow!("navigate_codebase: no root provided and no cached scan available")
    })?;

    // Reject invalid/nonexistent directories instead of silently replacing a
    // valid cache with an empty "fresh" scan. Keep the existing cache intact.
    anyhow::ensure!(
        root.is_dir(),
        "navigate_codebase: root is not an existing directory"
    );
    check_cancel(cancellation)?;

    // A forced scan is staged in a fresh navigator and only committed after
    // its scan and digest succeed and the request is still active. This makes
    // invalid paths, IO failures, and cancellation safe for the old cache.
    let mut staged = args
        .force_rescan
        .then(|| codebase_navigator::Navigator::new(navigator.config().clone()));
    let active = match staged.as_mut() {
        Some(fresh) => fresh,
        None => navigator,
    };

    let scan_result = active
        .scan(&root)
        .map_err(|err| anyhow::anyhow!("scan failed: {}", err))?;
    check_cancel(cancellation)?;

    let mut dcfg = codebase_navigator::DigestConfig::default();
    if let Some(max_tokens) = args.max_tokens {
        dcfg.max_tokens = max_tokens;
    }
    let digest = active.digest(args.query.as_deref(), &dcfg);
    check_cancel(cancellation)?;

    let original_estimate: usize = active
        .records()
        .iter()
        .map(|record| (record.byte_size as usize) / 4)
        .sum();
    let compressed_tokens = digest.total_tokens;

    let mut value = serde_json::to_value(digest)?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "scan_result".to_string(),
            serde_json::to_value(&scan_result)?,
        );
    }
    check_cancel(cancellation)?;
    if let Some(refreshed) = staged {
        *navigator = refreshed;
    }
    Ok((value, original_estimate, compressed_tokens))
}

/// Read modules and totals from one atomic statistics snapshot.
fn get_token_stats_value(stats: &StatsCollector, arguments: Value) -> anyhow::Result<Value> {
    let args: StatsArgs = serde_json::from_value(arguments)?;
    let period = args.period.unwrap_or_else(|| "session".to_string());
    let (modules, totals) = stats.snapshot_and_totals();
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

/// Compute per-section semantic similarity scores using the embedder
/// from the retriever (if available). Extracts XML/markdown sections
/// from the prompt, embeds each alongside the query, and returns
/// a section_id → cosine_similarity map.
///
/// Returns `None` if no retriever/embedder is available, keeping the
/// fallback to keyword-only scoring.
fn compute_section_scores(
    compressor: &prompt_compressor::PromptCompressor,
    retriever: &Option<semantic_retriever::Retriever>,
    prompt: &str,
    query: &str,
) -> Option<HashMap<String, f32>> {
    let retriever = retriever.as_ref()?;
    let embedder = retriever.embedder();

    // Quick parse to get section IDs + content
    let parsed = compressor.parse(prompt).ok()?;
    if parsed.sections.is_empty() {
        return None;
    }

    // Embed the query
    let query_vec = embedder.embed(query).ok()?;

    let mut scores = HashMap::new();
    for section in &parsed.sections {
        // Use section content for embedding (not just the name)
        let text = if section.content.len() > 500 {
            &section.content[..500]
        } else {
            &section.content
        };
        if text.trim().is_empty() {
            continue;
        }
        if let Ok(sec_vec) = embedder.embed(text) {
            let sim: f32 = query_vec.iter().zip(&sec_vec).map(|(a, b)| a * b).sum();
            scores.insert(section.id.clone(), sim);
        }
    }

    Some(scores)
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
    retriever: &mut Option<semantic_retriever::Retriever>,
    stats: &StatsCollector,
    messages: &[Message],
    query: Option<&str>,
    top_k_override: Option<usize>,
    cancellation: Option<&CancellationToken>,
) -> anyhow::Result<Option<Value>> {
    let Some(query) = query else {
        return Ok(None);
    };
    let Some(retriever) = retriever.as_mut() else {
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

    check_cancel(cancellation)?;
    // Prepare every embedding before starting the durable commit. During
    // the commit no cancellation checkpoint can expose a half-indexed batch.
    let _ = retriever.index_messages_with_cancel(&inputs, || {
        cancellation.is_some_and(CancellationToken::is_cancelled)
    })?;

    let t0 = std::time::Instant::now();

    // Adaptive compression budget — opt-in via SOPHON_ADAPTIVE=1.
    //
    // One Haiku classification call decides whether the query is factual
    // recall (specific date/name/place/quantity buried somewhere) or
    // general/adversarial. The handler then picks retrieval sizing that
    // fits:
    //
    //   FactualRecall → top_k × 2 (default 5 → 10), retrieved budget × 2
    //                   (default 1000 → 2000 tokens). Wider net so the
    //                   answer chunk lands in the visible slice.
    //   General       → default values (tight context reduces hallucination
    //                   on open-ended and adversarial questions).
    //
    // Failure-safe: if the classifier returns None (LLM fail / ambiguous
    // output), default to General — the conservative choice.
    let adaptive_on = std::env::var("SOPHON_ADAPTIVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let base_top_k = top_k_override.unwrap_or(retriever.config().top_k);
    let base_budget = retriever.config().max_retrieved_tokens;
    let (effective_top_k, effective_budget, question_mode): (usize, usize, Option<&'static str>) =
        if adaptive_on {
            match classify_question(query) {
                Some(QuestionMode::FactualRecall) => (
                    base_top_k.saturating_mul(2),
                    base_budget.saturating_mul(2),
                    Some("factual_recall"),
                ),
                Some(QuestionMode::General) => (base_top_k, base_budget, Some("general")),
                None => (base_top_k, base_budget, Some("unknown")),
            }
        } else {
            (base_top_k, base_budget, None)
        };

    // Build the list of queries to run in parallel. The original query is
    // always included; additional queries are appended when the
    // corresponding feature flag is set:
    //
    // - SOPHON_MULTIHOP_LLM=1 + heuristic hit → 2-3 sub-questions (P0)
    // - SOPHON_HYDE=1 → 2-3 hypothetical-answer rewrites (HyDE)
    //
    // All rankings are then fused via RRF. Each flag is independent and
    // composable; flipping both combines decomposition AND vocabulary
    // bridging in a single retrieval pass.
    let multihop_llm_on = std::env::var("SOPHON_MULTIHOP_LLM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let hyde_on = std::env::var("SOPHON_HYDE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let react_on = std::env::var("SOPHON_REACT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Number of additional ReAct refinement passes on TOP of the base
    // retrieval (original + HyDE rewrites + multi-hop sub-queries). So
    // `SOPHON_REACT_MAX_ROUNDS=2` means at most 2 extra LLM-guided
    // retrievals, not 2 total.
    let react_max_rounds: usize = std::env::var("SOPHON_REACT_MAX_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let mut decomposed_into: Option<Vec<String>> = None;
    let mut hyde_rewrites: Option<Vec<String>> = None;

    if multihop_llm_on && is_likely_multihop(query) {
        if let Some(subs) = decompose_query(query) {
            if !subs.is_empty() {
                decomposed_into = Some(subs);
            }
        }
    }
    if hyde_on {
        if let Some(rewrites) = hyde_rewrite_query(query) {
            if !rewrites.is_empty() {
                hyde_rewrites = Some(rewrites);
            }
        }
    }

    let mut extra_queries: Vec<String> = Vec::new();
    if let Some(ref subs) = decomposed_into {
        extra_queries.extend(subs.iter().cloned());
    }
    if let Some(ref rws) = hyde_rewrites {
        extra_queries.extend(rws.iter().cloned());
    }

    // Always start with the original query. `rankings` accumulates every
    // retrieval pass (original, HyDE rewrites, multi-hop sub-queries, and
    // any ReAct follow-ups) so a single RRF fusion at the end produces the
    // final chunk order.
    let mut rankings: Vec<Vec<ScoredChunk>> = Vec::with_capacity(extra_queries.len() + 2);
    check_cancel(cancellation)?;
    let primary = retriever.retrieve(query)?;
    let mut total_searched = primary.total_searched;
    rankings.push(primary.chunks);
    for q in &extra_queries {
        check_cancel(cancellation)?;
        let r = retriever.retrieve(q)?;
        total_searched = total_searched.max(r.total_searched);
        rankings.push(r.chunks);
    }

    // Entity graph ranking (Rec #2, opt-in via SOPHON_ENTITY_GRAPH=1) —
    // a parallel retrieval path that scores chunks by IDF-weighted entity
    // overlap with the query, with a bounded 1-hop bridge for multi-hop
    // cases where the answer chunk doesn't mention the query entity
    // directly but shares an entity with a chunk that does. Pure Rust,
    // zero ML, deterministic; see `entity_graph.rs`.
    let entity_graph_on = std::env::var("SOPHON_ENTITY_GRAPH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if entity_graph_on {
        let eg_chunks = retriever.search_entity_graph(query, effective_top_k.saturating_mul(4));
        if !eg_chunks.is_empty() {
            rankings.push(eg_chunks);
        }
    }

    // ReAct-lite iterative refinement — opt-in via SOPHON_REACT=1.
    //
    // When the flag is set, we trust the caller's intent and bypass the
    // multi-hop heuristic (it under-triggers on surface-simple questions
    // like "When did X Y?" that are nonetheless multi-session underneath).
    //
    // First round **always** asks for a follow-up — the decider's
    // `has_answer` shortcut is only honoured from round 2 onward. This
    // reflects an empirical failure mode: Haiku is over-confident after
    // one pass and declares completeness when the answer chunk isn't even
    // in the top-5. Forcing at least one refinement round fixes that
    // without adding cost on single-hop questions (the extra query is
    // cheap; the LLM decide call is one extra Haiku round-trip).
    let mut react_followups: Vec<String> = Vec::new();
    if react_on {
        let mut round = 0usize;
        while react_followups.len() < react_max_rounds {
            check_cancel(cancellation)?;
            let fused_so_far = rrf_fuse(&rankings, RRF_K);
            let top_texts: Vec<&str> = fused_so_far
                .iter()
                .take(5)
                .map(|sc| sc.chunk.content.as_str())
                .collect();
            let decision = react_decide(query, &top_texts);
            match decision {
                ReactDecision::Unknown => break,
                ReactDecision::HasAnswer if round >= 1 => break,
                // Round 0 falls through even if HasAnswer: over-confident
                // early stops are the failure mode we're guarding against.
                ReactDecision::HasAnswer | ReactDecision::FollowUp(_) => {
                    // If the LLM returned HasAnswer on round 0, synthesise a
                    // follow-up by asking in plain form — the original
                    // query text is the safest sub-query to re-run, since
                    // it's guaranteed non-empty and semantically relevant.
                    let sub = match decision {
                        ReactDecision::FollowUp(s) => s,
                        _ => query.to_string(),
                    };
                    let r = retriever.retrieve(&sub)?;
                    total_searched = total_searched.max(r.total_searched);
                    rankings.push(r.chunks);
                    react_followups.push(sub);
                }
            }
            round += 1;
        }
    }

    check_cancel(cancellation)?;
    let fused = rrf_fuse(&rankings, RRF_K);

    // LLM reranking (Rec #1, opt-in via SOPHON_LLM_RERANK=1) — one Haiku
    // call re-orders the top 3×effective_top_k candidates by direct query
    // relevance. This is the "query-aware" layer that BM25/HashEmbedder
    // keyword similarity cannot provide. Rest of the list stays in RRF
    // order.
    let llm_rerank_on = std::env::var("SOPHON_LLM_RERANK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let fused: Vec<ScoredChunk> = if llm_rerank_on && !fused.is_empty() {
        let n = (effective_top_k.saturating_mul(3)).min(fused.len());
        let top_texts: Vec<&str> = fused[..n]
            .iter()
            .map(|sc| sc.chunk.content.as_str())
            .collect();
        match rerank_chunks(query, &top_texts) {
            Some(scores) => {
                let mut indexed: Vec<(usize, f32)> = scores
                    .iter()
                    .enumerate()
                    .take(n)
                    .map(|(i, &s)| (i, s))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut reranked: Vec<ScoredChunk> = Vec::with_capacity(fused.len());
                for (idx, _) in &indexed {
                    reranked.push(fused[*idx].clone());
                }
                // Preserve any candidates past the reranked window — they
                // still matter for the tail-summary step below, even if
                // they don't make the top_k.
                reranked.extend(fused.iter().skip(n).cloned());
                reranked
            }
            None => fused,
        }
    } else {
        fused
    };

    // Apply token budget + top-k to the (possibly reranked) list. Both are
    // adaptive-mode aware (see `effective_budget` / `effective_top_k`
    // above). We iterate by reference so `fused` stays available for the
    // tail-summary step.
    let budget = effective_budget;
    let mut trimmed: Vec<ScoredChunk> = Vec::new();
    let mut tokens = 0usize;
    for sc in &fused {
        if trimmed.len() >= effective_top_k {
            break;
        }
        if tokens + sc.chunk.token_count > budget && !trimmed.is_empty() {
            break;
        }
        tokens += sc.chunk.token_count;
        trimmed.push(sc.clone());
    }

    // Tail summary (Rec #1, opt-in via SOPHON_TAIL_SUMMARY=1) — compress
    // up to 10 candidates sitting right after the verbatim top-K into a
    // single short paragraph via Haiku. Cheap recovery of signal from
    // chunks that would otherwise be discarded.
    let tail_summary_on = std::env::var("SOPHON_TAIL_SUMMARY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let tail_summary_text: Option<String> = if tail_summary_on {
        let start = trimmed.len();
        let end = (start + 10).min(fused.len());
        if end > start {
            let tail_texts: Vec<&str> = fused[start..end]
                .iter()
                .map(|sc| sc.chunk.content.as_str())
                .collect();
            summarise_tail(query, &tail_texts)
        } else {
            None
        }
    } else {
        None
    };
    let (chunks_out, total_searched, latency_ms) =
        (trimmed, total_searched, t0.elapsed().as_millis() as u64);

    let total_retrieved_tokens: usize = chunks_out.iter().map(|c| c.chunk.token_count).sum();
    stats.record(
        "semantic_retriever",
        count_tokens(query),
        total_retrieved_tokens,
    );

    let mut payload = json!({
        "query": query,
        "embedder": retriever.embedder_name(),
        "total_searched": total_searched,
        "latency_ms": latency_ms,
        "total_tokens": total_retrieved_tokens,
        "chunks": chunks_out,
    });
    if let Some(subs) = decomposed_into {
        payload["decomposed_into"] = json!(subs);
    }
    if let Some(rws) = hyde_rewrites {
        payload["hyde_rewrites"] = json!(rws);
    }
    if !react_followups.is_empty() {
        payload["react_followups"] = json!(react_followups);
    }
    if let Some(mode) = question_mode {
        payload["question_mode"] = json!(mode);
    }
    if let Some(summary) = tail_summary_text {
        payload["tail_summary"] = json!(summary);
    }
    Ok(Some(payload))
}

#[cfg(test)]
mod path_security_tests {
    use super::authorize_path_with_root;
    use std::fs;

    #[test]
    fn rejects_existing_file_outside_allowed_root() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let outside = temp.path().join("secret.txt");
        fs::write(&outside, "private").unwrap();
        assert!(authorize_path_with_root(&outside, &root).is_err());
    }

    #[test]
    fn accepts_existing_and_new_files_within_root() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        // macOS resolves /var to /private/var; Windows may add a verbatim prefix.
        // The helper deliberately returns canonical paths on all platforms.
        let canonical_root = fs::canonicalize(&root).unwrap();
        let existing = root.join("source.rs");
        fs::write(&existing, "fn main() {}").unwrap();
        assert_eq!(
            authorize_path_with_root(&existing, &root).unwrap(),
            canonical_root.join("source.rs")
        );
        let new_file = root.join("new.rs");
        assert_eq!(
            authorize_path_with_root(&new_file, &root).unwrap(),
            canonical_root.join("new.rs")
        );
    }

    #[test]
    fn rejects_parent_traversal_for_new_file() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let traversal = root.join("..").join("outside.txt");
        assert!(authorize_path_with_root(&traversal, &root).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_dangling_symlink_outside_allowed_root() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        // The target does not exist yet. Path::exists() would return false
        // for the link, but opening it for writing would follow it.
        let future_outside = temp.path().join("future-secret.txt");
        let link = root.join("dangling.txt");
        symlink(&future_outside, &link).unwrap();
        assert!(!link.exists());
        assert!(fs::symlink_metadata(&link).is_ok());
        assert!(authorize_path_with_root(&link, &root).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_new_file_via_symlinked_parent_outside_root() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        let outside = temp.path().join("other");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        let link = root.join("linked-dir");
        symlink(&outside, &link).unwrap();
        assert!(authorize_path_with_root(&link.join("new.txt"), &root).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_to_outside_file() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let outside = temp.path().join("secret.txt");
        fs::write(&outside, "private").unwrap();
        let link = root.join("linked.txt");
        symlink(&outside, &link).unwrap();
        assert!(authorize_path_with_root(&link, &root).is_err());
    }
}

#[cfg(test)]
mod phase1_concurrency_tests {
    use super::*;

    #[test]
    fn cancellation_before_mutation_preserves_memory() {
        let mut server = SophonServer::new();
        server
            .handle_tool_call(
                "update_memory",
                json!({
                    "messages": [{"role": "user", "content": "original state"}],
                    "return_snapshot": false
                }),
            )
            .unwrap();

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let result = handle_tool_call_with_cancel(
            &mut server,
            "update_memory",
            json!({"reset": true, "return_snapshot": false}),
            Some(&cancelled),
        );

        assert!(result.is_err());
        assert_eq!(server.memory_manager.history_len(), 1);
    }

    #[test]
    fn invalid_navigator_root_keeps_the_existing_scan() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("lib.rs"),
            "pub fn cached_symbol() -> usize { 42 }",
        )
        .unwrap();

        let mut server = SophonServer::new();
        let first = server
            .handle_tool_call(
                "navigate_codebase",
                json!({"root": directory.path().display().to_string()}),
            )
            .unwrap();
        assert_eq!(first["scan_result"]["kind"], "fresh");

        let invalid = server.handle_tool_call(
            "navigate_codebase",
            json!({
                "root": directory.path().join("missing").display().to_string(),
                "force_rescan": true
            }),
        );
        assert!(invalid.is_err());

        let reused = server
            .handle_tool_call("navigate_codebase", json!({"query": "cached_symbol"}))
            .unwrap();
        assert_eq!(reused["scan_result"]["kind"], "incremental");
        assert_eq!(reused["scan_result"]["unchanged"], 1);
    }

    #[test]
    fn cancelled_fragment_encode_keeps_configuration_and_store() {
        let mut server = SophonServer::new();
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let result = handle_tool_call_with_cancel(
            &mut server,
            "encode_fragments",
            json!({"content": "brief", "auto_detect": false}),
            Some(&cancelled),
        );
        assert!(result.is_err());
        assert!(server.fragment_cache.config.auto_detect);
    }

    #[test]
    fn delta_versions_remain_monotonic_and_read_cache_is_consistent() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("file.rs");
        std::fs::write(&path, "pub fn original() {}\n").unwrap();
        let mut server = SophonServer::new();
        let filepath = path.to_string_lossy().to_string();
        let first = server
            .handle_tool_call("read_file_delta", json!({"path": filepath}))
            .unwrap();
        assert_eq!(first["Full"]["version"], 1);

        for version in 2..=6 {
            let response = server
                .handle_tool_call(
                    "write_file_delta",
                    json!({
                        "path": filepath,
                        "changes": {"Full": {"content": format!("pub const VALUE: u64 = {version};\n")}}
                    }),
                )
                .unwrap();
            assert_eq!(response["Full"]["version"], version);
        }
        let unchanged = server
            .handle_tool_call(
                "read_file_delta",
                json!({"path": filepath, "known_version": 6}),
            )
            .unwrap();
        assert_eq!(unchanged["Unchanged"]["version"], 6);
        assert_eq!(server.stats.snapshot()["delta_streamer"].calls, 2);
    }

    #[test]
    fn encoded_fragments_are_available_to_decoder_immediately() {
        let mut server = SophonServer::new();
        let content = format!(
            "```rust\n{}```",
            "pub fn long_repeated_rust_code(value: usize) -> usize { value * 2 + 1 }\n".repeat(80)
        );
        let encoded = server
            .handle_tool_call("encode_fragments", json!({"content": content}))
            .unwrap();
        let compressed = encoded["content"].as_str().unwrap();
        assert!(compressed.contains("[FRAGMENT:"), "expected reference");
        let decoded = server
            .handle_tool_call("decode_fragments", json!({"content": compressed}))
            .unwrap();
        assert_eq!(decoded["content"], content);
        assert_eq!(server.stats.snapshot()["fragment_cache"].calls, 1);
    }

    #[test]
    fn stateless_path_matches_regular_tool_response() {
        let mut server = SophonServer::new();
        let args = json!({"command": "cargo test", "output": "test result: ok\\n"});
        let expected = server
            .handle_tool_call("compress_output", args.clone())
            .unwrap();
        let (independent, stats) = independent_tool_call(
            "compress_output",
            args,
            &server.output_compressor,
            &server.stats,
        )
        .unwrap()
        .unwrap();
        assert_eq!(independent, expected);
        assert!(stats.is_some());

        let args = json!({"text": "Sophon parallel tokens"});
        let expected = server
            .handle_tool_call("count_tokens", args.clone())
            .unwrap();
        let (independent, stats) = independent_tool_call(
            "count_tokens",
            args,
            &server.output_compressor,
            &server.stats,
        )
        .unwrap()
        .unwrap();
        assert_eq!(independent, expected);
        assert!(stats.is_none());
    }
}
