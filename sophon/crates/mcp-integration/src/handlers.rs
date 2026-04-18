use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use delta_streamer::protocol::{FileChanges, FileWriteRequest};
use memory_manager::graph::{ingest_messages_batched, query_graph, render_facts};
use memory_manager::{
    classify_question, decompose_query, extract_fact_cards, hyde_rewrite_query,
    is_likely_multihop, react_decide, rerank_chunks, summarise_tail, Message, QuestionMode,
    ReactDecision, Role,
};
use semantic_retriever::chunker::{ChunkInputMessage, ChunkInputRole};
use semantic_retriever::{rrf_fuse, ScoredChunk, RRF_K};
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

            // When a semantic embedder is available (via the retriever),
            // compute per-section cosine similarity scores so the compressor
            // can include sections that match semantically even without a
            // keyword hit (e.g. "loop" ≈ "iteration").
            let section_scores = compute_section_scores(server, &args.prompt, &args.query);

            let parsed = server
                .prompt_compressor
                .compress_with_scores(
                    &args.prompt,
                    &args.query,
                    None,
                    args.max_tokens,
                    section_scores.as_ref(),
                )
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

            // Path A: pure-Rust graph query. Runs when graph memory is
            // active AND a query is provided. Zero LLM calls at this
            // step — all the extraction work was done at update_memory.
            let graph_facts_value = if let (Some(graph), Some(q)) =
                (server.graph_memory.as_ref(), query.as_deref())
            {
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
        "update_memory" => {
            let args: UpdateMemoryArgs = serde_json::from_value(arguments).unwrap_or_default();

            if args.reset {
                server.memory_manager.reset();
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
                if let Some(graph) = server.graph_memory.as_mut() {
                    let chunk_id = format!(
                        "update-{}",
                        Utc::now().timestamp_millis()
                    );
                    let now = Utc::now().to_rfc3339();
                    // Parallelised extraction via rayon — up to N×
                    // speed-up on long batches (one Haiku call per
                    // ~30-message slice, run concurrently).
                    let report = ingest_messages_batched(
                        graph, &new_msgs, &chunk_id, &now, None,
                    );
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

                server.memory_manager.append(new_msgs);
            }

            let return_snapshot = args.return_snapshot.unwrap_or(true);
            if !return_snapshot {
                let mut resp = json!({
                    "appended": true,
                    "history_len": server.memory_manager.history_len(),
                });
                if let Some(g) = graph_ingest_summary.clone() {
                    resp["graph_ingest"] = g;
                }
                return Ok(resp);
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
            if let Some(g) = graph_ingest_summary {
                value["graph_ingest"] = g;
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

/// Compute per-section semantic similarity scores using the embedder
/// from the retriever (if available). Extracts XML/markdown sections
/// from the prompt, embeds each alongside the query, and returns
/// a section_id → cosine_similarity map.
///
/// Returns `None` if no retriever/embedder is available, keeping the
/// fallback to keyword-only scoring.
fn compute_section_scores(
    server: &SophonServer,
    prompt: &str,
    query: &str,
) -> Option<HashMap<String, f32>> {
    let retriever = server.retriever.as_ref()?;
    let embedder = retriever.embedder();

    // Quick parse to get section IDs + content
    let parsed = server.prompt_compressor.parse(prompt).ok()?;
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
                Some(QuestionMode::FactualRecall) => {
                    (base_top_k.saturating_mul(2), base_budget.saturating_mul(2), Some("factual_recall"))
                }
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
    let primary = retriever.retrieve(query)?;
    let mut total_searched = primary.total_searched;
    rankings.push(primary.chunks);
    for q in &extra_queries {
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
                indexed.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut reranked: Vec<ScoredChunk> = Vec::with_capacity(fused.len());
                for (idx, _) in &indexed {
                    reranked.push(fused[*idx].clone());
                }
                // Preserve any candidates past the reranked window — they
                // still matter for the tail-summary step below, even if
                // they don't make the top_k.
                for i in n..fused.len() {
                    reranked.push(fused[i].clone());
                }
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
    server.stats.record(
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
