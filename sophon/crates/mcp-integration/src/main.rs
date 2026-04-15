use std::{env, fs, io::Read, process::Stdio};

use cli_hooks::{CommandRewriter, HookInstaller, RewriteResult, SupportedAgent};
use mcp_integration::{SophonConfig, SophonServer};
use output_compressor::OutputCompressor;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = env::args().collect::<Vec<_>>();

    if args.len() < 2 {
        eprintln!("{}", usage());
        return Ok(());
    }

    let command = args[1].as_str();
    let config_path = arg_value(&args, "--config");

    // Commands that do NOT need a SophonServer can short-circuit here so
    // they don't pay the startup cost of loading config / opening the
    // retriever store / etc.
    match command {
        "exec" => return run_exec(&args),
        "compress-output" => return run_compress_output(&args),
        "hook" => return run_hook(&args),
        "codebase-scan" => return run_codebase_scan(&args),
        _ => {}
    }

    let cfg = SophonConfig::resolve(config_path.as_deref().map(std::path::Path::new))?;
    let mut server = SophonServer::with_config(cfg);

    match command {
        "serve" => {
            server.run_stdio().await?;
        }
        "compress-prompt" => {
            let prompt_path =
                arg_value(&args, "--prompt").ok_or_else(|| anyhow::anyhow!("missing --prompt"))?;
            let query =
                arg_value(&args, "--query").ok_or_else(|| anyhow::anyhow!("missing --query"))?;
            let max_tokens = arg_value(&args, "--max-tokens").and_then(|s| s.parse::<usize>().ok());
            let prompt = fs::read_to_string(prompt_path)?;

            let output = server.handle_tool_call(
                "compress_prompt",
                json!({"prompt": prompt, "query": query, "max_tokens": max_tokens}),
            )?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "compress-history" => {
            let input_path =
                arg_value(&args, "--input").ok_or_else(|| anyhow::anyhow!("missing --input"))?;
            let raw = fs::read_to_string(input_path)?;
            let messages: serde_json::Value = serde_json::from_str(&raw)?;

            let output =
                server.handle_tool_call("compress_history", json!({"messages": messages}))?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "stats" => {
            let period = arg_value(&args, "--period").unwrap_or_else(|| "session".to_string());
            let output = server.handle_tool_call("get_token_stats", json!({"period": period}))?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        other => {
            return Err(anyhow::anyhow!("unknown command: {}", other));
        }
    }

    Ok(())
}

fn usage() -> &'static str {
    "Usage:\n  \
     sophon serve\n  \
     sophon exec -- <command>\n  \
     sophon compress-output --cmd <command> [--input <file>]\n  \
     sophon compress-prompt --prompt <file> --query <text> [--max-tokens <n>]\n  \
     sophon compress-history --input <json>\n  \
     sophon stats [--period session|day|week|all]\n  \
     sophon hook rewrite --agent <claude|cursor|gemini>\n  \
     sophon hook install --agent <claude> [--global]\n  \
     sophon hook uninstall --agent <claude> [--global]\n  \
     sophon hook status\n  \
     sophon codebase-scan --root <dir> [--prefer-git true|false] [--query <text>] [--max-tokens <n>]"
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

/// `sophon codebase-scan --root <dir> [--prefer-git true|false]
///  [--query <text>] [--max-tokens <n>]` — one-shot scan + digest
///
/// Exposes the `prefer_git` knob directly so benchmarks can toggle
/// between the git-ls-files and walkdir paths without going through
/// the MCP tool (which is default-true). Prints the full digest JSON
/// to stdout, plus timing on stderr.
fn run_codebase_scan(args: &[String]) -> anyhow::Result<()> {
    use codebase_navigator::{DigestConfig, Navigator, NavigatorConfig};
    use std::time::Instant;

    let root = arg_value(args, "--root").ok_or_else(|| anyhow::anyhow!("missing --root"))?;
    let prefer_git = arg_value(args, "--prefer-git")
        .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
        .unwrap_or(true);
    let query = arg_value(args, "--query");
    let max_tokens = arg_value(args, "--max-tokens")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1500);

    let cfg = NavigatorConfig {
        prefer_git,
        ..Default::default()
    };
    let mut nav = Navigator::new(cfg);

    let t0 = Instant::now();
    let scan_result = nav
        .scan(&std::path::PathBuf::from(&root))
        .map_err(|e| anyhow::anyhow!("scan failed: {}", e))?;
    let scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut dcfg = DigestConfig::default();
    dcfg.max_tokens = max_tokens;
    let t1 = Instant::now();
    let digest = nav.digest(query.as_deref(), &dcfg);
    let digest_ms = t1.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[codebase-scan] prefer_git={} files={} symbols={} edges={} scan_ms={:.1} digest_ms={:.1}",
        prefer_git,
        digest.total_files_scanned,
        digest.total_symbols_found,
        digest.edges_in_graph,
        scan_ms,
        digest_ms,
    );

    let mut output = serde_json::to_value(digest)?;
    if let Some(obj) = output.as_object_mut() {
        obj.insert(
            "scan_result".to_string(),
            serde_json::to_value(&scan_result)?,
        );
        obj.insert(
            "timing".to_string(),
            serde_json::json!({
                "scan_ms": scan_ms,
                "digest_ms": digest_ms,
                "total_ms": scan_ms + digest_ms,
            }),
        );
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// `sophon exec -- <cmd...>` — run `cmd` with the inherited
/// environment, capture its combined stdout/stderr, pipe it through
/// the output compressor, and print the compressed result to stdout.
/// Exits with the child's exit code.
fn run_exec(args: &[String]) -> anyhow::Result<()> {
    // Find the `--` separator that marks the beginning of the nested
    // command. Anything before `--` is a sophon-exec flag; everything
    // after is the command to run.
    let sep_idx = args.iter().position(|a| a == "--").ok_or_else(|| {
        anyhow::anyhow!("sophon exec: missing `--` separator before the nested command")
    })?;
    if sep_idx + 1 >= args.len() {
        return Err(anyhow::anyhow!("sophon exec: no command after `--`"));
    }

    let nested = &args[sep_idx + 1..];
    let display = nested.join(" ");

    let mut child = std::process::Command::new(&nested[0])
        .args(&nested[1..])
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn `{}`: {}", display, e))?;

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_string(&mut stdout_buf).ok();
    }
    if let Some(mut stderr) = child.stderr.take() {
        stderr.read_to_string(&mut stderr_buf).ok();
    }
    let status = child.wait()?;
    let combined = if stderr_buf.is_empty() {
        stdout_buf
    } else if stdout_buf.is_empty() {
        stderr_buf
    } else {
        format!("{}\n{}", stdout_buf, stderr_buf)
    };

    let compressor = OutputCompressor::default();
    let result = compressor.compress(&display, &combined);
    println!("{}", result.compressed);

    // Emit a one-line footer on stderr so the caller (and the LLM, if
    // piping) sees the compression stats but the main output stays
    // clean.
    eprintln!(
        "[sophon-exec] filter={} original={}t compressed={}t ratio={:.3} strategies={:?}",
        result.filter_name,
        result.original_tokens,
        result.compressed_tokens,
        result.ratio,
        result.strategies_applied
    );

    std::process::exit(status.code().unwrap_or(0));
}

/// `sophon compress-output --cmd "<cmd>" [--input <file>|-]` — same as
/// `sophon exec` but the raw output is provided by the caller (file or
/// stdin). Useful for agents that capture output themselves and just
/// want Sophon to compress it.
fn run_compress_output(args: &[String]) -> anyhow::Result<()> {
    let cmd = arg_value(args, "--cmd").ok_or_else(|| anyhow::anyhow!("missing --cmd"))?;
    let input_path = arg_value(args, "--input");

    let raw = match input_path.as_deref() {
        None | Some("-") => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
        Some(path) => fs::read_to_string(path)?,
    };

    let compressor = OutputCompressor::default();
    let result = compressor.compress(&cmd, &raw);
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `sophon hook ...` — rewrite / install / uninstall / status.
fn run_hook(args: &[String]) -> anyhow::Result<()> {
    let sub = args
        .get(2)
        .ok_or_else(|| {
            anyhow::anyhow!("missing hook subcommand (rewrite|install|uninstall|status)")
        })?
        .as_str();

    match sub {
        "rewrite" => {
            // The command to rewrite can be passed either via `--command
            // "<str>"` or via a trailing positional list. Agents typically
            // pass their own JSON on stdin; for a first landing we
            // support the simple positional form which is easiest to
            // wire into shell hooks.
            let rewriter = CommandRewriter::new();

            // Two accepted forms:
            //   sophon hook rewrite --command "git status"
            //   sophon hook rewrite [--agent X] -- git status
            // For the second form we locate the `--` separator and
            // take everything after it as the raw command.
            let cmd = if let Some(c) = arg_value(args, "--command") {
                c
            } else if let Some(sep_idx) = args.iter().position(|a| a == "--") {
                args.iter()
                    .skip(sep_idx + 1)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                // Fallback: skip the known flags and join the tail.
                let mut skip = 3; // sophon hook rewrite
                if arg_value(args, "--agent").is_some() {
                    skip += 2;
                }
                args.iter()
                    .skip(skip)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            if cmd.trim().is_empty() {
                // Nothing to rewrite — emit a neutral passthrough JSON.
                println!("{}", json!({"action": "passthrough", "command": ""}));
                return Ok(());
            }

            match rewriter.rewrite(&cmd) {
                RewriteResult::Passthrough(c) => {
                    println!(
                        "{}",
                        json!({"action": "passthrough", "command": c}).to_string()
                    );
                }
                RewriteResult::Rewritten {
                    original,
                    rewritten,
                    rule,
                } => {
                    println!(
                        "{}",
                        json!({
                            "action": "rewrite",
                            "original": original,
                            "command": rewritten,
                            "rule": rule,
                        })
                        .to_string()
                    );
                }
            }
        }
        "install" => {
            let agent_name = arg_value(args, "--agent").unwrap_or_else(|| "claude".to_string());
            let agent = SupportedAgent::parse(&agent_name)
                .ok_or_else(|| anyhow::anyhow!("unknown agent: {}", agent_name))?;
            let global = args.iter().any(|a| a == "--global");
            let report = HookInstaller::install(agent, global)?;
            println!(
                "installed hooks for {:?} at {:?} (action={:?})",
                report.agent, report.settings_path, report.action
            );
            if let Some(notes) = report.notes_path {
                println!("notes written to {:?}", notes);
            }
        }
        "uninstall" => {
            let agent_name = arg_value(args, "--agent").unwrap_or_else(|| "claude".to_string());
            let agent = SupportedAgent::parse(&agent_name)
                .ok_or_else(|| anyhow::anyhow!("unknown agent: {}", agent_name))?;
            let global = args.iter().any(|a| a == "--global");
            let report = HookInstaller::uninstall(agent, global)?;
            println!(
                "uninstalled hooks for {:?} at {:?} (action={:?})",
                report.agent, report.settings_path, report.action
            );
        }
        "status" => {
            let rewriter = CommandRewriter::new();
            println!(
                "Sophon hook rewriter — {} rules loaded",
                rewriter.rules().len()
            );
            for rule in rewriter.rules() {
                println!("  {:20}  {}", rule.name, rule.pattern.as_str());
            }
        }
        other => {
            return Err(anyhow::anyhow!(
                "unknown hook subcommand: {} (expected rewrite|install|uninstall|status)",
                other
            ));
        }
    }

    Ok(())
}
