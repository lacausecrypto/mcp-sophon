use std::{env, fs, io::Read, process::Stdio};

use cli_hooks::{CommandRewriter, HookInstaller, RewriteResult, SupportedAgent};
use mcp_integration::runtime_flags::{RuntimeFlag, RuntimeFlags};
use mcp_integration::{SophonConfig, SophonServer};
use output_compressor::OutputCompressor;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

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
        "doctor" => return run_doctor(config_path.as_deref()),
        _ => {}
    }

    // Parse + validate every SOPHON_* env var once, up-front. Invalid
    // values emit a tracing::warn so users see the diagnostic before
    // the MCP handshake starts — and `sophon doctor` surfaces the
    // same data without a running server.
    RuntimeFlags::from_env().log_warnings();

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
     sophon doctor [--config <file>]\n  \
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

    tracing::info!(
        prefer_git,
        files = digest.total_files_scanned,
        symbols = digest.total_symbols_found,
        edges = digest.edges_in_graph,
        scan_ms = %format!("{scan_ms:.1}"),
        digest_ms = %format!("{digest_ms:.1}"),
        "codebase-scan complete"
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
    tracing::info!(
        filter = %result.filter_name,
        original_tokens = result.original_tokens,
        compressed_tokens = result.compressed_tokens,
        ratio = %format!("{:.3}", result.ratio),
        strategies = ?result.strategies_applied,
        "sophon-exec output compressed"
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

/// `sophon doctor` — read-only diagnostic for the installation.
///
/// Prints binary metadata, resolved config source + parse status, the
/// `SOPHON_*` env vars currently set in the environment (with
/// sanitised values so no paths / commands are echoed verbatim), path
/// existence / writability for the persistence locations, and MCP
/// client-config hints for the common agents. Non-zero exit only on
/// hard failures that would block `sophon serve` from starting —
/// warnings still exit 0 so the command is CI-safe.
fn run_doctor(config_path: Option<&str>) -> anyhow::Result<()> {
    use std::io::Write;

    let mut out = std::io::stdout().lock();
    let mut exit_code = 0i32;
    writeln!(out, "Sophon Doctor — v{}", env!("CARGO_PKG_VERSION"))?;
    writeln!(out)?;

    // -------- binary --------
    writeln!(out, "[Binary]")?;
    writeln!(out, "  version:        {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(
        out,
        "  target:         {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    )?;
    writeln!(
        out,
        "  bge embedder:   {}",
        if cfg!(feature = "bge") {
            "enabled"
        } else {
            "disabled (default build)"
        }
    )?;
    writeln!(out)?;

    // -------- config --------
    writeln!(out, "[Config]")?;
    let explicit = config_path.map(std::path::PathBuf::from);
    let source = if let Some(p) = &explicit {
        format!("--config {}", p.display())
    } else if let Ok(p) = std::env::var("SOPHON_CONFIG") {
        format!("SOPHON_CONFIG={p}")
    } else if std::path::Path::new("sophon.toml").exists() {
        "./sophon.toml".to_string()
    } else {
        "defaults (no TOML file found)".to_string()
    };
    writeln!(out, "  source: {source}")?;
    match SophonConfig::resolve(explicit.as_deref()) {
        Ok(_) => writeln!(out, "  status: ok")?,
        Err(e) => {
            writeln!(out, "  status: ERROR — {e}")?;
            exit_code = 1;
        }
    }
    writeln!(out)?;

    // -------- env vars --------
    writeln!(out, "[Runtime flags]")?;
    let snap = RuntimeFlags::from_env();
    if snap.set_flags.is_empty() {
        writeln!(out, "  (no SOPHON_* env vars set — using defaults)")?;
    } else {
        let total = RuntimeFlag::ALL.len();
        writeln!(
            out,
            "  {}/{} documented flags in use:",
            snap.set_flags.len(),
            total
        )?;
        for (name, value) in &snap.set_flags {
            let scope = RuntimeFlag::ALL
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.scope.as_str())
                .unwrap_or("unknown");
            writeln!(out, "    {name:32}  {value:5}  [{scope}]")?;
        }
    }
    if !snap.warnings.is_empty() {
        writeln!(out, "  warnings:")?;
        for w in &snap.warnings {
            writeln!(out, "    ! {w}")?;
        }
    }
    if !snap.deprecated_in_use.is_empty() {
        writeln!(out, "  deprecated recall-chasing flags active:")?;
        for name in &snap.deprecated_in_use {
            writeln!(
                out,
                "    ! {name}  (v0.4.0 experiment — scheduled for removal)"
            )?;
        }
    }
    writeln!(out)?;

    // -------- paths --------
    writeln!(out, "[Paths]")?;
    let paths: Vec<(&str, Option<std::path::PathBuf>)> = vec![
        (
            "memory",
            std::env::var("SOPHON_MEMORY_PATH")
                .ok()
                .map(std::path::PathBuf::from)
                .map(expand_tilde),
        ),
        (
            "retriever",
            std::env::var("SOPHON_RETRIEVER_PATH")
                .ok()
                .map(std::path::PathBuf::from)
                .map(expand_tilde),
        ),
        (
            "graph memory",
            std::env::var("SOPHON_GRAPH_MEMORY_PATH")
                .ok()
                .map(std::path::PathBuf::from)
                .map(expand_tilde),
        ),
    ];
    let mut any_path = false;
    for (label, path) in paths {
        let Some(p) = path else { continue };
        any_path = true;
        let exists = p.exists();
        let writable = if exists {
            // Best-effort writable check: on unix we can test the permissions
            // bits; on other platforms we defer to the actual write attempt
            // at tool-call time.
            !std::fs::metadata(&p)
                .map(|m| m.permissions().readonly())
                .unwrap_or(true)
        } else {
            // Not-yet-created files are "writable" iff the parent dir is.
            p.parent()
                .map(|parent| {
                    parent.exists()
                        && !parent
                            .metadata()
                            .map(|m| m.permissions().readonly())
                            .unwrap_or(true)
                })
                .unwrap_or(false)
        };
        let status = match (exists, writable) {
            (true, true) => "ok",
            (true, false) => "exists but not writable",
            (false, true) => "not-yet-created (parent writable)",
            (false, false) => "MISSING and parent not writable",
        };
        writeln!(out, "  {label:14}  {}  [{status}]", p.display())?;
        if !writable {
            exit_code = 1;
        }
    }
    if !any_path {
        writeln!(
            out,
            "  (no SOPHON_*_PATH env vars set — nothing to persist)"
        )?;
    }
    writeln!(out)?;

    // -------- LLM command --------
    writeln!(out, "[LLM shell-out]")?;
    let llm_cmd = std::env::var("SOPHON_LLM_CMD")
        .unwrap_or_else(|_| memory_manager::DEFAULT_LLM_CMD.to_string());
    writeln!(out, "  command: {llm_cmd}")?;
    if let Some(binary) = llm_cmd.split_whitespace().next() {
        let found = std::env::var("PATH")
            .ok()
            .map(|path| {
                path.split(':')
                    .any(|dir| std::path::Path::new(dir).join(binary).is_file())
            })
            .unwrap_or(false)
            || std::path::Path::new(binary).is_file();
        writeln!(
            out,
            "  binary on PATH: {}  ({})",
            if found { "yes" } else { "no" },
            binary
        )?;
        if !found {
            writeln!(
                out,
                "  note: Sophon falls back to the heuristic summariser when the LLM command is \
                 missing. No startup failure."
            )?;
        }
    }
    writeln!(out)?;

    // -------- MCP client hints --------
    writeln!(out, "[MCP client config]")?;
    writeln!(
        out,
        "  Claude Desktop:  ~/Library/Application Support/Claude/claude_desktop_config.json"
    )?;
    writeln!(out, "  Claude Code:     ~/.claude/settings.json (per-project) or ~/.claude/config.json (global)")?;
    writeln!(out, "  Cursor:          ~/.cursor/mcp.json")?;
    writeln!(
        out,
        "  example entry:   {{\"mcpServers\": {{\"sophon\": {{\"command\": \"sophon\", \"args\": [\"serve\"]}}}}}}"
    )?;
    writeln!(out)?;

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}

fn expand_tilde(path: std::path::PathBuf) -> std::path::PathBuf {
    if let Some(rest) = path.to_str().and_then(|s| s.strip_prefix("~/")) {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(rest);
        }
    }
    path
}

/// Install a `tracing_subscriber` that writes formatted events to stderr.
/// Filter via `RUST_LOG` (e.g. `RUST_LOG=sophon=debug`); defaults to
/// `info` so the `sophon-exec` / `codebase-scan` footers still appear as
/// they did with `eprintln!`. Stdio MCP clients that parse stdout stay
/// unaffected — tracing only writes to stderr.
///
/// Safe to call more than once (uses `try_init`) so sub-commands can
/// re-invoke it without panicking if the global subscriber is already
/// set.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .try_init();
}
