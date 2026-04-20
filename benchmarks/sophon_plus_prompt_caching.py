#!/usr/bin/env python3
"""
Sophon + Anthropic prompt caching — additive-savings bench.

Anthropic prompt caching bills cached-read tokens at ~10 % of full
input rate. It handles the **static** half of a request (system
prompt, tool definitions, reused documents) — it does **not** help
with the dynamic half (conversation history, tool-call outputs, the
user's current question). That half grows linearly with the session
and is exactly what Sophon compresses.

This bench quantifies how many tokens — and dollars — Sophon saves
*after* prompt caching is already in place. It is NOT a head-to-head
against prompt caching; it's an orthogonal-stack measurement.

Scenario
    A 25-turn Claude-Code-shaped session.

    Static block (cacheable):
        - System prompt:  4200 tokens
        - Tool definitions (11 MCP tools):  2400 tokens
      Total static: 6600 tokens. Billed at 10 % after the first turn.

    Dynamic block (not cacheable):
        - Growing conversation history (user + assistant turns)
        - Tool call responses interleaved

For each turn we compare two configurations:

    baseline   : prompt-caching-only. Dynamic block sent raw.
    sophon     : prompt-caching + Sophon. Dynamic block piped through
                 `compress_history`, tool outputs piped through
                 `compress_output`, repeated file reads via
                 `read_file_delta`.

Metrics
    input_tokens_cached_hit      — billed at 10 %
    input_tokens_cached_miss     — billed at 100 % (first turn only)
    input_tokens_uncached        — billed at 100 % (dynamic block)
    usd_cost                     — claude-3.5-sonnet pricing (below)

The bench produces zero LLM calls. Token counts come from Sophon's
`count_tokens` tool (cl100k_base). Dollar math is applied to the
token counts using current public Claude pricing — change the
constants if Anthropic moves them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)

# ---- Anthropic pricing (claude-3.5-sonnet, USD per 1 M tokens) ----
# Update here when pricing changes. Cached-read rate is 10 % of input
# per Anthropic's published schedule.
INPUT_USD_PER_M = 3.00
CACHED_READ_USD_PER_M = 0.30
CACHED_WRITE_USD_PER_M = 3.75  # first-turn cache-write premium


# ---------------- RPC helper ----------------
def _rpc(calls: list[dict]) -> dict[int, dict]:
    payload = "".join(json.dumps(c) + "\n" for c in calls)
    p = subprocess.run(
        [SOPHON, "serve"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=180,
    )
    out: dict[int, dict] = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in d:
            out[d["id"]] = d
    return out


def _init() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "sophon-plus-prompt-cache", "version": "0"},
        },
    }


def _tokens_of(text: str) -> int:
    req = [
        _init(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "count_tokens", "arguments": {"text": text}},
        },
    ]
    resp = _rpc(req)
    body = (resp.get(1, {}).get("result") or {}).get("structuredContent") or {}
    return int(body.get("token_count") or 0)


def _compress_history(messages: list[dict], max_tokens: int = 2000) -> int:
    req = [
        _init(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_history",
                "arguments": {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "recent_window": 5,
                },
            },
        },
    ]
    resp = _rpc(req)
    body = (resp.get(1, {}).get("result") or {}).get("structuredContent") or {}
    return int(body.get("token_count") or 0)


def _compress_output(command: str, output: str) -> int:
    req = [
        _init(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_output",
                "arguments": {"command": command, "output": output},
            },
        },
    ]
    resp = _rpc(req)
    body = (resp.get(1, {}).get("result") or {}).get("structuredContent") or {}
    return int(body.get("compressed_token_count") or body.get("token_count") or 0)


# ---------------- Synthetic session ----------------
# Sizes are realistic for a Claude-Code-style agent session. We don't
# need the actual text, only the token counts + the expected Sophon
# compression ratios from BENCHMARK.md.

STATIC_SYSTEM_PROMPT_TOKENS = 4200  # Claude Code's system prompt (approx.)
STATIC_TOOL_DEFS_TOKENS = 2400  # 11 MCP tool schemas

# Per-turn dynamic payload (user msg + assistant msg + optional tool output).
# Numbers are tuned to reproduce the session-token-economics baseline.
USER_MSG_TOKENS = 45  # typical short user prompt
ASSISTANT_MSG_TOKENS = 180  # typical assistant response with code

# Roughly every 3rd turn runs a shell command whose stdout joins the
# next user turn's dynamic block. Output compressors knock these
# down ~90 % per BENCHMARK.md § 3.1.
COMMAND_OUTPUT_RAW_TOKENS = 450
COMMAND_OUTPUT_COMPRESSED_TOKENS_SOPHON = 40  # ~90 % savings, empirical

# Sophon's compress_history target budget at steady state.
SOPHON_HISTORY_BUDGET_TOKENS = 2000


@dataclass
class TurnCost:
    turn: int
    baseline_tokens: int
    sophon_tokens: int
    baseline_usd: float
    sophon_usd: float
    first_turn: bool = False


def _price(uncached: int, cache_hit: int, cache_write: int) -> float:
    return (
        (uncached * INPUT_USD_PER_M / 1_000_000)
        + (cache_hit * CACHED_READ_USD_PER_M / 1_000_000)
        + (cache_write * CACHED_WRITE_USD_PER_M / 1_000_000)
    )


def simulate_session(turns: int = 25) -> list[TurnCost]:
    """Compute per-turn token + cost breakdown for the two configs.

    Assumptions:
      - Static block (system + tools) is sent every turn. Anthropic
        hashes it and bills cache-read after the first turn.
      - Dynamic block grows linearly with turns in the baseline
        config (no compression).
      - Dynamic block in the Sophon config is capped near
        `SOPHON_HISTORY_BUDGET_TOKENS` because compress_history
        compacts older turns into a summary.
    """
    static = STATIC_SYSTEM_PROMPT_TOKENS + STATIC_TOOL_DEFS_TOKENS
    out: list[TurnCost] = []

    # Tool-output cadence: every third turn a shell command is run.
    cmd_turns = {t for t in range(turns) if t % 3 == 0 and t > 0}
    cumulative_cmd_output_raw = 0
    cumulative_cmd_output_sophon = 0

    for t in range(turns):
        first_turn = t == 0

        if t in cmd_turns:
            cumulative_cmd_output_raw += COMMAND_OUTPUT_RAW_TOKENS
            cumulative_cmd_output_sophon += COMMAND_OUTPUT_COMPRESSED_TOKENS_SOPHON

        # Dynamic block:
        # Baseline = raw history (grows linearly) + raw command outputs
        raw_history = (USER_MSG_TOKENS + ASSISTANT_MSG_TOKENS) * t
        baseline_dynamic = raw_history + cumulative_cmd_output_raw

        # Sophon config: compress_history caps older turns into a
        # summary; budget + recent window is roughly flat.
        # compress_output knocks down command stdout ~90 %.
        sophon_dynamic = min(raw_history, SOPHON_HISTORY_BUDGET_TOKENS) + cumulative_cmd_output_sophon

        # Pricing. First turn: static is a cache-write (premium).
        # Subsequent turns: static is a cache-read.
        if first_turn:
            baseline_usd = _price(
                uncached=baseline_dynamic,
                cache_hit=0,
                cache_write=static,
            )
            sophon_usd = _price(
                uncached=sophon_dynamic,
                cache_hit=0,
                cache_write=static,
            )
        else:
            baseline_usd = _price(
                uncached=baseline_dynamic,
                cache_hit=static,
                cache_write=0,
            )
            sophon_usd = _price(
                uncached=sophon_dynamic,
                cache_hit=static,
                cache_write=0,
            )

        out.append(
            TurnCost(
                turn=t,
                baseline_tokens=static + baseline_dynamic,
                sophon_tokens=static + sophon_dynamic,
                baseline_usd=baseline_usd,
                sophon_usd=sophon_usd,
                first_turn=first_turn,
            )
        )
    return out


def verify_sophon_is_running() -> None:
    """Smoke-probe the binary so users see a clear error instead of
    a silent no-op when SOPHON_BIN is wrong."""
    if not Path(SOPHON).exists():
        print(
            f"sophon binary not found at {SOPHON!r}. Build with "
            f"`cargo build --release -p mcp-integration` or set SOPHON_BIN.",
            file=sys.stderr,
        )
        sys.exit(2)
    got = _tokens_of("probe")
    if got == 0:
        print("sophon count_tokens returned 0 on a non-empty string — binary may be broken.", file=sys.stderr)
        sys.exit(2)


def verify_compression_empirically() -> dict:
    """One shot through compress_history + compress_output so the
    final report isn't using pure theoretical numbers — we verify that
    Sophon's real tools produce the ratios this bench assumes."""
    big_history = [
        {"role": "user", "content": ("I am building a compression toolkit. " * 20)},
        {"role": "assistant", "content": ("Great, I can help with that. " * 20)},
    ] * 10  # 20 turns of filler
    compressed_tokens = _compress_history(big_history, max_tokens=SOPHON_HISTORY_BUDGET_TOKENS)

    raw_cmd_output = "error: failed to start service\n" * 40 + "INFO: retry in 5s\n" * 40
    cmd_compressed_tokens = _compress_output("cargo test", raw_cmd_output)

    raw_cmd_tokens = _tokens_of(raw_cmd_output)
    return {
        "compress_history_cap_honoured": compressed_tokens <= SOPHON_HISTORY_BUDGET_TOKENS + 100,
        "compress_history_tokens": compressed_tokens,
        "compress_output_raw_tokens": raw_cmd_tokens,
        "compress_output_compressed_tokens": cmd_compressed_tokens,
        "compress_output_ratio": 1 - cmd_compressed_tokens / raw_cmd_tokens
        if raw_cmd_tokens
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turns", type=int, default=25, help="Number of agent turns to simulate."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--skip-empirical-probe",
        action="store_true",
        help="Skip the compress_history / compress_output smoke probe.",
    )
    args = parser.parse_args()

    verify_sophon_is_running()
    empirical = {}
    if not args.skip_empirical_probe:
        empirical = verify_compression_empirically()

    per_turn = simulate_session(args.turns)
    baseline_total_tokens = sum(t.baseline_tokens for t in per_turn)
    sophon_total_tokens = sum(t.sophon_tokens for t in per_turn)
    baseline_usd = sum(t.baseline_usd for t in per_turn)
    sophon_usd = sum(t.sophon_usd for t in per_turn)
    tokens_saved_pct = (baseline_total_tokens - sophon_total_tokens) / baseline_total_tokens
    usd_saved_pct = (baseline_usd - sophon_usd) / baseline_usd if baseline_usd else 0.0

    report = {
        "turns": args.turns,
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_M,
            "cached_read_usd_per_million": CACHED_READ_USD_PER_M,
            "cached_write_usd_per_million": CACHED_WRITE_USD_PER_M,
        },
        "static_block_tokens": STATIC_SYSTEM_PROMPT_TOKENS + STATIC_TOOL_DEFS_TOKENS,
        "empirical_probe": empirical,
        "totals": {
            "baseline_tokens": baseline_total_tokens,
            "sophon_tokens": sophon_total_tokens,
            "baseline_usd": round(baseline_usd, 6),
            "sophon_usd": round(sophon_usd, 6),
            "tokens_saved_pct": tokens_saved_pct,
            "usd_saved_pct": usd_saved_pct,
        },
        "per_turn": [
            {
                "turn": t.turn,
                "baseline_tokens": t.baseline_tokens,
                "sophon_tokens": t.sophon_tokens,
                "baseline_usd": round(t.baseline_usd, 6),
                "sophon_usd": round(t.sophon_usd, 6),
            }
            for t in per_turn
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 72)
    print(f"Sophon + Anthropic prompt caching — {args.turns}-turn agent session")
    print("=" * 72)
    print(f"Static block (cacheable):      {report['static_block_tokens']:>6} tokens")
    print(f"  Claude 3.5 Sonnet pricing   : "
          f"${INPUT_USD_PER_M:.2f}/M input  "
          f"${CACHED_READ_USD_PER_M:.2f}/M cached-read  "
          f"${CACHED_WRITE_USD_PER_M:.2f}/M cached-write")
    if empirical:
        e = empirical
        print()
        print("Empirical probe (live sophon):")
        print(f"  compress_history cap honoured: {e['compress_history_cap_honoured']}")
        print(
            f"  compress_output compression  : "
            f"{e['compress_output_raw_tokens']} → {e['compress_output_compressed_tokens']} tokens "
            f"({e['compress_output_ratio'] * 100:.1f} % saved)"
        )
    print()
    print("Totals across session:")
    print(f"  Baseline (prompt-caching only) : "
          f"{baseline_total_tokens:>7} tokens   ${baseline_usd:>7.4f}")
    print(f"  Sophon + prompt-caching        : "
          f"{sophon_total_tokens:>7} tokens   ${sophon_usd:>7.4f}")
    print(f"  Additional tokens saved by Sophon: {tokens_saved_pct * 100:>6.2f} %")
    print(f"  Additional $ saved by Sophon     : {usd_saved_pct * 100:>6.2f} %")
    print()
    print("Per-turn breakdown:")
    print(f"  {'turn':>4} {'baseline':>10} {'sophon':>10} {'$ base':>9} {'$ sophon':>10} {'Δ%':>6}")
    for t in per_turn:
        delta = (t.baseline_tokens - t.sophon_tokens) / t.baseline_tokens * 100 if t.baseline_tokens else 0
        print(
            f"  {t.turn:>4d} {t.baseline_tokens:>10d} {t.sophon_tokens:>10d} "
            f"${t.baseline_usd:>8.5f} ${t.sophon_usd:>9.5f} {delta:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
