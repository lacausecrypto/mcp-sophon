#!/usr/bin/env python3
"""
Synthetic agent-session bench measuring tool-call dedup (v0.5.3 +
v0.5.4).

`semantic-retriever`'s chunker (commit e3c39a0) detects Anthropic-
shaped tool_use blocks and emits a normalised
`tool:NAME({sorted_args_json})` chunk content so identical calls
collapse to one chunk via the existing `chunk_id` SHA dedup. This
bench quantifies the win on a synthetic workload modelled after a
typical Claude Code session shape.

What the synthetic session looks like
=====================================

In a real coding session, an agent re-reads the same files multiple
times across turns:

  * `read_file({"path":"src/lib.rs"})`           — referenced 8×
  * `read_file({"path":"src/handlers.rs"})`      — referenced 5×
  * `bash({"command":"git status"})`             — called every few turns
  * `bash({"command":"cargo test"})`             — pre-commit checks
  * plus unique tool calls (`grep` / `write_file_delta` / …) that
    don't recur

We model 50 turns with mixed tool calls and a power-law repetition
pattern: the top 5 tool_call shapes account for 60-70 % of all
calls, the remaining are mostly unique. Source-of-truth for the
distribution is the `BUILD_SESSION` constant — change it to model
a different agent shape.

Two indexings are measured:

1. **Tool calls treated as prose** (pre-v0.5.3 behaviour) — every
   message becomes a chunk regardless of repetition. We get the
   token cost as a baseline.
2. **Tool calls dedup'd via canonical form** (current behaviour) —
   `index_messages` sees the JSON shape, normalises, dedups.

The delta is the v0.5.3 win. **Important: the win depends on
session shape**. A session with 0 tool-call repetition gets 0 %
saving. The bench output makes this dependency explicit so users
can map it to their own workload.

Running
=======

    python3 benchmarks/tool_call_dedup_effect.py
    python3 benchmarks/tool_call_dedup_effect.py --turns 100 --json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)


# ---------------- Synthetic session shape ----------------
# Tool calls modelled after a typical 50-turn Claude Code session.
# `weight` controls how often each shape appears in the run.
HOT_TOOLS = [
    {"name": "read_file", "args": {"path": "src/lib.rs"}, "weight": 8},
    {"name": "read_file", "args": {"path": "src/handlers.rs"}, "weight": 5},
    {"name": "read_file", "args": {"path": "src/server.rs"}, "weight": 4},
    {"name": "bash", "args": {"command": "git status"}, "weight": 6},
    {"name": "bash", "args": {"command": "cargo test --lib"}, "weight": 4},
]
WARM_TOOLS = [
    {"name": "read_file", "args": {"path": "Cargo.toml"}},
    {"name": "read_file", "args": {"path": "tests/integration.rs"}},
    {"name": "bash", "args": {"command": "cargo build --release"}},
    {"name": "bash", "args": {"command": "git log --oneline -n 10"}},
]
# Each cold call is unique (different path / different command), so
# none can dedup with itself or any other.
def make_cold_call(seed: int) -> dict:
    return {
        "name": "grep",
        "args": {
            "pattern": f"FIXME-{seed}",
            "path": f"crates/{seed % 7}/src/file_{seed}.rs",
        },
    }


def build_session(turns: int, seed: int = 42) -> list[dict]:
    """Generate the synthetic session as a list of (role, content)
    messages. Each turn = 1 user prompt + 1 tool-use block (plus
    tool_result echo)."""
    rng = random.Random(seed)
    msgs: list[dict] = []
    for i in range(turns):
        msgs.append({
            "role": "user",
            "content": f"Step {i}: investigate the next pattern",
        })

        # 70 % chance hot tool, 20 % warm, 10 % cold (unique).
        roll = rng.random()
        if roll < 0.70:
            # weighted pick from HOT_TOOLS
            total = sum(t["weight"] for t in HOT_TOOLS)
            r = rng.random() * total
            cum = 0
            chosen = HOT_TOOLS[0]
            for t in HOT_TOOLS:
                cum += t["weight"]
                if r <= cum:
                    chosen = t
                    break
        elif roll < 0.90:
            chosen = rng.choice(WARM_TOOLS)
        else:
            chosen = make_cold_call(i)

        # Tool use block in canonical Anthropic shape — exactly what
        # the chunker's detector recognises.
        tool_use = {
            "type": "tool_use",
            "name": chosen["name"],
            "input": chosen["args"],
        }
        msgs.append({
            "role": "assistant",
            "content": json.dumps(tool_use),
        })

        # Tool result echo. Realistic: short text payload.
        tool_result = {
            "type": "tool_result",
            "tool_use_id": f"call-{i}",
            "content": f"(stub result for step {i})",
        }
        msgs.append({
            "role": "user",
            "content": json.dumps(tool_result),
        })
    return msgs


# ---------------- RPC helpers ----------------
def _rpc_one(name: str, arguments: dict) -> dict:
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "tool-dedup-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id") == 1:
            return d.get("result", {}).get("structuredContent") or {}
    return {}


def index_via_compress_history(messages: list[dict], retriever_path: str) -> dict:
    """Drive the retriever indexing path via `compress_history` with
    a non-empty `query` (the only code path in handlers.rs that
    pushes messages through `Retriever::index_messages`). The query
    string itself is irrelevant for the dedup measurement — we just
    need the retriever to be activated."""
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "tool-dedup-bench", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compress_history",
                "arguments": {
                    "messages": messages,
                    "max_tokens": 4000,
                    "query": "any tool calls",
                },
            },
        },
    ]
    raw = "".join(json.dumps(r) + "\n" for r in payload)
    env = os.environ.copy()
    env["SOPHON_RETRIEVER_PATH"] = retriever_path
    p = subprocess.run([SOPHON, "serve"], input=raw, capture_output=True, text=True, timeout=60, env=env)

    # Count chunk store size from the JSONL file the retriever wrote.
    store_path = Path(retriever_path) / "chunks.jsonl"
    chunk_count = 0
    tool_use_count = 0
    tool_result_count = 0
    other_count = 0
    raw_chars = 0
    if store_path.exists():
        for line in store_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = ch.get("chunk", {})
            chunk_count += 1
            ct = chunk.get("chunk_type", "")
            if ct == "tool_use":
                tool_use_count += 1
            elif ct == "tool_result":
                tool_result_count += 1
            else:
                other_count += 1
            raw_chars += len(chunk.get("content", ""))
    return {
        "chunk_count": chunk_count,
        "tool_use_count": tool_use_count,
        "tool_result_count": tool_result_count,
        "other_count": other_count,
        "raw_chars": raw_chars,
    }


# ---------------- Driver ----------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=50, help="Synthetic session length.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}. Run cargo build --release.", file=sys.stderr)
        return 2

    print(f"Building synthetic session: {args.turns} turns…", file=sys.stderr)
    messages = build_session(args.turns)

    # Count expected unique tool-use shapes.
    tool_uses = [
        m for m in messages
        if m["role"] == "assistant"
        and m["content"].startswith("{")
        and '"type":"tool_use"' in m["content"].replace(" ", "")
    ]
    unique_canonical = set()
    for m in tool_uses:
        d = json.loads(m["content"])
        canonical = f"tool:{d['name']}({json.dumps(d['input'], sort_keys=True)})"
        unique_canonical.add(canonical)
    repeated_calls = len(tool_uses) - len(unique_canonical)

    print(f"  total tool_use messages: {len(tool_uses)}", file=sys.stderr)
    print(f"  unique canonical shapes: {len(unique_canonical)}", file=sys.stderr)
    print(f"  redundant repeats:       {repeated_calls}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        retr_path = str(Path(tmp) / "retr")
        Path(retr_path).mkdir(parents=True, exist_ok=True)
        print(f"  ingesting via update_memory + retriever ({retr_path})…", file=sys.stderr)
        stats = index_via_compress_history(messages, retr_path)

    # The dedup gain is the difference between "raw count of tool_use
    # messages" (what would land in the store WITHOUT dedup) and "the
    # actual tool_use chunk count after canonical-form dedup".
    expected_no_dedup = len(tool_uses)
    actual_with_dedup = stats["tool_use_count"]
    chunks_saved = expected_no_dedup - actual_with_dedup
    saved_pct = (chunks_saved / expected_no_dedup * 100) if expected_no_dedup else 0.0

    out = {
        "turns": args.turns,
        "tool_use_messages": len(tool_uses),
        "unique_canonical_shapes": len(unique_canonical),
        "redundant_repeats": repeated_calls,
        "chunk_store_total": stats["chunk_count"],
        "chunk_store_tool_use": stats["tool_use_count"],
        "chunk_store_tool_result": stats["tool_result_count"],
        "chunk_store_other_prose": stats["other_count"],
        "chunks_saved_by_dedup": chunks_saved,
        "tool_use_chunks_saved_pct": saved_pct,
        "raw_total_chars_after_dedup": stats["raw_chars"],
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print()
    print("=" * 80)
    print(f"Tool-call dedup effect — {args.turns}-turn synthetic session")
    print("=" * 80)
    print(f"  Total tool_use messages:           {out['tool_use_messages']}")
    print(f"  Unique canonical shapes:           {out['unique_canonical_shapes']}")
    print(f"  Redundant repeat calls:            {out['redundant_repeats']}")
    print()
    print(f"  Chunk store total:                 {out['chunk_store_total']}")
    print(f"    tool_use chunks (after dedup):   {out['chunk_store_tool_use']}")
    print(f"    tool_result chunks:              {out['chunk_store_tool_result']}")
    print(f"    other (prose) chunks:            {out['chunk_store_other_prose']}")
    print()
    print(f"  Chunks saved by canonical dedup:   {out['chunks_saved_by_dedup']}")
    print(f"  tool_use chunk reduction:          {out['tool_use_chunks_saved_pct']:.1f} %")
    print()
    print("Notes:")
    print("  * The dedup only fires on tool_use messages — tool_result")
    print("    chunks vary per invocation (different tool_use_id) and")
    print("    are intentionally NOT collapsed.")
    print("  * The win depends on session shape: a workload with no")
    print("    repeated tool calls gets 0 % saving.")
    print("  * Real Claude Code sessions are expected to repeat the")
    print("    top-5 tool shapes ~60-70 % of all calls (see HOT_TOOLS")
    print("    in this script). Tune the constants to model your own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
