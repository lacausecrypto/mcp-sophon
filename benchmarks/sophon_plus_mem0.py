#!/usr/bin/env python3
"""
Sophon + mem0 — orthogonal-stack bench.

Reproduces Sophon's v0.5.0 positioning: Sophon is **not** a memory
platform, it's a deterministic compressor that sits in front of one.
This bench measures what Sophon adds on top of mem0's output, not
against it.

Pipeline under test
    user turn ─► mem0.search(query) ─► relevant memories (text)
              │                         │
              │                         ▼
              │                    Sophon.compress_prompt(memories, query)
              │                         │
              ▼                         ▼
         final prompt ──────► LLM (out of scope — not called here)

The bench runs two configurations and diffs the token counts that
would hit the downstream LLM:

    A. mem0 alone          — memories concatenated as-is
    B. mem0 + Sophon       — memories piped through `compress_prompt`

We report:
    * mean tokens sent to LLM per turn, both configs
    * additional tokens saved by Sophon on top of mem0
    * latency overhead of Sophon's compression step (p50, p99)
    * answer-preservation canary: proper nouns / dates / numbers
      that were in the raw memories but dropped by Sophon

### Mem0 dependency

Real mem0 (`pip install mem0ai`) needs OpenAI / Anthropic keys and
network. To keep this bench reproducible in CI, it defaults to a
**surrogate** that emulates mem0's retrieval profile: a keyword-gated
lookup over a fixed pool of pre-written memories. The surrogate is
honest about its limits — see `SURROGATE_NOTE` printed at run start.

To run against the real mem0:

    pip install mem0ai
    export OPENAI_API_KEY=...
    python sophon_plus_mem0.py --real-mem0

Not passing `--real-mem0` prints the surrogate note in the header
so numbers are never misread as apples-to-apples neural retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)

SURROGATE_NOTE = (
    "[warn] running against the *surrogate* mem0 retriever. Pass "
    "`--real-mem0` after `pip install mem0ai` + setting OPENAI_API_KEY "
    "to compare against the real thing. Surrogate numbers are useful "
    "for Sophon-overhead measurement but NOT for absolute recall "
    "claims vs mem0."
)


# ---------------- RPC helper (mirrors other benches) ----------------
def _rpc(requests: list[dict]) -> dict[int, dict]:
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    p = subprocess.run(
        [SOPHON, "serve"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
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


def _sophon_compress(text: str, query: str, max_tokens: int = 400) -> tuple[str, int]:
    """Run `compress_prompt` via sophon stdio. Returns (compressed, tokens)."""
    init = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "sophon-plus-mem0", "version": "0"},
        },
    }
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "compress_prompt",
            "arguments": {
                "prompt": text,
                "query": query,
                "max_tokens": max_tokens,
            },
        },
    }
    resp = _rpc([init, call])
    body = resp.get(1, {}).get("result", {})
    # Structured response from MCP 2025-06-18.
    structured = body.get("structuredContent") or {}
    compressed = structured.get("compressed_prompt") or ""
    tokens = int(structured.get("token_count") or 0)
    return compressed, tokens


def _count_tokens(text: str) -> int:
    """Use sophon's cl100k_base tokenizer so the diff is apples-to-apples."""
    req = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "tok", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "count_tokens",
                "arguments": {"text": text},
            },
        },
    ]
    resp = _rpc(req)
    structured = (resp.get(1, {}).get("result") or {}).get("structuredContent") or {}
    return int(structured.get("token_count") or 0)


# ---------------- Surrogate mem0 ----------------
@dataclass
class Memory:
    text: str
    keywords: frozenset[str]


SURROGATE_POOL: list[Memory] = [
    Memory(
        "Alice started her new role as Staff Engineer at TechCorp on 2024-02-05. She leads the "
        "observability team.",
        frozenset({"alice", "techcorp", "staff", "engineer", "observability", "2024"}),
    ),
    Memory(
        "Alice's favourite pastry is a pain au chocolat; she mentioned her allergy to hazelnuts "
        "in a team lunch in March 2024.",
        frozenset({"alice", "pastry", "hazelnut", "allergy", "lunch"}),
    ),
    Memory(
        "Bob moved to Paris in June 2024 for a one-year secondment at the Paris office. His "
        "French is still shaky.",
        frozenset({"bob", "paris", "2024", "secondment", "french"}),
    ),
    Memory(
        "Bob's side project is a Rust compiler plugin that instruments async code for latency "
        "profiling. He open-sourced it under MIT in September 2024.",
        frozenset({"bob", "rust", "compiler", "async", "latency", "open-source"}),
    ),
    Memory(
        "Charlie flew to Tokyo in July 2024 for the Rust Asia conference. He presented a talk on "
        "deterministic embedding for zero-ML retrieval.",
        frozenset({"charlie", "tokyo", "rust", "conference", "embedding", "retrieval"}),
    ),
    Memory(
        "Charlie is debating whether to rewrite his company's Python ETL pipeline in Rust. He's "
        "prototyped two stages and saw a 4x throughput gain.",
        frozenset({"charlie", "python", "etl", "rust", "rewrite", "throughput"}),
    ),
    Memory(
        "Dana joined the team in October 2024 as Senior PM. Her first project is the v0.5.0 "
        "release of the Sophon context compressor.",
        frozenset({"dana", "pm", "sophon", "release", "v0.5.0"}),
    ),
    Memory(
        "Dana's daughter's name is Léa, born in April 2023. Dana usually works from 8 AM to 4 PM "
        "to match school hours.",
        frozenset({"dana", "daughter", "lea", "2023"}),
    ),
    Memory(
        "The team standup is at 10:00 UTC on Tuesdays and Thursdays. Wednesday is reserved for "
        "deep-work blocks with no meetings.",
        frozenset({"standup", "tuesday", "thursday", "wednesday", "meeting"}),
    ),
    Memory(
        "Q3 2024 OKR: reduce p99 retrieval latency from 180ms to under 100ms. Current status is "
        "on track with 112ms as of October 2024.",
        frozenset({"q3", "okr", "latency", "retrieval", "2024"}),
    ),
    Memory(
        "The production cluster runs on AWS us-east-1 with a warm standby in eu-west-3. Failover "
        "was last tested on 2024-09-14.",
        frozenset({"aws", "us-east-1", "eu-west-3", "failover", "2024"}),
    ),
    Memory(
        "Bob reviewed the mem0 integration PR on 2024-10-21 and flagged three issues around "
        "embedding cache invalidation.",
        frozenset({"bob", "mem0", "pr", "cache", "embedding"}),
    ),
]


def surrogate_mem0_search(query: str, top_k: int = 5) -> list[Memory]:
    """Keyword-gated retrieval. Not semantic; not neural. Deterministic."""
    q_kw = set(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]+", query.lower()))
    scored = []
    for m in SURROGATE_POOL:
        overlap = len(q_kw & m.keywords)
        if overlap:
            scored.append((overlap, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def real_mem0_search(query: str, user_id: str = "sophon-plus-mem0-bench") -> list[Memory]:
    """Invoke the real mem0 library. Imports are lazy so the surrogate path
    works without the dependency installed."""
    try:
        from mem0 import Memory as Mem0Memory  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "--real-mem0 requires `pip install mem0ai`. original error: " + str(e)
        ) from e
    m = Mem0Memory()
    # Seed with surrogate pool the first time. Real mem0 dedupes.
    if not m.search("seed-check", user_id=user_id)["results"]:
        for mem in SURROGATE_POOL:
            m.add(mem.text, user_id=user_id)
    result = m.search(query, user_id=user_id)
    out = []
    for entry in result.get("results", []):
        text = entry.get("memory", "")
        kw = frozenset(re.findall(r"[a-z0-9][a-z0-9-]+", text.lower()))
        out.append(Memory(text, kw))
    return out


# ---------------- Scenario ----------------
QUERIES: list[str] = [
    "What is Alice's new role?",
    "Where did Bob move last year and why?",
    "Tell me about Charlie's Tokyo talk.",
    "When is our team standup?",
    "What's the Q3 OKR on retrieval latency?",
    "Who is the PM on the Sophon release?",
    "Which region hosts the primary production cluster?",
    "What allergies do teammates mention?",
    "What's Bob's side project in Rust about?",
    "Who reviewed the mem0 integration PR and when?",
]


CANARY_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}|[A-Z][a-zé]+(?=\s)|\bv?\d+\.\d+(?:\.\d+)?\b|\b\d{1,3}(?:\.\d+)?\s?(?:ms|%)\b"
)


def canaries(text: str) -> set[str]:
    return set(CANARY_PATTERN.findall(text))


# ---------------- Bench loop ----------------
def run_turn(query: str, search) -> dict:
    hits = search(query)
    raw_memories = "\n\n".join(h.text for h in hits)
    raw_canaries = canaries(raw_memories)
    raw_tokens = _count_tokens(raw_memories) if raw_memories else 0

    t0 = time.perf_counter()
    compressed, sent_tokens = _sophon_compress(raw_memories, query, max_tokens=300)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    kept_canaries = canaries(compressed)
    preserved = len(raw_canaries & kept_canaries) / max(1, len(raw_canaries))

    return {
        "query": query,
        "hits": len(hits),
        "raw_tokens": raw_tokens,
        "sent_tokens": sent_tokens,
        "additional_saved_pct": (raw_tokens - sent_tokens) / raw_tokens
        if raw_tokens
        else 0.0,
        "canary_preserved_pct": preserved,
        "sophon_latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-mem0",
        action="store_true",
        help="Use the real mem0 library (requires `pip install mem0ai`).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    if args.real_mem0:
        search = real_mem0_search
    else:
        print(SURROGATE_NOTE, file=sys.stderr)
        search = surrogate_mem0_search

    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}. Set SOPHON_BIN.", file=sys.stderr)
        return 2

    results = [run_turn(q, search) for q in QUERIES]
    raw_total = sum(r["raw_tokens"] for r in results)
    sent_total = sum(r["sent_tokens"] for r in results)
    saved_pct = (raw_total - sent_total) / raw_total if raw_total else 0.0
    canary_mean = statistics.mean(r["canary_preserved_pct"] for r in results)
    lat_sorted = sorted(r["sophon_latency_ms"] for r in results)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p99 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.99))]

    report = {
        "mode": "real-mem0" if args.real_mem0 else "surrogate-mem0",
        "turns": len(results),
        "raw_tokens_sum": raw_total,
        "sent_tokens_sum": sent_total,
        "additional_tokens_saved_by_sophon_pct": saved_pct,
        "canary_preservation_mean_pct": canary_mean,
        "sophon_latency_ms_p50": p50,
        "sophon_latency_ms_p99": p99,
        "per_turn": results,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print(f"Sophon + mem0 bench — mode={report['mode']}, turns={report['turns']}")
        print("=" * 70)
        print(f"Raw mem0 output  : {raw_total:>8} tokens (sum across turns)")
        print(f"After Sophon     : {sent_total:>8} tokens")
        print(
            f"Additional saved : {saved_pct * 100:6.2f} %   "
            "(tokens Sophon removes AFTER mem0 retrieval)"
        )
        print(f"Canary preserved : {canary_mean * 100:6.2f} %   (proper nouns + dates + numbers)")
        print(
            f"Sophon latency   : p50={p50:>6.1f} ms   p99={p99:>6.1f} ms   "
            "(compress_prompt per turn)"
        )
        print()
        print("Per-turn detail:")
        print(
            f"  {'query':<60} {'raw':>5} {'sent':>5} {'+sav%':>6} {'canary%':>7} {'ms':>6}"
        )
        for r in results:
            print(
                f"  {r['query'][:59]:<60} "
                f"{r['raw_tokens']:>5} "
                f"{r['sent_tokens']:>5} "
                f"{r['additional_saved_pct'] * 100:>5.1f}% "
                f"{r['canary_preserved_pct'] * 100:>6.1f}% "
                f"{r['sophon_latency_ms']:>6.1f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
