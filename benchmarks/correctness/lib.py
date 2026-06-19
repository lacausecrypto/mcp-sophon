#!/usr/bin/env python3
"""
Shared helpers for the correctness bench.

Covers three concerns the bench reuses everywhere:
  - RPC to the Sophon MCP server (batched per task: one `sophon serve` process
    handles init + all the count_tokens/compress calls a task needs)
  - Sophon's own tokenizer via `count_tokens` (one measure for all three arms)
  - LLM shell-out (`claude -p`) with a disk cache so the run is resumable —
    a crash mid-run never re-pays for answers/judgements already obtained.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # mcp-Sophon/
SOPHON = os.environ.get("SOPHON_BIN", str(REPO / "sophon/target/release/sophon"))
CACHE_DIR = Path(os.environ.get("CORRECTNESS_CACHE", str(HERE / "results" / "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "correctness-bench", "version": "1"},
    },
}


# ---------------- Sophon RPC ----------------
def rpc(requests, timeout=180):
    """Run a batch of JSON-RPC requests through one `sophon serve` process.
    Returns {id: response}. INIT is prepended automatically."""
    payload = "".join(json.dumps(r) + "\n" for r in [INIT, *requests])
    p = subprocess.run(
        [SOPHON, "serve"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d.get("id")] = d
    return out


def call(name, args, rid):
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }


def extract(resp):
    """Pull the structured payload out of an MCP tools/call response."""
    if not resp:
        return None
    res = resp.get("result")
    if not res:
        return None
    if "structuredContent" in res:
        return res["structuredContent"]
    try:
        return json.loads(res["content"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def count_tokens(text, _cache={}):
    """Token count via Sophon's own tokenizer. Memoised within a process."""
    key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    if key in _cache:
        return _cache[key]
    e = extract(rpc([call("count_tokens", {"text": text}, 1)]).get(1)) or {}
    n = int(e.get("token_count", 0))
    _cache[key] = n
    return n


# ---------------- Compression wrappers ----------------
# Each returns (compressed_text, compressed_tokens): the exact text an agent
# would re-inject after running the tool, plus Sophon's own token count of it.


def compress_prompt(prompt, query, max_tokens):
    e = (
        extract(
            rpc(
                [
                    call(
                        "compress_prompt",
                        {"prompt": prompt, "query": query, "max_tokens": max_tokens},
                        1,
                    )
                ]
            ).get(1)
        )
        or {}
    )
    txt = e.get("compressed_prompt", prompt)
    return txt, int(e.get("token_count", count_tokens(txt)))


def compress_output(command, output):
    e = (
        extract(
            rpc(
                [call("compress_output", {"command": command, "output": output}, 1)]
            ).get(1)
        )
        or {}
    )
    txt = e.get("compressed", output)
    return txt, int(e.get("compressed_tokens", count_tokens(txt)))


def assemble_history(result):
    """compress_history returns a structured view (summary + stable_facts +
    recent verbatim messages). Reassemble it into the single text blob an
    agent actually re-injects as conversation context."""
    parts = []
    summary = (result.get("summary") or "").strip()
    if summary:
        parts.append("[CONVERSATION SUMMARY]\n" + summary)
    facts = result.get("stable_facts") or []
    if facts:
        body = "\n".join(
            f"- {f if isinstance(f, str) else f.get('text', json.dumps(f))}"
            for f in facts
        )
        parts.append("[STABLE FACTS]\n" + body)
    recent = result.get("recent_messages") or []
    if recent:
        body = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in recent
        )
        parts.append("[RECENT MESSAGES]\n" + body)
    return "\n\n".join(parts)


def compress_history(messages, max_tokens, recent_window=4, query=None):
    args = {
        "messages": messages,
        "max_tokens": max_tokens,
        "recent_window": recent_window,
    }
    # Query-aware history: when the caller knows the question, pass it so the
    # summary of the dropped tail is biased toward relevant lines.
    if query:
        args["query"] = query
    e = extract(rpc([call("compress_history", args, 1)]).get(1)) or {}
    txt = assemble_history(e)
    return txt, int(e.get("token_count", count_tokens(txt)))


# ---------------- Naive truncation baseline ----------------
def truncate_to_tokens(text, budget_tokens):
    """Head-truncate `text` to ~budget_tokens, the dumbest cap a naive agent
    applies. Estimates chars-per-token from the full count, then reports the
    ACTUAL token count so budget parity with Sophon is measured, not assumed."""
    if budget_tokens <= 0:
        return "", 0
    full = count_tokens(text)
    if full <= budget_tokens:
        return text, full
    cpt = max(1.0, len(text) / max(1, full))
    cut = text[: int(budget_tokens * cpt)]
    # one corrective pass: trim if we overshot the token budget
    for _ in range(3):
        n = count_tokens(cut)
        if n <= budget_tokens or len(cut) < 40:
            break
        cut = cut[: int(len(cut) * budget_tokens / max(1, n))]
    return cut, count_tokens(cut)


def truncate_to_tokens_tail(text, budget_tokens):
    """Keep the most RECENT ~budget_tokens (drop the start). This is the
    realistic naive baseline for a conversation: a sliding context window
    that retains recent turns and forgets old ones — exactly what
    compress_history claims to improve on by summarising the dropped tail."""
    if budget_tokens <= 0:
        return "", 0
    full = count_tokens(text)
    if full <= budget_tokens:
        return text, full
    cpt = max(1.0, len(text) / max(1, full))
    cut = text[-int(budget_tokens * cpt) :]
    for _ in range(3):
        n = count_tokens(cut)
        if n <= budget_tokens or len(cut) < 40:
            break
        cut = cut[-int(len(cut) * budget_tokens / max(1, n)) :]
    return cut, count_tokens(cut)


# ---------------- LLM shell-out with disk cache ----------------
def _cache_path(kind, model, prompt):
    h = hashlib.sha256(
        f"{kind}\x00{model}\x00{prompt}".encode("utf-8", "replace")
    ).hexdigest()
    return CACHE_DIR / f"{kind}_{model}_{h[:24]}.json"


def claude(prompt, model="haiku", timeout=240, kind="answer", use_cache=True):
    """Shell out to `claude -p --model M --output-format json`, return the
    `result` text. Cached on disk by (kind, model, prompt) for resumability.
    Returns None on failure (caller decides how to score a missing answer)."""
    cp = _cache_path(kind, model, prompt)
    if use_cache and cp.exists():
        try:
            return json.loads(cp.read_text())["result"]
        except (json.JSONDecodeError, KeyError):
            pass
    # Retry transient failures (rate limits / timeouts under parallel load).
    # Only successes are cached, so a failed cell is re-attempted on the next
    # run — but retrying in-process keeps a single pass reproducible.
    answer = None
    for _ in range(3):
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", model, "--output-format", "json"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            continue
        raw = proc.stdout.strip()
        answer = raw
        if raw.startswith("{"):
            try:
                env = json.loads(raw)
                if isinstance(env, dict) and "result" in env:
                    answer = str(env["result"]).strip()
            except json.JSONDecodeError:
                pass
        break
    if answer is None:
        return None
    if use_cache:
        cp.write_text(json.dumps({"result": answer}))
    return answer
