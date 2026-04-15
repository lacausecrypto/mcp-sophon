#!/usr/bin/env python3
"""
Head-to-head: Sophon compress_prompt vs LLMLingua-2 on identical inputs.
Reproduces BENCHMARK.md § 7.8.d.

Fair comparison principles:
  * Same raw inputs for both systems (structured XML + long docs).
  * Measured: input tokens, output tokens, ratio, wall-clock latency.
  * LLMLingua-2 with its published default (rate=0.5 → target 50% retention)
    and rate=0.33 (target 3x compression) — Sophon's compress_prompt runs
    with its default max_tokens.
  * Tokenizer: tiktoken cl100k_base for BOTH systems, apples-to-apples.

Usage:
  SOPHON_BIN=./sophon/target/release/sophon \
  SOPHON_REPO_ROOT=. \
      python3 benchmarks/llmlingua_compare.py

Prereqs:
  pip install llmlingua tiktoken
  cargo build --release -p mcp-integration

Output: llmlingua_results.json in the current directory.
"""
import json, os, subprocess, sys, time
from pathlib import Path

SOPHON = os.environ.get("SOPHON_BIN", "sophon")
REPO_ROOT = Path(os.environ.get("SOPHON_REPO_ROOT", "."))
FIXTURE_DIR = Path(os.environ.get("SOPHON_BENCH_DIR", "./benchmarks/data"))
OUT_PATH = Path(os.environ.get("LLMLINGUA_OUT", "llmlingua_results.json"))

# ---- inputs ------------------------------------------------------
# Mix of structured XML prompts + long technical content. We keep it
# modest (4 items) so LLMLingua (which loads ~280 MB XLM-RoBERTa on
# CPU) doesn't take all day.

def read(p: Path) -> str:
    return Path(p).read_text()

INPUTS = [
    {
        "name": "structured_xml",
        "text": read(FIXTURE_DIR / "system_prompt_large.txt"),
        "query": "fix a Python off-by-one bug in a Fibonacci function",
    },
    {
        "name": "structured_xml_q2",
        "text": read(FIXTURE_DIR / "system_prompt_large.txt"),
        "query": "write a portable bash one-liner to find the 10 largest files",
    },
    {
        "name": "long_readme",
        "text": read(REPO_ROOT / "sophon" / "README.md"),
        "query": "how do I enable the tree-sitter backend",
    },
    {
        "name": "bench_doc",
        "text": read(REPO_ROOT / "BENCHMARK.md")[:20000],
        "query": "what is the recall at 5 for the tree-sitter backend",
    },
]

# ---- tiktoken (shared) -------------------------------------------
import tiktoken
TIK = tiktoken.get_encoding("cl100k_base")

def toks(text): return len(TIK.encode(text))

# ---- Sophon ------------------------------------------------------
INIT = {"jsonrpc":"2.0","id":0,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
                  "clientInfo":{"name":"bench","version":"0"}}}

def call(name, args, rid):
    return {"jsonrpc":"2.0","id":rid,"method":"tools/call",
            "params":{"name":name,"arguments":args}}

def sophon_compress(text, query):
    args = {"prompt": text, "query": query}
    requests = [INIT, call("compress_prompt", args, 1)]
    payload = "".join(json.dumps(r)+"\n" for r in requests)
    t0 = time.perf_counter()
    p = subprocess.run([SOPHON, "serve"], input=payload, capture_output=True,
                       text=True, timeout=120)
    lat_ms = (time.perf_counter()-t0)*1000
    for line in p.stdout.splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        if d.get("id") == 1:
            sc = d.get("result",{}).get("structuredContent",{})
            return sc.get("compressed_prompt",""), lat_ms
    return "", lat_ms

# ---- LLMLingua-2 -------------------------------------------------
print("loading LLMLingua-2 (XLM-RoBERTa-base, ~280 MB on first run)...",
      file=sys.stderr, flush=True)
from llmlingua import PromptCompressor
LL = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",
)

def llmlingua_compress(text, rate=0.5):
    t0 = time.perf_counter()
    r = LL.compress_prompt([text], rate=rate, force_tokens=['\n'])
    lat_ms = (time.perf_counter()-t0)*1000
    return r["compressed_prompt"], lat_ms

# ---- run ---------------------------------------------------------
rows = []
for item in INPUTS:
    name, text, query = item["name"], item["text"], item["query"]
    in_tok = toks(text)
    print(f"\n== {name} ({in_tok} tokens in, query='{query[:40]}...') ==", flush=True)

    sc, slat = sophon_compress(text, query)
    s_out = toks(sc)

    lc50, llat50 = llmlingua_compress(text, rate=0.5)
    l50_out = toks(lc50)

    lc33, llat33 = llmlingua_compress(text, rate=0.33)
    l33_out = toks(lc33)

    row = {
        "name": name,
        "input_tokens": in_tok,
        "sophon": {
            "out_tokens": s_out,
            "ratio": round(s_out/in_tok, 3) if in_tok else 0,
            "saved_pct": round(100*(1-s_out/in_tok), 1) if in_tok else 0,
            "latency_ms": round(slat, 1),
        },
        "llmlingua2_rate050": {
            "out_tokens": l50_out,
            "ratio": round(l50_out/in_tok, 3) if in_tok else 0,
            "saved_pct": round(100*(1-l50_out/in_tok), 1) if in_tok else 0,
            "latency_ms": round(llat50, 1),
        },
        "llmlingua2_rate033": {
            "out_tokens": l33_out,
            "ratio": round(l33_out/in_tok, 3) if in_tok else 0,
            "saved_pct": round(100*(1-l33_out/in_tok), 1) if in_tok else 0,
            "latency_ms": round(llat33, 1),
        },
    }
    rows.append(row)
    print(f"  sophon:            {s_out:>5} tok  ({row['sophon']['saved_pct']:>5.1f}%) in {slat:>6.0f} ms")
    print(f"  llmlingua2 r=0.5:  {l50_out:>5} tok  ({row['llmlingua2_rate050']['saved_pct']:>5.1f}%) in {llat50:>6.0f} ms")
    print(f"  llmlingua2 r=0.33: {l33_out:>5} tok  ({row['llmlingua2_rate033']['saved_pct']:>5.1f}%) in {llat33:>6.0f} ms")

# ---- summary ------------------------------------------------------
print("\n" + "="*90)
print(f"{'Input':<24}{'in_tok':>8}{'sophon':>16}{'LL-2 r=0.5':>16}{'LL-2 r=0.33':>16}")
print("-"*90)
for r in rows:
    print(f"{r['name']:<24}{r['input_tokens']:>8}"
          f"{r['sophon']['saved_pct']:>11.1f}%"
          f"{r['sophon']['latency_ms']:>5.0f}ms"
          f"{r['llmlingua2_rate050']['saved_pct']:>11.1f}%"
          f"{r['llmlingua2_rate050']['latency_ms']:>5.0f}ms"
          f"{r['llmlingua2_rate033']['saved_pct']:>11.1f}%"
          f"{r['llmlingua2_rate033']['latency_ms']:>5.0f}ms")
print("-"*90)

def mean(key, sub):
    vals = [r[sub][key] for r in rows if r[sub][key] is not None]
    return sum(vals)/len(vals) if vals else 0

print(f"\nMEAN saved:  sophon {mean('saved_pct','sophon'):5.1f}%   "
      f"LL-2 r=0.5 {mean('saved_pct','llmlingua2_rate050'):5.1f}%   "
      f"LL-2 r=0.33 {mean('saved_pct','llmlingua2_rate033'):5.1f}%")
print(f"MEAN latency ms: sophon {mean('latency_ms','sophon'):5.0f}   "
      f"LL-2 r=0.5 {mean('latency_ms','llmlingua2_rate050'):5.0f}   "
      f"LL-2 r=0.33 {mean('latency_ms','llmlingua2_rate033'):5.0f}")

OUT_PATH.write_text(json.dumps(rows, indent=2))
print(f"\nsaved to {OUT_PATH}")
