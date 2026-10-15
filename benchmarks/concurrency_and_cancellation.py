#!/usr/bin/env python3
"""MCP concurrency/latency smoke benchmark (real stdio process).

Measures individual RPC completion latencies at 1/4/16 in-flight requests.
Asserts completion/no duplicate replies, per-module memory consistency, and
liveness following queued cancellations. Timing is informational, not a speed
claim: it depends on runner CPU, filesystem and command mix.

Usage: SOPHON_BIN=sophon/target/release/sophon python3 benchmarks/concurrency_and_cancellation.py --json
"""
import argparse
import json
import math
import os
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = os.getenv("SOPHON_BIN", str(ROOT / "sophon/target/release/sophon"))
TIMEOUT = 40.0


class Client:
    def __init__(self, max_workers=8):
        env = os.environ.copy()
        env.update(SOPHON_MAX_CONCURRENT_TOOLS=str(max_workers))
        # A CI bench should never invoke external models or use user storage.
        for key in ("SOPHON_LLM_CMD", "SOPHON_RETRIEVER_PATH",
                    "SOPHON_MEMORY_PATH", "SOPHON_GRAPH_MEMORY",
                    "SOPHON_FACT_CARDS", "SOPHON_ROLLING_SUMMARY"):
            env.pop(key, None)
        self.proc = subprocess.Popen(
            [BIN, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
        )
        self.messages = queue.Queue()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        self.send({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "load-bench", "version": "1"}}})
        self.expect(0)

    def _reader(self):
        for line in self.proc.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                self.messages.put({"error": {"message": "invalid JSON-RPC stdout"}})

    def send(self, msg):
        assert self.proc.poll() is None, f"Sophon exited {self.proc.returncode}"
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def call(self, id_, name, **arguments):
        self.send({"jsonrpc": "2.0", "id": id_, "method": "tools/call",
                   "params": {"name": name, "arguments": arguments}})

    def expect(self, id_=None):
        item = self.messages.get(timeout=TIMEOUT)
        if id_ is not None:
            assert item.get("id") == id_, f"Expected id={id_}: {item}"
        assert "error" not in item, f"Server error: {item}"
        assert not item.get("result", {}).get("isError"), f"Tool failure: {item}"
        return item

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise AssertionError("MCP server deadlocked on stdin EOF")
        assert self.proc.returncode == 0, f"exit code {self.proc.returncode}"


def workload(i):
    group = i % 4
    if group == 0:
        return "count_tokens", {"text": ("tokenisation for simultaneous requests. " * 180)}
    if group == 1:
        return "compress_output", {
            "command": "git status",
            "output": "\n".join(f"modified: src/project/module_{j % 35}.rs" for j in range(280)),
        }
    if group == 2:
        return "compress_prompt", {
            "prompt": "\n".join(f"## section-{j}\nImplement async worker {j}; cancellation is mandatory." for j in range(40)),
            "query": "async cancellation worker", "max_tokens": 400,
        }
    return "compress_history", {
        "messages": [{"role": "user" if j % 2 == 0 else "assistant",
                      "content": f"Turn {j}, load test, ensure race-free memory state and cancellation."}
                     for j in range(32)],
        "max_tokens": 450,
    }


def percentile(v, p):
    a = sorted(v)
    return round(a[max(0, math.ceil(p * len(a)) - 1)] * 1000, 3)


def measure(window, total=64):
    client = Client(max_workers=window)
    start = time.perf_counter()
    pending = {}
    samples = []
    next_id = 1
    try:
        while len(samples) < total:
            while len(pending) < window and next_id <= total:
                name, args = workload(next_id)
                sent_at = time.perf_counter()
                client.call(next_id, name, **args)
                pending[next_id] = sent_at
                next_id += 1
            response = client.expect()
            id_ = response["id"]
            assert id_ in pending, f"Unexpected or duplicate response id={id_}"
            samples.append(time.perf_counter() - pending.pop(id_))
        assert not pending
        return {
            "concurrency": window, "requests": total,
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99),
            "throughput_rps": round(total / (time.perf_counter() - start), 2),
        }
    finally:
        client.close()


def integrity():
    client = Client(max_workers=16)
    try:
        client.call(1, "update_memory", reset=True, messages=[], return_snapshot=False)
        client.expect(1)
        for i in range(2, 18):
            client.call(i, "update_memory",
                        messages=[{"role": "user", "content": f"isolated-{i}"}],
                        return_snapshot=False)
        seen = []
        for _ in range(16):
            msg = client.expect()
            assert msg["id"] in range(2, 18)
            seen.append(msg["result"]["content"])
        # MCP wraps results as content blocks; parse whichever shape is available.
        lengths = []
        for content in seen:
            if isinstance(content, list):
                block = content[0]["text"]
                lengths.append(json.loads(block)["history_len"])
            else:
                lengths.append(content["history_len"])
        assert sorted(lengths) == list(range(1, 17)), lengths
        client.call(20, "get_token_stats", period="session")
        client.expect(20)
        return {"parallel_updates": 16, "history_lengths_consistent": True}
    finally:
        client.close()


def cancellation():
    client = Client(max_workers=1)
    try:
        # Block the only execution permit with a nontrivial request.
        _, args = workload(3)
        args["messages"] *= 25
        client.call(100, "compress_history", **args)
        # These requests should be cancellable while queued, even if
        # they race the fast worker. Only the noncancelled sentinel is
        # required to complete; a response which won a race is permitted.
        for id_ in range(101, 117):
            client.call(id_, "count_tokens", text="cancel this queued request " * 1000)
        for id_ in range(101, 117):
            client.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                         "params": {"requestId": id_}})
        client.call(200, "count_tokens", text="after all cancellations")
        responses = {}
        deadline = time.monotonic() + TIMEOUT
        while 200 not in responses:
            item = client.messages.get(timeout=max(.001, deadline - time.monotonic()))
            i = item.get("id")
            assert i not in responses, f"duplicate response {i}"
            responses[i] = item
        assert "error" not in responses[200], responses[200]
        # The pending registry must tolerate unknown/repeated cancels.
        client.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                     "params": {"requestId": 999999}})
        return {"cancellations_sent": 16, "post_cancel_liveness": True,
                "cancelled_replies_that_won_race": len(set(range(101, 117)) & responses.keys())}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--requests", type=int, default=64)
    args = parser.parse_args()
    assert args.requests >= 16
    results = {"loads": [measure(n, args.requests) for n in (1, 4, 16)],
               "integrity": integrity(), "cancellation": cancellation()}
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results["loads"]:
            print(f"{row['concurrency']:>2} in-flight: p50={row['p50_ms']}ms "
                  f"p95={row['p95_ms']}ms p99={row['p99_ms']}ms "
                  f"throughput={row['throughput_rps']} req/s")
        print(results["integrity"], results["cancellation"])


if __name__ == "__main__":
    main()
