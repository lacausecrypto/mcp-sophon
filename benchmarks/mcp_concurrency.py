#!/usr/bin/env python3
"""Stdio concurrency and cancellation regression driver for Sophon MCP.

Runs real JSON-RPC requests against the release binary. No external Python
dependencies or LLM service are required. Prints p50/p95/p99 response latency
and verifies shared-memory integrity, cancellation and process liveness.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class Client:
    def __init__(self, binary: Path, max_workers: int = 8):
        env = os.environ.copy()
        env["SOPHON_MAX_CONCURRENT_TOOLS"] = str(max_workers)
        env["SOPHON_NO_LLM_SUMMARY"] = "1"
        # Keep test state hermetic.
        for name in ("SOPHON_MEMORY_PATH", "SOPHON_RETRIEVER_PATH",
                     "SOPHON_GRAPH_MEMORY", "SOPHON_GRAPH_MEMORY_PATH",
                     "SOPHON_ROLLING_SUMMARY", "SOPHON_ALLOWED_ROOT"):
            env.pop(name, None)
        self.proc = subprocess.Popen(
            [str(binary.resolve()), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, bufsize=0,
        )
        self.received: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            if raw.strip():
                try:
                    self.received.put(json.loads(raw))
                except ValueError as exc:
                    self.received.put({"parse_error": str(exc), "raw": repr(raw)})

    def send(self, messages):
        assert self.proc.stdin is not None
        payload = b"".join((json.dumps(m, separators=(",", ":")) + "\n").encode()
                           for m in messages)
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def collect(self, expected_ids, deadline_seconds=45):
        deadline = time.monotonic() + deadline_seconds
        responses = {}
        expected = set(expected_ids)
        while not expected.issubset(responses):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"Timeout/deadlock waiting for {sorted(expected - responses.keys())}; "
                    f"process={self.proc.poll()}"
                )
            try:
                result = self.received.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise AssertionError(f"Sophon exited unexpectedly: {self.proc.returncode}")
                continue
            if "parse_error" in result:
                raise AssertionError(f"Invalid JSON-RPC output: {result}")
            request_id = result.get("id")
            if request_id in responses:
                raise AssertionError(f"Duplicate response id={request_id}")
            responses[request_id] = result
        return responses

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
            self.proc.stdin = None
        try:
            self.proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise AssertionError("Sophon did not drain workers on EOF (deadlock)")
        if self.proc.returncode != 0:
            raise AssertionError(f"Sophon exited with code {self.proc.returncode}")


def call(request_id, name, arguments):
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}


def percentile(values, pct):
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(pct * len(ordered)) - 1)], 3)


def check_ok(response):
    assert "error" not in response, response
    result = response["result"]
    assert result.get("isError") is False, response
    return result["structuredContent"]


def benchmark(client, width, samples=64):
    assert samples % width == 0
    timings = []
    payload = "Sophon bounded cancellation and parallel tool execution. " * 80
    for wave in range(samples // width):
        messages = []
        start = {}
        for i in range(width):
            request_id = 1000 + width * 100 + wave * width + i
            if i % 2:
                message = call(request_id, "compress_output", {
                    "command": "cargo test",
                    "output": "test run finished successfully\n" * 80})
            else:
                message = call(request_id, "count_tokens", {"text": payload})
            start[request_id] = time.perf_counter()
            messages.append(message)
        client.send(messages)
        # Drain responses one-by-one to timestamp their actual arrival.
        remaining = set(start)
        deadline = time.monotonic() + 45
        while remaining:
            assert time.monotonic() < deadline, f"Deadlock: {remaining}"
            try:
                response = client.received.get(timeout=0.5)
            except queue.Empty:
                assert client.proc.poll() is None, "Sophon exited during load"
                continue
            request_id = response.get("id")
            assert request_id in remaining, f"unexpected/duplicate id {request_id}"
            remaining.remove(request_id)
            check_ok(response)
            timings.append((time.perf_counter() - start[request_id]) * 1000)
    return {"concurrency": width, "samples": samples,
            "p50_ms": percentile(timings, 0.50),
            "p95_ms": percentile(timings, 0.95),
            "p99_ms": percentile(timings, 0.99)}


def shared_state_test(client):
    client.send([call(4000, "update_memory", {"reset": True, "return_snapshot": False})])
    assert check_ok(client.collect([4000])[4000])["history_len"] == 0
    client.send([call(i, "update_memory", {
        "messages": [{"role": "user", "content": f"concurrency marker {i}"}],
        "return_snapshot": False,
    }) for i in range(4001, 4017)])
    responses = client.collect(range(4001, 4017))
    lengths = sorted(check_ok(value)["history_len"] for value in responses.values())
    assert lengths == list(range(1, 17)), f"Lost/duplicated memory writes: {lengths}"
    client.send([call(4017, "update_memory", {"return_snapshot": False})])
    assert check_ok(client.collect([4017])[4017])["history_len"] == 16
    return {"writes": 16, "history_len": 16, "integrity": "pass"}



def mixed_workload_test(binary):
    """Pure queries must be served while serialised memory jobs are queued."""
    client = Client(binary, max_workers=4)
    try:
        large = [{"role": "user", "content": f"mixed {i} " + "context " * 80}
                 for i in range(120)]
        stateful = [call(7000 + i, "compress_history", {
            "messages": large, "max_tokens": 500}) for i in range(16)]
        pure = [call(7100 + i, "count_tokens", {"text": "fast-path " * 30})
                for i in range(4)]
        client.send(stateful + pure)
        responses = client.collect([*range(7000, 7016), *range(7100, 7104)],
                                   deadline_seconds=60)
        for response in responses.values():
            check_ok(response)
        completion_order = list(responses)
        last_stateful = max(completion_order.index(i) for i in range(7000, 7016))
        first_pure = min(completion_order.index(i) for i in range(7100, 7104))
        assert first_pure < last_stateful, (
            f"Pure tools starved behind serialised tools: {completion_order}")
        return {"queued_stateful": 16, "pure": 4, "pure_serviced_early": True}
    finally:
        client.close()



def navigator_and_stats_test(binary):
    """Concurrent navigator requests must not lose the shared cache or stats."""
    with tempfile.TemporaryDirectory(prefix="sophon-nav-ci-") as directory:
        root = Path(directory)
        for i in range(100):
            (root / f"module_{i:03d}.rs").write_text(
                f"pub fn symbol_{i}() -> usize {{ {i} }}\\n"
                f"// cross-reference symbol_{(i + 1) % 100}\\n",
                encoding="utf-8",
            )
        client = Client(binary, max_workers=4)
        try:
            long_history = [
                {"role": "user", "content": f"history {i}: " + "context " * 90}
                for i in range(140)
            ]
            calls = [
                call(8000 + i, "compress_history",
                     {"messages": long_history, "max_tokens": 500})
                for i in range(8)
            ]
            calls.extend([
                call(8100, "navigate_codebase",
                     {"root": str(root), "query": "symbol_37", "max_tokens": 512}),
                call(8101, "get_token_stats", {}),
            ])
            client.send(calls)
            responses = client.collect([*range(8000, 8008), 8100, 8101],
                                       deadline_seconds=90)
            for response in responses.values():
                check_ok(response)

            first = check_ok(responses[8100])
            assert first["scan_result"]["kind"] == "fresh", first["scan_result"]

            # Consecutive scans without a root must reuse the same cache,
            # even though the stdio navigator lives outside SophonServer.
            client.send([
                call(8102, "navigate_codebase", {"query": "symbol_38"}),
                call(8103, "navigate_codebase", {"query": "symbol_39"}),
            ])
            cached = client.collect([8102, 8103], deadline_seconds=60)
            for response in cached.values():
                scan = check_ok(response)["scan_result"]
                assert scan["kind"] == "incremental", scan
                assert scan["updated"] == 0 and scan["added"] == 0, scan

            # Verify the separate stat collector observes all three scans and
            # returns totals that match the exact same module snapshot.
            client.send([call(8104, "get_token_stats", {})])
            snapshot = check_ok(client.collect([8104])[8104])
            modules = snapshot["modules"]
            totals = snapshot["totals"]
            assert modules["codebase_navigator"]["calls"] == 3, modules
            assert totals["calls"] == sum(x["calls"] for x in modules.values())
            for key in ("original_tokens", "compressed_tokens"):
                assert totals[key] == sum(x[key] for x in modules.values())

            # Invalid paths produce tool errors without erasing the cache.
            client.send([call(8105, "navigate_codebase",
                              {"root": str(root / "missing-dir")})])
            invalid = client.collect([8105])[8105]
            assert invalid["result"]["isError"] is True, invalid
            client.send([call(8106, "navigate_codebase", {"query": "symbol_40"})])
            recovered = check_ok(client.collect([8106])[8106])
            assert recovered["scan_result"]["kind"] == "incremental", recovered

            return {
                "files": 100,
                "concurrent_memory_calls": 8,
                "navigator_scans": 4,
                "cache_reused": True,
                "statistics_consistent": True,
                "invalid_path_recovered": True,
            }
        finally:
            client.close()


def cancellation_test(binary):
    client = Client(binary, max_workers=1)
    try:
        large = [{"role": "user", "content": f"large message {i} " + ("text " * 60)}
                 for i in range(100)]
        jobs = [call(6000 + i, "compress_history", {
            "messages": large, "max_tokens": 500}) for i in range(16)]
        client.send(jobs)
        client.send([{"jsonrpc": "2.0", "method": "notifications/cancelled",
                      "params": {"requestId": 6015, "reason": "queued cancellation"}},
                     call(6100, "count_tokens", {"text": "still alive"})])
        # The last expensive request is queued behind the other 15 under
        # concurrency=1; cancellation must prevent its response.
        replies = client.collect([*range(6000, 6015), 6100], deadline_seconds=60)
        assert 6015 not in replies, "Cancelled queued job unexpectedly responded"
        assert check_ok(replies[6100])["token_count"] > 0
        return {"cancelled_request_id": 6015, "followup": "pass",
                "queued_response_dropped": True, "deadlock": False}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=Path("sophon/target/release/sophon"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    client = Client(args.binary)
    try:
        client.send([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
        assert "result" in client.collect([1])[1]
        client.send([call(2, "count_tokens", {"text": "warmup"})])
        check_ok(client.collect([2])[2])
        report = {"latency": [benchmark(client, n) for n in (1, 4, 16)],
                  "shared_state": shared_state_test(client)}
    finally:
        client.close()
    report["cancellation"] = cancellation_test(args.binary)
    report["mixed_workload"] = mixed_workload_test(args.binary)
    report["navigator_and_stats"] = navigator_and_stats_test(args.binary)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
