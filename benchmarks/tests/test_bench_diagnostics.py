"""Hermetic regression tests for benchmark transport and failure reporting."""
import io
import json
import queue
import unittest

from benchmarks.run_all_with_baseline import score_run
from benchmarks.session_scaling_curve import _send_and_read


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()

    def poll(self):
        return None


class BenchDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.proc = FakeProcess()
        self.request = {"jsonrpc": "2.0", "id": 42, "method": "tools/call",
                        "params": {"name": "update_memory", "arguments": {}}}
        self.responses = queue.Queue()

    def test_valid_mcp_response(self):
        self.responses.put(json.dumps({"jsonrpc": "2.0", "id": 42, "result": {}}))
        response, duration_ms = _send_and_read(
            self.proc, self.request, self.responses, timeout_seconds=0.1
        )
        self.assertEqual(response["id"], 42)
        self.assertGreaterEqual(duration_ms, 0)
        self.assertIn('"id": 42', self.proc.stdin.getvalue())

    def test_missing_response_reports_request(self):
        with self.assertRaisesRegex(TimeoutError, "update_memory.*id=42"):
            _send_and_read(self.proc, self.request, self.responses, timeout_seconds=0.01)

    def test_wrong_response_id_is_rejected(self):
        self.responses.put(json.dumps({"jsonrpc": "2.0", "id": 99, "result": {}}))
        with self.assertRaisesRegex(RuntimeError, "response ID mismatch"):
            _send_and_read(self.proc, self.request, self.responses, timeout_seconds=0.1)

    def test_mcp_error_is_reported(self):
        self.responses.put(json.dumps({"jsonrpc": "2.0", "id": 42,
                                       "error": {"code": -32603, "message": "boom"}}))
        with self.assertRaisesRegex(RuntimeError, "MCP error"):
            _send_and_read(self.proc, self.request, self.responses, timeout_seconds=0.1)

    def test_bench_error_includes_diagnostic_tail(self):
        regressions, _, _, _ = score_run(
            {"session_scaling_curve": {"_error": "exit 1",
                                       "_stderr_tail": "stalled update_memory id=42"}},
            {},
        )
        self.assertTrue(any("stalled update_memory id=42" in r for r in regressions))


if __name__ == "__main__":
    unittest.main()
