#!/usr/bin/env python3
"""
`compress_output` per-command coverage bench.

Sophon's output compressor ships 20+ command-aware filters (git /
cargo / docker / npm / kubectl / pytest / …). The aggregate number
published at v0.4.0 (~90 % savings on shell stdout) is a useful
headline but it hides the per-filter distribution: users want to
know *which of their commands* benefit most before wiring the
hook.

This bench runs canned, realistic outputs from 15 common dev
commands through `compress_output` and reports the ratio per
command. Inputs are hard-coded samples so the bench is
deterministic and network-free; they are modelled on typical
production-grade output (verbose, mixed with noise, not toy
one-liners).

What it measures
    raw_tokens          tokens in the original stdout/stderr
    compressed_tokens   tokens after compress_output
    saved_pct           1 − (compressed / raw)
    filter_name         which filter Sophon picked (empirical
                        validation that the dispatcher routed to
                        the right domain-aware filter)
    strategies_applied  which strategies Sophon applied (e.g.
                        `git_status`, `error_only`, `truncate`)

Output
    Text mode (default): per-command table + aggregate summary.
    JSON mode           (--json): one record per command.

Reproducibility: no shell commands are actually executed. Samples
are embedded in this file. Change a sample, re-run, the numbers
change only along that sample.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SOPHON = os.environ.get(
    "SOPHON_BIN",
    str(Path(__file__).resolve().parent.parent / "sophon/target/release/sophon"),
)


# ---------------- RPC helper ----------------
def _rpc_one(name: str, arguments: dict) -> dict:
    payload = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "compress-output-bench", "version": "0"},
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
    p = subprocess.run(
        [SOPHON, "serve"],
        input=raw,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id") == 1:
            return d.get("result", {})
    return {}


def run_compress_output(command: str, output: str) -> dict:
    return _rpc_one(
        "compress_output", {"command": command, "output": output}
    ).get("structuredContent", {}) or {}


def run_count_tokens(text: str) -> int:
    body = _rpc_one("count_tokens", {"text": text}).get("structuredContent", {}) or {}
    return int(body.get("token_count") or 0)


# ---------------- Sample outputs ----------------
# Each entry: (command, realistic stdout+stderr blob).
# Sizes roughly match what the typical dev sees day-to-day.

SAMPLES: list[tuple[str, str]] = [
    (
        "git status",
        "On branch feature/orthogonal-bench\n"
        "Your branch is up to date with 'origin/feature/orthogonal-bench'.\n\n"
        "Changes not staged for commit:\n"
        '  (use "git add <file>..." to update what will be committed)\n'
        '  (use "git restore <file>..." to discard changes in working directory)\n'
        + "\tmodified:   src/lib.rs\n" * 8
        + "\tmodified:   tests/integration.rs\n" * 3
        + "\nUntracked files:\n"
        '  (use "git add <file>..." to include in what will be committed)\n'
        + "\tbenchmarks/new_bench.py\n" * 2
        + '\nno changes added to commit (use "git add" and/or "git commit -a")\n',
    ),
    (
        "git diff HEAD~3",
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "index 1234abc..5678def 100644\n"
        "--- a/src/lib.rs\n"
        "+++ b/src/lib.rs\n"
        "@@ -10,6 +10,8 @@ pub fn foo() {\n"
        + "-    old_line();\n+    new_line();\n" * 10
        + "+    extra_line();\n\n"
        + ("diff --git a/src/bar.rs b/src/bar.rs\nindex aaa..bbb 100644\n"
           "--- a/src/bar.rs\n+++ b/src/bar.rs\n"
           "@@ -1,3 +1,4 @@\n+use std::io;\n fn main() {}\n") * 3,
    ),
    (
        "git log --oneline -n 20",
        "\n".join(
            f"{hex(0xdeadbeef + i)[2:10]} feat: bump version and add bench script {i}"
            for i in range(20)
        )
        + "\n",
    ),
    (
        "cargo test",
        "   Compiling sophon-core v0.5.0\n"
        "   Compiling memory-manager v0.5.0\n"
        "   Compiling semantic-retriever v0.5.0\n"
        "    Finished test profile [unoptimized + debuginfo] target(s) in 4.21s\n"
        "     Running unittests src/lib.rs\n"
        "running 79 tests\n"
        + "test chunker::tests::basic ... ok\n" * 78
        + "test store::tests::persistence_round_trip ... FAILED\n"
        "\nfailures:\n\n"
        "---- store::tests::persistence_round_trip stdout ----\n"
        "thread 'store::tests::persistence_round_trip' panicked at 'file not found'\n"
        "\nfailures:\n    store::tests::persistence_round_trip\n\n"
        "test result: FAILED. 78 passed; 1 failed; 0 ignored; finished in 0.19s\n"
        "error: test failed, to rerun pass -p semantic-retriever --lib\n",
    ),
    (
        "cargo build --release",
        "   Compiling serde v1.0.228\n"
        "   Compiling regex v1.11.3\n"
        + "   Compiling tokio v1.48.0\n" * 1
        + ("    Checking tracing v0.1\n" * 5)
        + "   Compiling mcp-integration v0.5.0\n"
        "    Finished `release` profile [optimized] target(s) in 43.21s\n",
    ),
    (
        "npm install",
        "npm warn deprecated inflight@1.0.6: This module is not supported\n"
        + "npm warn deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported\n" * 2
        + "\nadded 1337 packages, and audited 1423 packages in 8s\n"
        "\n218 packages are looking for funding\n"
        "  run `npm fund` for details\n"
        "\n7 vulnerabilities (3 low, 4 moderate)\n"
        "\nTo address all issues (including breaking changes), run:\n"
        "  npm audit fix --force\n"
        "\nRun `npm audit` for details.\n",
    ),
    (
        "pytest",
        "============================= test session starts ==============================\n"
        "platform darwin -- Python 3.13.1, pytest-8.3.4, pluggy-1.5.0\n"
        "rootdir: /home/user/project\n"
        "collected 412 items\n\n"
        + "tests/test_api.py " + "." * 58 + "  [ 14%]\n"
        + "tests/test_core.py " + "." * 128 + "     [ 45%]\n"
        + "tests/test_integration.py " + "." * 89 + "F" + "." * 12 + "  [ 70%]\n"
        + "tests/test_utils.py " + "." * 124 + "     [100%]\n"
        "\n=================================== FAILURES ===================================\n"
        "_________________ test_integration.py::test_webhook_signature __________________\n"
        "E   AssertionError: expected HMAC-SHA256 but got HMAC-SHA1\n"
        "tests/test_integration.py:87: AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_integration.py::test_webhook_signature\n"
        "======================== 1 failed, 411 passed in 12.47s ========================\n",
    ),
    (
        "docker ps",
        (lambda entrypoint='"/entrypoint.sh"': (
            "CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                    NAMES\n"
            + "".join(
                f"{hex(0xabc0000 + i)[2:14]:<14} {f'app-{i}:v1.2.3':<20} "
                f"{entrypoint:<24} {'2 hours ago':<15} "
                f"{'Up 2 hours':<14} {f'0.0.0.0:{8000 + i}->8000/tcp':<24} {f'app_{i}_1'}\n"
                for i in range(12)
            )
        ))(),
    ),
    (
        "docker logs app_1",
        "\n".join(
            f"2026-04-20T{hour:02d}:{minute:02d}:{second:02d}.{micro:06d}Z INFO  "
            f"request.handled method=GET path=/api/v1/resource status=200 duration_ms=12"
            for hour in range(14, 20)
            for minute in (0, 15, 30, 45)
            for second in (1, 31)
            for micro in (123456, 654321)
        )
        + "\n2026-04-20T19:12:43.001122Z ERROR upstream.timeout retrying in 5s\n"
        + "2026-04-20T19:12:48.123456Z INFO  upstream.recovered\n",
    ),
    (
        "kubectl get pods -A",
        "NAMESPACE    NAME                             READY   STATUS    RESTARTS   AGE\n"
        + "".join(
            f"{ns:<12} {f'pod-{i}-7f8d9cb6-abc12':<32} "
            f"1/1     Running   {i % 3:<9}  "
            f"{(i // 4) + 1}d\n"
            for ns in ("default", "kube-system", "monitoring")
            for i in range(8)
        ),
    ),
    (
        "make",
        "make: Entering directory '/home/user/project'\n"
        "cc -c -O2 -Wall src/main.c -o build/main.o\n"
        + "cc -c -O2 -Wall src/module_{n}.c -o build/module_{n}.o\n".replace(
            "{n}", "1"
        ) * 10
        + "ld -o build/project.bin build/main.o build/module_*.o\n"
        "make: Leaving directory '/home/user/project'\n",
    ),
    (
        "curl -v https://api.example.com/v1/users",
        "*   Trying 93.184.216.34:443...\n"
        "* Connected to api.example.com (93.184.216.34) port 443 (#0)\n"
        "* ALPN: offers h2,http/1.1\n"
        "* TLSv1.3 (OUT), TLS handshake, Client hello (1):\n"
        + "* TLSv1.3 (IN), TLS handshake, Server hello (2):\n"
        + "* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384\n"
        + "* Server certificate:\n*  subject: CN=api.example.com\n"
        + "> GET /v1/users HTTP/2\n> Host: api.example.com\n"
        + "< HTTP/2 200 \n< content-type: application/json\n"
        + '{\n  "data": [\n    {"id": 1, "name": "Alice"},\n    {"id": 2, "name": "Bob"}\n  ]\n}\n'
        + "* Connection #0 to host api.example.com left intact\n",
    ),
    (
        "tail -n 200 app.log",
        "\n".join(
            f"2026-04-20 19:{minute:02d}:00 INFO  request processed id={i:06d} ms=15"
            for minute in range(10, 55)
            for i in range(minute * 10, minute * 10 + 5)
        )
        + "\n2026-04-20 19:45:12 WARN  rate-limit approaching quota=85%\n"
        + "2026-04-20 19:46:03 ERROR  database connection reset\n",
    ),
    (
        "grep -r TODO src/",
        "\n".join(
            f"src/module_{i}/file_{j}.rs:{line}:    // TODO: refactor this"
            for i in range(6)
            for j in range(5)
            for line in (12, 45, 89)
        )
        + "\n",
    ),
    (
        "ls -la /var/log",
        "total 487360\n"
        "drwxr-xr-x  40 root  wheel      1280 Apr 20 19:00 .\n"
        "drwxr-xr-x  26 root  wheel       832 Apr 18 16:00 ..\n"
        + "".join(
            f"-rw-r-----  1 root  admin  {(i * 1024 * 17) % 10_000_000:>8} "
            f"Apr 20 19:{(i * 3) % 60:02d} app-{i}.log\n"
            for i in range(30)
        ),
    ),
    # ---- JSON-heavy samples (added v0.5.3 to exercise JsonStructural) ----
    (
        "kubectl get pods -o json",
        json.dumps(
            {
                "kind": "List",
                "apiVersion": "v1",
                "items": [
                    {
                        "kind": "Pod",
                        "apiVersion": "v1",
                        "metadata": {
                            "name": f"app-deployment-{i}-{abs(hash(i)) % 100000:05d}",
                            "namespace": "production" if i % 3 else "staging",
                            "uid": f"abc-{i}-def-{abs(hash(i)) % 99999}",
                            "resourceVersion": str(1000000 + i * 17),
                            "labels": {
                                "app": f"service-{i % 8}",
                                "version": f"v1.{i % 12}.{i % 30}",
                                "tier": "backend",
                            },
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": f"registry.example.com/service-{i % 8}:v1.{i % 12}.{i % 30}",
                                    "resources": {
                                        "requests": {"cpu": "100m", "memory": "256Mi"},
                                        "limits": {"cpu": "500m", "memory": "1Gi"},
                                    },
                                }
                            ],
                            "nodeName": f"node-{i % 12}",
                        },
                        "status": {
                            "phase": "Running",
                            "podIP": f"10.0.{i // 8}.{i % 256}",
                            "startTime": "2026-04-20T19:00:00Z",
                        },
                    }
                    for i in range(50)
                ],
            },
            indent=2,
        ),
    ),
    (
        "aws s3api list-objects-v2 --bucket data-archive --output json",
        json.dumps(
            {
                "Contents": [
                    {
                        "Key": f"archive/2026/04/data-{i:06d}.parquet",
                        "LastModified": "2026-04-20T19:00:00.000Z",
                        "ETag": f'"{abs(hash(i)) % (10**32):032x}"',
                        "Size": (i + 1) * 1024 * 1024 + (i * 7) % 9999,
                        "StorageClass": "STANDARD_IA" if i % 2 else "STANDARD",
                        "Owner": {
                            "DisplayName": "data-team",
                            "ID": "abc123def456" + str(i % 100),
                        },
                    }
                    for i in range(80)
                ],
                "Name": "data-archive",
                "Prefix": "archive/2026/04/",
                "MaxKeys": 1000,
                "EncodingType": "url",
                "KeyCount": 80,
                "IsTruncated": False,
            },
            indent=2,
        ),
    ),
    (
        "gh api repos/lacausecrypto/mcp-sophon/issues",
        json.dumps(
            [
                {
                    "url": f"https://api.github.com/repos/lacausecrypto/mcp-sophon/issues/{i}",
                    "id": 1000000 + i,
                    "node_id": f"I_kw{abs(hash(i)) % (10**14):014d}",
                    "number": i,
                    "title": f"Issue {i}: tracking refactor of module_{i % 12}",
                    "user": {
                        "login": f"contributor-{i % 7}",
                        "id": 50000 + (i % 7),
                        "type": "User",
                    },
                    "state": "open" if i % 4 else "closed",
                    "comments": i % 15,
                    "created_at": "2026-04-15T10:00:00Z",
                    "updated_at": "2026-04-20T15:00:00Z",
                    "body": f"Description for issue {i}. Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3,
                }
                for i in range(40)
            ],
            indent=2,
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--only", type=str, help="Run only the command whose substring matches."
    )
    args = parser.parse_args()

    if not Path(SOPHON).exists():
        print(f"sophon binary not found at {SOPHON!r}. Set SOPHON_BIN.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for cmd, blob in SAMPLES:
        if args.only and args.only not in cmd:
            continue
        raw_tokens = run_count_tokens(blob)
        result = run_compress_output(cmd, blob)
        compressed_tokens = int(
            result.get("compressed_tokens") or result.get("token_count") or 0
        )
        saved = (raw_tokens - compressed_tokens) / raw_tokens if raw_tokens else 0.0
        rows.append(
            {
                "command": cmd,
                "filter_name": result.get("filter_name"),
                "strategies_applied": result.get("strategies_applied"),
                "raw_tokens": raw_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_pct": saved,
                "raw_chars": len(blob),
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    saved_values = [r["saved_pct"] for r in rows if r["raw_tokens"]]
    agg_raw = sum(r["raw_tokens"] for r in rows)
    agg_sent = sum(r["compressed_tokens"] for r in rows)
    weighted = (agg_raw - agg_sent) / agg_raw if agg_raw else 0.0
    mean_saved = statistics.mean(saved_values) if saved_values else 0.0
    median_saved = statistics.median(saved_values) if saved_values else 0.0

    print("=" * 96)
    print(f"compress_output per-command coverage — {len(rows)} commands")
    print("=" * 96)
    print(
        f"  {'command':<40} {'filter':<14} {'raw':>6} {'out':>5} {'saved%':>7}"
    )
    print(f"  {'-' * 40} {'-' * 14} {'-' * 6} {'-' * 5} {'-' * 7}")
    for r in sorted(rows, key=lambda x: -x["saved_pct"]):
        filter_name = r["filter_name"] or "(generic)"
        print(
            f"  {r['command'][:39]:<40} "
            f"{filter_name[:13]:<14} "
            f"{r['raw_tokens']:>6} "
            f"{r['compressed_tokens']:>5} "
            f"{r['saved_pct'] * 100:>6.1f}%"
        )

    print()
    print(
        f"  Aggregate (weighted by raw tokens):  "
        f"{agg_raw:>6} → {agg_sent:<6}   {weighted * 100:>5.1f} % saved"
    )
    print(f"  Mean per-command saved                                  {mean_saved * 100:>5.1f} %")
    print(f"  Median per-command saved                                {median_saved * 100:>5.1f} %")
    print()
    print("  Strategies applied across the run:")
    from collections import Counter

    strat_counter: Counter[str] = Counter()
    for r in rows:
        for s in r["strategies_applied"] or []:
            strat_counter[s] += 1
    for s, n in strat_counter.most_common():
        print(f"    {s:<30} {n:>3}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
