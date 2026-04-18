#!/usr/bin/env python3
"""
Universal LLM CLI wrapper — takes a prompt on stdin, runs any supported
provider, emits the clean answer text on stdout. Normalises the many
output-shape quirks so bench harnesses (and Sophon itself) can treat
every provider the same way.

Usage:
    echo "prompt" | python3 llm_cli.py --provider claude --model sonnet
    echo "prompt" | python3 llm_cli.py --provider codex
    echo "prompt" | python3 llm_cli.py --provider claude --model haiku --json

Providers:
    claude : shells out to `claude -p --model <M> --output-format json`
             and extracts the `result` field. Default for Anthropic.
    codex  : shells out to `codex exec --ephemeral --output-last-message <f>`
             and returns the file content. Default for OpenAI via ChatGPT.

Flags:
    --provider PROV       one of: claude, codex
    --model M             provider-specific model id
    --timeout SEC         wall-clock timeout (default 180)
    --json                emit `{"result": "<text>"}` instead of plain text.
                          Useful when Sophon expects the JSON envelope.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from typing import Optional


def run_claude(prompt: str, model: str, timeout: int) -> str:
    """Shell out to `claude -p --output-format json`. Returns the stripped
    `result` field, or raw stdout if the JSON envelope is missing."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exit {proc.returncode}: {proc.stderr[:500]}"
        )
    raw = proc.stdout.strip()
    if raw.startswith("{"):
        try:
            env = json.loads(raw)
            if isinstance(env, dict) and "result" in env:
                return str(env["result"]).strip()
        except json.JSONDecodeError:
            pass
    return raw


def run_codex(
    prompt: str,
    model: Optional[str],
    timeout: int,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Shell out to `codex exec --ephemeral --output-last-message FILE`.
    Codex writes the final agent message to the file (clean, no UI noise)
    so we read that rather than parse stdout.

    With a ChatGPT-login account, only `gpt-5.3-codex` is accepted as the
    model. Reasoning effort is the main lever for speed-vs-quality in
    that setup: `low` / `medium` / `high` / `xhigh`. Bench harnesses use
    `medium` or `low` to keep latency reasonable.

    `--sandbox read-only` is forced so the agent never tries to mutate
    files during an answer-only task.
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False, prefix="codex_out_"
    ) as f:
        tmp_path = f.name
    try:
        cmd = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-last-message",
            tmp_path,
        ]
        if model:
            cmd.extend(["-m", model])
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex exit {proc.returncode}: {proc.stderr[:500]}"
            )
        with open(tmp_path) as f:
            return f.read().strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


PROVIDERS = {
    "claude": {
        "runner": run_claude,
        "default_model": "sonnet",
        "model_required": True,
    },
    "codex": {
        "runner": run_codex,
        "default_model": None,  # codex picks from its own config when None
        "model_required": False,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal LLM CLI wrapper")
    parser.add_argument(
        "--provider",
        required=True,
        choices=list(PROVIDERS.keys()),
        help="LLM backend to shell out to",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider-specific model id (falls back to provider default)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Subprocess wall-clock timeout in seconds (default 180)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help='Emit {"result": "..."} JSON envelope instead of plain text',
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=[None, "low", "medium", "high", "xhigh"],
        help="Codex only: reasoning effort level. ChatGPT accounts support "
        "low/medium/high/xhigh; default is picked by codex config.toml.",
    )
    args = parser.parse_args()

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("llm_cli: empty stdin", file=sys.stderr)
        return 2

    spec = PROVIDERS[args.provider]
    model = args.model or spec["default_model"]

    try:
        if args.provider == "codex":
            answer = run_codex(
                prompt, model, args.timeout, reasoning_effort=args.reasoning_effort
            )
        else:
            answer = spec["runner"](prompt, model, args.timeout)
    except subprocess.TimeoutExpired:
        print(
            f"llm_cli: timeout after {args.timeout}s for {args.provider}",
            file=sys.stderr,
        )
        return 124
    except Exception as exc:
        print(f"llm_cli: {args.provider} failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        sys.stdout.write(json.dumps({"result": answer}))
    else:
        sys.stdout.write(answer)
    if not answer.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
