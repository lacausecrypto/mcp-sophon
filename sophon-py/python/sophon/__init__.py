"""Sophon — honest token economics for MCP agents.

Python bindings for the Sophon Rust library. Provides prompt compression,
output compression, and token counting without spawning subprocesses.

Usage::

    from sophon import Sophon

    s = Sophon()
    result = s.compress_prompt("<rust>use Result</rust><web>fetch()</web>", "rust errors")
    print(result.compressed_prompt, result.token_count)

    out = s.compress_output("cargo test", long_test_output)
    print(out.compressed, out.ratio)

    tokens = Sophon.count_tokens("hello world")
"""

try:
    from sophon._sophon import Sophon, PromptResult, OutputResult
except ImportError:
    from sophon.sophon import Sophon, PromptResult, OutputResult

__all__ = ["Sophon", "PromptResult", "OutputResult"]
