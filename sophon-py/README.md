# sophon (Python)

Python bindings for [Sophon](https://github.com/lacausecrypto/mcp-sophon) — honest token economics for MCP agents.

## Install

```bash
pip install sophon
```

## Usage

```python
from sophon import Sophon

s = Sophon()

# Prompt compression (query-driven section selection)
result = s.compress_prompt(
    "<rust>use Result and ?</rust><web>fetch()</web>",
    "rust errors",
    max_tokens=500,
)
print(result.compressed_prompt)
print(f"{result.token_count} tokens, ratio={result.compression_ratio:.2f}")

# Output compression (command-aware filtering)
out = s.compress_output("cargo test", long_test_output)
print(out.compressed)
print(f"ratio={out.ratio:.2f}, filter={out.filter_name}")

# Token counting (cl100k_base)
tokens = Sophon.count_tokens("hello world")
```

## Build from source

```bash
pip install maturin
cd sophon-py
maturin develop --release
pytest tests/
```

## License

MIT
