# mcp-sophon

npm wrapper for [Sophon](https://github.com/lacausecrypto/mcp-sophon).

> **Honest token economics for MCP agents.** One Rust binary. Zero ML.
> Reproducible benchmarks.

Sophon is a deterministic context layer for MCP-speaking agents:
query-driven prompt compression, conversation memory, code navigation,
file delta streaming, output compression. Measured **67 %** session
token savings, **~10 pts more saved than LLMLingua-2** on structured
prompts at **35× lower latency**, and **parity with mem0** on LOCOMO
retrieval at **sub-second vs 8.7 minutes**. Every number reproducible
from [BENCHMARK.md](https://github.com/lacausecrypto/mcp-sophon/blob/main/BENCHMARK.md).

Installing this package downloads a prebuilt native binary from the
matching GitHub release and exposes it as the `sophon` command.

## Install

```bash
npm install -g mcp-sophon
sophon --help
```

Supported platforms (via GitHub Release prebuilds):

| OS | Arch |
|---|---|
| macOS | arm64, x64 |
| Linux | arm64, x64 |
| Windows | x64 |

## Use as an MCP server

```json
{
  "mcpServers": {
    "sophon": {
      "command": "sophon",
      "args": ["serve"]
    }
  }
}
```

## Environment

- `SOPHON_SKIP_DOWNLOAD=1` — do not attempt to download the binary on install (useful in Docker / air-gapped CI). Provide a `sophon` binary on `PATH` yourself.
- `SOPHON_REPO=org/repo` — override which GitHub repository the postinstall fetches from. Defaults to `lacausecrypto/mcp-sophon`.

## See also

- [Main README](https://github.com/lacausecrypto/mcp-sophon#readme)
- [Measured benchmarks (BENCHMARK.md)](https://github.com/lacausecrypto/mcp-sophon/blob/main/BENCHMARK.md)

## License

MIT
