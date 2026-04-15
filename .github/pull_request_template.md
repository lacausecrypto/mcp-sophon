## Summary

What does this change and why.

## Type

- [ ] Bug fix
- [ ] New feature (new MCP tool / new module)
- [ ] Refactor / cleanup (no behavior change)
- [ ] Docs / benchmark update
- [ ] CI / release infra

## Checklist

- [ ] `cargo test --workspace` passes locally
- [ ] `cargo fmt --all --check` passes
- [ ] New or changed behavior is covered by a test
- [ ] [BENCHMARK.md](../BENCHMARK.md) updated if this affects measured numbers
- [ ] [README.md](../README.md) updated if this adds/removes/renames a tool
- [ ] No new runtime ML dependency introduced without an opt-in flag

## Benchmark impact (if any)

If this touches compression logic, paste the before/after numbers from the
relevant bench script.
