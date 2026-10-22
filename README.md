# Monad REVM

[![Crates.io](https://img.shields.io/crates/v/monad-revm.svg)](https://crates.io/crates/monad-revm)
[![Documentation](https://docs.rs/monad-revm/badge.svg)](https://docs.rs/monad-revm)
[![License](https://img.shields.io/crates/l/monad-revm.svg)](LICENSE)

Monad execution and tooling built on [REVM](https://github.com/bluealloy/revm).
This workspace provides a Rust execution engine, a blockchain fixture runner, and
JavaScript/WebAssembly bindings for local development tools.

## Crates

| Crate | Purpose |
|-------|---------|
| [`monad-revm`](crates/monad-revm/README.md) | Rust engine implementing Monad execution rules, opcodes and precompiles. |
| [`monad-fixtures`](crates/monad-fixtures/README.md) | CLI for replaying blockchain fixtures and checking state and receipts. |
| [`monad-revmjs`](crates/monad-revmjs/README.md) | Experimental JavaScript/WASM package for local execution and inspection. |

Each crate README covers setup and links to detailed guides in its own `docs/` directory.

## Development

The workspace requires Rust 1.91 or newer. From the repository root:

```sh
cargo test --workspace --locked
cargo fmt --all --check
```

See the [WASM build guide](crates/monad-revmjs/docs/development.md) for its additional toolchain requirements.

## Related projects

- [`alloy-monad-evm`](https://github.com/category-labs/alloy-monad-evm): Alloy integration for the engine.
- [Foundry Monad integration](https://github.com/foundry-rs/foundry/pull/15343): Forge, Anvil, Cast and Chisel support.

## License

[MIT](LICENSE). Contributions are licensed under the same terms unless explicitly stated otherwise.
