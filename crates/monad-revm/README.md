# monad-revm

Monad's Rust execution engine, built on [REVM](https://github.com/bluealloy/revm).
It implements Monad gas accounting, opcodes and protocol precompiles, with database
and inspector interfaces for embedding in other tools.

## Quick start

```toml
[dependencies]
monad-revm = "0.8.0"
revm = "43.0.0"
```

See the [basic example](examples/basic.rs), or run it from the workspace root:

```sh
cargo run -p monad-revm --example basic
```

The default hardfork is `MonadTen`; `MonadNext` includes experimental, unscheduled
extensions. `no_std` with `alloc` is supported by disabling default features.

## Documentation

- [Usage, feature flags and integration](docs/usage.md)
- [Hardforks and execution rules](docs/execution.md)
- [Staking precompile](docs/staking.md), including current parity limitations
- [Reserve-balance precompile and required chain context](docs/reserve-balance.md)
- [Rust API reference](https://docs.rs/monad-revm)
