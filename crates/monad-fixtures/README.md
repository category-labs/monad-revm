# monad-fixtures

A CLI that replays blockchain JSON fixtures through `monad-revm` and checks their
state and receipts. This workspace tool is not published to crates.io.

## Quick start

From the workspace root, pass a fixture file or directory:

```sh
cargo run -p monad-fixtures -- /path/to/blockchain_tests
cargo run -p monad-fixtures -- --help
```

The runner reports each case as passed, failed or unsupported. It currently supports
valid-block execution for `MONAD_EIGHT` and `MONAD_NINE`; a pass covers the fixture
assertions, not full consensus validation.

## Documentation

- [Running fixtures, supported scope, reports and exit codes](docs/running.md)
- [Development, CI corpus and known exception](docs/development.md)
