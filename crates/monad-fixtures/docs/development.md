# Fixture runner development

[Crate overview](../README.md)

Run commands from the workspace root.

## Continuous integration

The `execution fixtures` CI job runs the checked-in corpus from
[`monad-execution-specs` at `12e2c05`](https://github.com/category-labs/monad-execution-specs/tree/12e2c05bcfa6daeef9c0f4c54c5c7a13fde39e89/test/execution/fixtures/blockchain_tests)
on every pull request and push to `main`. It uploads the raw JSON report and writes a
job summary, including when the fixture check fails.

The gate requires all 227 results: 226 passes and the known `highGasUsage` failure.
That exception matches the exact fixture file, case name, network, block index, receipt
roots, and reserve-violation outcome. Any other failure, unsupported case, missing or
duplicate result, unexpected pass, or changed diagnostic fails the job. The report
retains the failure status; the CLI's normal exit codes are unchanged.

The spec revision is pinned in [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml). When updating it, review the
report and the acceptance policy in [`.github/scripts/fixture_ci.py`](../../../.github/scripts/fixture_ci.py) together.
To reproduce the gate with that spec checkout locally:

```bash
cargo build -p monad-fixtures
python3 .github/scripts/fixture_ci.py \
  --runner target/debug/monad-fixtures \
  --fixtures ../monad-execution-specs/test/execution/fixtures/blockchain_tests \
  --report target/fixture-report.json
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
```

## Runner tests

```bash
cargo test -p monad-fixtures
cargo clippy -p monad-fixtures --all-targets -- -D warnings
cargo fmt --all --check
```

The integration tests construct signed fixtures with independently calculated expected
effects, exercise execution and reporting, and deliberately corrupt expectations to check
failure detection. They do not require a neighboring fixture repository.

- [`src/fixture.rs`](../src/fixture.rs): input parsing, fork selection, and capability checks.
- [`src/execute.rs`](../src/execute.rs): database initialization, block execution, and commitment verification.
- [`src/lib.rs`](../src/lib.rs): discovery, filtering, per-case classification, and report aggregation.
- [`src/main.rs`](../src/main.rs): CLI output and exit codes.

The CLI requires `std`; bare-metal `no_std` checks remain scoped to the `monad-revm`
library. Workspace resolver version 3 prefers dependencies compatible with the declared
Rust version when generating a fresh lockfile.
