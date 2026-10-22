# Running fixtures

[Crate overview](../README.md)

## Usage

Run these commands from the workspace root. They assume `monad-execution-specs` is checked
out next to `monad-revm`; an explicit fixture path works as well. No node, RPC endpoint,
OCaml runtime, or fixture download is needed to replay existing JSON files.

```bash
# Run every case in the local corpus.
cargo run -p monad-fixtures -- \
  ../monad-execution-specs/test/execution/fixtures/blockchain_tests

# Select cases by a substring of their name.
cargo run -p monad-fixtures -- \
  ../monad-execution-specs/test/execution/fixtures/blockchain_tests \
  --filter sstoreGasCosts

# Save a machine-readable report and fail if any selected case is unsupported.
cargo run -p monad-fixtures -- \
  ../monad-execution-specs/test/execution/fixtures/blockchain_tests \
  --json --fail-on-unsupported > target/fixture-report.json
```

Pass one or more files or directories. Directories are searched recursively for `.json`
files, excluding `index.json` corpus manifests. Directory traversal does not follow
symlinks. Paths are canonicalized and deduplicated, and results are ordered by file path
then case name. Every top-level fixture entry is a separate case. A malformed file is
reported as a failed `<file>` case, even with a name filter, and other files still run.

Use `cargo run -p monad-fixtures -- --help` for the available options. For larger corpora,
add `--release` before the `--` separator. Fixtures are read without modifying them.

## What a pass means

The report identifies its verification scope as `valid_block_execution`. Each selected
case starts with a fresh in-memory database initialized from `pre`, then executes its
blocks in order. The runner checks:

- The genesis state root against `pre`, the genesis header against `genesisRLP`, and each
  declared block header against the RLP block header.
- Block linkage, consecutive numbers, and transaction and withdrawal roots.
- Transaction signatures and execution under the fixture's explicit fork and chain ID.
- Each block's state root, receipt root, logs bloom, and gas used. Monad receipts charge
  the full transaction gas limit, including for reverted and halted transactions.
- `lastblockhash`, complete final accounts in `postState` (or `post`), and `postStateHash`
  when supplied. Accounts are compared for existence, nonce, balance, bytecode, and storage;
  zero storage entries are equivalent to absent entries.

Execution includes the EIP-4788 beacon-root and EIP-2935 history system calls when their
contracts are deployed, followed by transactions and withdrawals. Sender and recoverable
authority metadata is supplied for the current block, parent, and grandparent so the
reserve-balance implementation has its chain context.

Block RLP is the source of transactions, withdrawals, and ommers. The JSON `blockHeader`
is checked against it; redundant JSON body fields are informational. Both a header-only
`genesisRLP` (used by the OCaml fixtures) and a full empty genesis block are accepted.
Hex addresses, hashes, and storage keys may omit leading zeros. Quantities may be hex
strings, decimal strings, or JSON integers.

The generator's `config.blobSchedule` is accepted. Supported blocks have zero excess blob
gas, so the blob base fee is one regardless of the schedule's positive update fraction;
blob transactions are rejected under Monad rules.

An EVM revert or halt can be a passing test: its failure status belongs in the expected
receipt. A transaction rejected before execution, a malformed input, or a mismatched
expectation is a runner failure. These results require investigation; a failure alone
does not identify whether the runner, fixture, or execution implementation is wrong.

`highGasUsage_MONAD_EIGHT` exercises a
[known reserve-check difference between the execution client, OCaml spec, and Foundation spec](https://github.com/category-labs/monad-execution-specs/blob/12e2c05bcfa6daeef9c0f4c54c5c7a13fde39e89/lib/reserve_balance.ml#L54-L68).
`monad-revm` follows the C++ execution behavior, so the runner reports this fixture as failed.

## Supported scope and exclusions

The first version accepts blockchain fixtures for `MONAD_EIGHT` and `MONAD_NINE`, plus
timestamp transitions such as `MONAD_EIGHTToMONAD_NINEAtTime15000` or
`MONAD_EIGHTToMONAD_NINEAtTime15k`. Fork selection follows the fixture's block timestamp,
not the network's live activation schedule or `monad-revm`'s default fork.

These cases are reported as **unsupported**, with a reason:

- Other forks, including `MONAD_TEN` and `MONAD_NEXT`.
- State-test and other non-blockchain fixture formats.
- Any case containing a block with `expectException`. Matching arbitrary errors would
  hide incorrect rejection behavior; precise invalid-block validation is future work.
- Genesis blocks above height zero, which need historical block hashes and reserve metadata.
- Nonzero `excessBlobGas`, which needs a fork-specific blob-base-fee environment.
- Blocks with ommers, seal engines other than `NoProof`, and unrecognized fixture, config,
  block, or header fields that might require additional semantics.

The input needs `config.chainid`, a network in `config.network` or `network`, `pre`,
`genesisBlockHeader`, `genesisRLP`, `blocks`, and `lastblockhash`. If both network fields
are present they must agree. Final account assertions are optional because every executed
block already supplies a state root; supplied final assertions are always checked.

This is not a complete consensus validator: it does not independently validate every
header rule, base-fee schedule, epoch transition, or staking reward/snapshot lifecycle.
It does not run the C++ node or compare two implementations directly. Passing the
supported fixture assertions is evidence of compatibility for those cases, not proof of
full C++ node compatibility. New capabilities should add their execution context and
assertion checks before marking their fixtures supported.

## Reports and exit codes

Text output gives one `PASS`, `FAIL`, or `UNSUPPORTED` line per case, with its file, name,
and diagnostic, followed by totals. JSON output contains `schema_version`, `verification`,
`summary`, and `cases`. Each case has `file`, `name`, `network`, `status`, and `message`;
status is `passed`, `failed`, or `unsupported`. Diagnostics include expected and actual
roots where available and transaction outcomes for receipt or bloom mismatches.

Reports go to stdout; invocation and input-discovery errors go to stderr. `cargo run`
also writes its build output to stderr, leaving redirected JSON clean.

| Exit code | Meaning |
|-----------|---------|
| `0` | At least one pass, no failures, and no unsupported cases if strict mode was requested. |
| `1` | At least one failure, or a mix of passes and unsupported cases with `--fail-on-unsupported`. |
| `2` | Nothing passed and nothing failed (empty selection or entirely unsupported), or an invocation/discovery error. |

Thus an empty directory, a filter matching no cases, and an entirely unsupported corpus
cannot produce a successful run. Without strict mode, a mixture of passed and unsupported
cases exits successfully but still reports every unsupported case and its reason.
