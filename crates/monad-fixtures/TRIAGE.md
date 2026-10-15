# Fixture compatibility investigation

The 15 failures in the initial 227-case run were execution compatibility failures in the
older `monad-revm` checkout. All 15 pass with the existing reserve-enforcement fix in
[`03bd8be`](https://github.com/category-labs/monad-revm/commit/03bd8be4bea7d3ab7fb76c76bc52a169b956d4af).
The runner's parsing, execution, receipt construction, and assertions were unchanged
between those runs. No production execution change was needed in this investigation.

Current `main` passes 226 cases and fails one different case, `highGasUsage_MONAD_EIGHT`.
That case exposes an explicitly documented difference between the OCaml specification
and the C++ execution client's handling of transaction fees above the reserve.

## Inputs and controlled comparison

Investigation date: 2026-10-15. The corpus is the 227 cases checked into
[`monad-execution-specs` at `12e2c05`](https://github.com/category-labs/monad-execution-specs/tree/12e2c05bcfa6daeef9c0f4c54c5c7a13fde39e89/test/execution/fixtures/blockchain_tests).
This excludes the separately downloaded fixture releases. Every run used the same JSON
files, runner implementation, dependency lockfile, and execution features. The local
crates were rebuilt when changing the execution-library revision.

| `monad-revm` revision | Passed | Failed | Unsupported |
|----------------------|-------:|-------:|------------:|
| `d339ed5` — original checkout | 212 | 15 | 0 |
| `08999f0` — immediately before reserve enforcement | 212 | 15 | 0 |
| `03bd8be` — transaction reserve enforcement | 226 | 1 | 0 |
| `83f3678` — reserve adjudication before root callbacks | 226 | 1 | 0 |
| `bc7f7af` — current `main` at investigation time | 226 | 1 | 0 |

The C++ reference was inspected at
[`7fc140e`](https://github.com/category-labs/monad/tree/7fc140eef12fa03e8d96e64150d4e4dc7270f67b).
This investigation replayed fixtures in Rust and compared the relevant C++ and OCaml
source paths; it did not execute the C++ binary.

## Why the original 15 failed

The older Rust implementation tracked reserve violations for introspection but did not
enforce them at the outer transaction boundary. Transactions that should have rolled
back therefore retained successful receipts, transfers, and sometimes logs. A receipt
root or logs-bloom mismatch was the first observable failure; subsequent state checks
were not reached.

The OCaml host
[checks reserves after successful outer execution and restores execution state on failure](https://github.com/category-labs/monad-execution-specs/blob/12e2c05bcfa6daeef9c0f4c54c5c7a13fde39e89/lib/host.ml#L233-L245).
The C++ node
[adjudicates reserves at the outer CALL boundary](https://github.com/category-labs/monad/blob/7fc140eef12fa03e8d96e64150d4e4dc7270f67b/category/execution/ethereum/execute_message.cpp#L333-L347)
and likewise checks contract creation. Fees, sender nonce, and processed authorizations
are outside the execution rollback. The upstream Rust fix adds that enforcement.

All cases below use `MONAD_EIGHT`. The block index is zero-based within the fixture's
`blocks` array. Each case now passes every checked block commitment and its final account
assertions, rather than merely passing the first previously failing assertion.

| Fixture | First failing block index | Original mismatch | With `03bd8be` and current `main` |
|---------|--------------------------:|-------------------|----------------------------------|
| `OOGStateCopyContainingDeletedContract` | 0 | Receipt root | Pass |
| `RecallSuicidedContract` | 1 | Receipt root | Pass |
| `RecallSuicidedContractInOneBlock` | 1 | Receipt root | Pass |
| `gasLimitTooHigh2` | 0 | Receipt root | Pass |
| `logRevert` | 1 | Receipt root | Pass |
| `multimpleBalanceInstruction` | 1 | Receipt root | Pass |
| `suicideCoinbase` | 0 | Receipt root | Pass |
| `timeDiff12` | 1 | Logs bloom | Pass |
| `timeDiff13` | 1 | Logs bloom | Pass |
| `timeDiff14` | 1 | Logs bloom | Pass |
| `wallet2outOf3txs` | 1 | Logs bloom | Pass |
| `wallet2outOf3txs2` | 1 | Logs bloom | Pass |
| `wallet2outOf3txsRevoke` | 1 | Logs bloom | Pass |
| `wallet2outOf3txsRevokeAndConfirmAgain` | 1 | Logs bloom | Pass |
| `walletReorganizeOwners` | 1 | Logs bloom | Pass |

For example, `gasLimitTooHigh2` sends four transactions from the same low-balance sender
in one block. Only the first may dip into its reserve. Before the fix, all four executed
successfully. Correct enforcement changes the subsequent receipts and rolls back their
value transfers. Several other cases expose the same restriction through a sender that
already appeared in a parent block, with discarded logs making the bloom differ.

## The remaining `highGasUsage` mismatch

The first failure is block number 2 (fixture block index 1), transaction index 0:

| Field | Value |
|-------|-------|
| Sender | `0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b` |
| Gas limit | `850000` |
| Gas price | `12000000000000` wei |
| Full transaction fee | `10200000000000000000` wei = **10.2 MON** |
| Maximum reserve | **10 MON** |
| Sender allowed to dip | No; the sender also appeared in the parent block |
| Fixture expectation | Successful receipt |
| Rust result | `MonadHaltReason::ReserveBalanceViolation` |

The sender has ample funds. The distinguishing condition is that its fee alone exceeds
the reserve while it is not allowed to dip:

- The C++ implementation
  [treats `gas_fees > reserve` as a reserve violation, then applies the sender eligibility check](https://github.com/category-labs/monad/blob/7fc140eef12fa03e8d96e64150d4e4dc7270f67b/category/execution/monad/reserve_balance.cpp#L102-L124).
  The Rust tracker uses the same behavior: subtraction returns no threshold and marks a
  protected sender as violating its reserve.
- The OCaml specification
  [clamps that threshold to zero and explicitly documents the difference](https://github.com/category-labs/monad-execution-specs/blob/12e2c05bcfa6daeef9c0f4c54c5c7a13fde39e89/lib/reserve_balance.ml#L54-L68).
  Its comment explains that it assumes consensus will exclude transactions whose fees
  exceed the reserve. This synthetic fixture nevertheless reaches that condition.

For this case, Rust follows the inspected C++ execution rule. Changing Rust to match
this fixture would introduce a C++ compatibility difference. The runner therefore keeps
the case **failed**, with the receipt-root mismatch and reserve-rejection outcome visible;
it is neither excluded nor classified as unsupported.

## Regression coverage and reproduction

The [runner tests](tests/runner.rs) independently construct signed fixtures covering
current-block sender reuse, parent and grandparent participation, and eligibility after
two intervening blocks without that sender. A high-fee case passes with the C++-compatible
failed receipt and fails when given the OCaml-style successful receipt expectation.

The [library reserve tests](../monad-revm/tests/reserve_enforcement.rs) cover fees equal to
and above the reserve, protected and eligible senders, Monad Eight and Nine, and ordinary
and inspected execution. These tests retain full fees and the sender nonce in both outcomes.

From the workspace root, with the pinned spec checkout beside it:

```bash
cargo run -p monad-fixtures -- \
  ../monad-execution-specs/test/execution/fixtures/blockchain_tests \
  --json > target/fixture-report.json

cargo run -p monad-fixtures -- \
  ../monad-execution-specs/test/execution/fixtures/blockchain_tests \
  --filter highGasUsage --json
```

Both commands exit `1` while the documented expectation disagreement remains. The
full-corpus report is expected to contain 226 passed, 1 failed, and 0 unsupported cases
for the revisions recorded above. Fixture or implementation updates may change those
counts; preserve the input revisions when comparing future results.
