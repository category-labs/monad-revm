# Reserve-balance precompile (`0x1001`)

[Crate overview](../README.md)

## Activation

- Active on `MonadNine` and above.
- Exposes reserve-balance state during transaction execution.
- The precompile is available at `0x1001` and returns `None` before MonadNine.

## Solidity interface

```solidity
interface IReserveBalance {
    function dippedIntoReserve() external returns (bool);
}
```

- Selector: `0x3a61584e`
- Gas: `100`

## Semantics

- Returns `true` when the current transaction state would violate Monad reserve-balance rules if execution ended at that point.
- Intended for contracts that want to recover, branch, or revert early before transaction end.

## Call restrictions

- Only direct `CALL` is accepted.
- `STATICCALL`, `DELEGATECALL`, and `CALLCODE` are rejected.
- Ordinary direct internal `CALL`s to `0x1001` are supported. Top-level and internal calls routed
  through an EIP-7702 delegated address are rejected.
- Calldata must be exactly the 4-byte selector.
- Nonzero `msg.value` is rejected.

Error behavior matches the canonical Monad implementation:

- Unknown or short selector: `"method not supported"`
- Nonzero value: `"value is nonzero"`
- Extra calldata beyond the selector: `"input is invalid"`

## Chain context and tracker lifecycle

Canonical reserve-balance decisions require a populated `MonadChainContext`: the combined senders
and authorities from the parent and grandparent blocks, current-block senders and authorization
lists, the current transaction index, and the applicable maximum reserve balance. The default
context is intentionally empty and is not sufficient to reproduce historical block execution.
The default maximum reserve balance is 10 MON.

The standard transaction handler initializes and clears `ReserveBalanceTracker` at transaction
boundaries. On every supported Monad hardfork, it rejects the outer CALL or CREATE execution if
reserves are still violated at completion. Transfers, storage, deployed code, and logs roll back;
full fees, the sender nonce, and EIP-7702 authorizations remain. Temporary deficits that are repaid
before completion, including those rolled back by nested calls, do not reject the transaction.

Execution now returns `ExecutionResult<MonadHaltReason>`: reserve rejection is
`MonadHaltReason::ReserveBalanceViolation`, and ordinary Ethereum halts are wrapped in
`MonadHaltReason::Ethereum`. CALL reserve rejection preserves remaining execution gas, while
CREATE reserve rejection consumes it, matching the canonical execution behavior. Both retain
Monad's full gas-limit fee charging. Custom `MonadJournalTr` implementations must provide
`checkpoint_execution`, `commit_execution`, and `revert_execution`, which snapshot execution
without changing the observable call depth. Their ordinary frame rollback must call
`ReserveBalanceTracker::before_checkpoint_revert` before restoring state, except during preserved
synthetic execution, and always call `on_checkpoint_revert` afterward. This retains an outer
frame's reserve violation before its original failure rolls back state, while allowing a
synthetic frame to revert its own temporary deficit.

Reserve adjudication and execution rollback precede the root inspector `call_end`, `create_end`,
and `frame_end` callbacks. These callbacks observe a `Revert` outcome after reserve rejection,
including for immediate calls and inspector-supplied outcomes. Inspectors can still override that
outcome; changing its status or output replaces the reserve rejection in the final result without
restoring rolled-back execution. Nested callbacks retain their ordinary order and behavior.

Stock REVM tracers therefore report failed reserve transactions using their generic revert error
labels. The distinct reserve reason is available in `MonadHaltReason::ReserveBalanceViolation`;
the trace error string does not reproduce the custom canonical Monad label. Log callbacks record
attempted execution, and stock Geth call traces can retain attempted root logs even for ordinary
reverts. Committed logs are determined by `ExecutionResult`, which excludes rejected execution.

Embedders that execute a synthetic transaction as an enclosing call must set
`MonadJournalTr::set_preserve_reserve_balance_tracker(true)` only for that synthetic execution.
This defers enforcement along with tracker initialization and clearing: a synthetic call may
succeed while the enclosing transaction is below reserve. The embedder must check
`reserve_balance().has_violation()` before accepting the enclosing execution checkpoint, and
restore the flag to `false` before the next ordinary transaction. If the backing fork, journal state, or
chain metadata is replaced while a tracker is preserved, call `ReserveBalanceTracker::rebase` with
the replacement state and `MonadChainContext`; cached thresholds must not cross that boundary.
