# Local staking

User transactions call the native staking precompile at `0x1000` through `execute`.
All staking functions, including getters, require `CALL`; `STATICCALL` is rejected.
Use `commit: false` for a read without persisting changes.
`staking_transition` supplies the separate lifecycle operations normally triggered
by a block executor. Both Monad specifications use the existing Rust implementation.

```js
const request = { blockNumber: 1, timestamp: 1800000000, commit: true }
const transition = action => JSON.parse(engine.staking_transition(JSON.stringify({ ...request, action })))
transition({ kind: 'snapshot' })
transition({ kind: 'epochChange', epoch: '0x1' })
transition({ kind: 'reward', author: validatorAuthor, amount: '0xde0b6b3a7640000' })
```

The host selects block context, validators, rewards and when each operation occurs:

- `snapshot` starts the epoch delay period and updates validator-set views.
- `epochChange` completes a transition to a strictly greater epoch. Pass an exact
  hex quantity below `u64::MAX`; the engine prepares the following epoch too.
- `reward` uses the native author lookup and rewards the active validator. The
  wrapper mints the backing balance and applies reward accounting atomically.

`commit: false` leaves state unchanged. Invalid requests throw. Executed transitions
return success, error, return data, logs and gas spent; unsuccessful operations never
commit state, including a staged reward balance. They do not charge user fees or
advance user nonces. System operations have no EVM bytecode trace. Inspect ordinary
Solidity callers around staking calls through `execute` with `trace: true`.

Snapshots include staking storage and balances. Restoring one also restores its
epoch, pending delegations, rewards and withdrawals. No separate scheduler state
is stored by the wrapper.

Validator registration currently skips signature verification and uses simplified
key-derived lookups in the native engine (including `authAddress` as the reward
author). This behavior is unchanged. Local fixtures intentionally use unverified
signatures; native/WASM parity does not establish production registration fidelity.

The shared [fixtures](../tests/fixtures/staking.json) exercise delegation, both sides
of the epoch delay, rewards, claims, withdrawals, failed system operations, nested
rollback, static calls, malformed inputs and balance overflow. Native and installed
WASM tests compare complete results, logs, traces and state, continuing each workflow
from a restored snapshot after every operation.
