# Monad REVM

[![Crates.io](https://img.shields.io/crates/v/monad-revm.svg)](https://crates.io/crates/monad-revm)
[![Documentation](https://docs.rs/monad-revm/badge.svg)](https://docs.rs/monad-revm)
[![License](https://img.shields.io/crates/l/monad-revm.svg)](LICENSE)

`monad-revm` extends [revm](https://github.com/bluealloy/revm) with Monad-specific execution semantics: gas model changes, repriced precompiles, MIP-3 memory accounting, Monad staking, and the Monad reserve-balance precompile.

## EVM Compatibility

| Component | Version |
|-----------|---------|
| **revm** | v41.0.0 |
| **Supported Monad specs** | `MonadEight`, `MonadNine`, `MonadNext` |
| **Ethereum foundation** | Prague for `MonadEight`; Osaka for `MonadNine` and `MonadNext` |
| **Default Monad spec** | `MonadNine` |

The Ethereum spec is a foundation for instruction and precompile selection, not a claim of
protocol equivalence. Monad applies the additional rules described below.

### Hardfork schedule

| Network | Chain ID | `MonadEight` | `MonadNine` |
|---------|----------|--------------|-------------|
| Mainnet | `143` | 2025-11-20 14:30 UTC | 2026-03-19 14:30 UTC |
| Testnet | `10143` | 2025-11-19 14:30 UTC | 2026-03-10 14:30 UTC |

Use `MonadHardfork::from_chain_and_timestamp(chain_id, timestamp)` to resolve a known network.
Timestamps before `MonadNine` resolve to `MonadEight`; unknown chain IDs return `None`.
`Context::monad()` deliberately defaults to `MonadNine` and does not perform schedule lookup.

## What Monad Changes

### Gas model

Monad uses a different cold-access model and charges transactions against their full gas limit.

| Access Type | Ethereum | Monad |
|-------------|----------|-------|
| Cold storage (`SLOAD`) | 2,100 | 8,100 |
| Cold account (`BALANCE`, `EXTCODE*`, `CALL*`) | 2,600 | 10,100 |
| Warm access | 100 | 100 |

The caller pays `gas_limit * effective_gas_price`, unused gas is not reimbursed, and the refund
counter is zeroed. The block beneficiary receives the priority-fee component over the full gas
limit. Unless explicitly overridden, transactions are capped at 30 million gas.

### Repriced precompiles

| Precompile | Address | Ethereum | Monad | Multiplier |
|------------|---------|----------|-------|------------|
| `ecRecover` | `0x01` | 3,000 | 6,000 | 2x |
| `ecAdd` | `0x06` | 150 | 300 | 2x |
| `ecMul` | `0x07` | 6,000 | 30,000 | 5x |
| `ecPairing` | `0x08` | 45,000 + 34,000/pt | 225,000 + 170,000/pt | 5x |
| `blake2f` | `0x09` | rounds × 1 | rounds × 2 | 2x |
| KZG point evaluation | `0x0a` | 50,000 | 200,000 | 4x |
| P256VERIFY | `0x0100` | N/A | 6,900 | Monad-only |

### Bytecode and transaction rules

| Rule | Ethereum | Monad |
|------|----------|-------|
| Runtime bytecode limit | 24KB | 128KB |
| Initcode limit | 48KB | 256KB |
| EIP-4844 blob tx | Supported | Rejected (`Eip4844NotSupported`) |
| EIP-7702 system authority | Supported | Rejected for the system address |
| EIP-7702 delegated `CREATE` / `CREATE2` | Supported | Rejected |

### MIP-3 memory model

Monad replaces Ethereum's quadratic memory expansion formula with a linear `words / 2` cost on
`MonadNine` and later. With the default `memory_limit` feature, memory is pooled across the call
stack and the effective limit is `min(configured_limit, 8 MiB)`. A lower configured limit remains
in force; a higher configured value is retained so a transition back to `MonadEight` restores it.
`MonadEight` uses REVM's configured limit and quadratic memory pricing.

Instruction tables, available and warm precompiles, and the effective memory limit are selected
for every frame. Nested calls that cross the `MonadEight`/`MonadNine` boundary restore the parent
frame's behavior on success, revert, error, and immediate precompile completion.

## Staking Precompile (`0x1000`)

### Design overview

Monad staking uses three validator sets and two reward views to keep consensus transitions deterministic:

- `execution` set: real-time set updated by delegation/undelegation.
- `consensus` set: top validators selected at snapshot time.
- `snapshot` set: previous consensus image used during boundary-period rewards.

Validator state is split into:

- Execution state (`stake`, `commission`, `accumulated_reward_per_token`, flags, unclaimed rewards, keys/auth).
- Epoch views (`consensus` / `snapshot` stake+commission) used by reward paths.

Delegator state tracks active stake, pending stake windows (`delta_stake`, `next_delta_stake`), reward cursor (`accRewardPerToken`), and linked-list pointers used by `getDelegations` / `getDelegators` pagination.

### Epoch lifecycle

1. `syscallReward(blockAuthor)` distributes the per-block reward to the active validator pool.
2. `syscallSnapshot()` enters boundary mode, copies consensus to snapshot, rebuilds consensus from execution sorted by stake.
3. `syscallOnEpochChange(newEpoch)` finalizes the transition, updates epoch, and clears boundary mode.

`blockAuthor -> validatorId` resolution is via `ValIdSecp` mapping; rewards use consensus view outside boundary and snapshot view during boundary.

### Reward accounting

Pool rewards use an accumulator model:

- `acc += reward * UNIT_BIAS / active_stake`
- Delegator rewards are computed from accumulator deltas.
- Undelegation creates a `WithdrawalRequest` with an accumulator snapshot.
- `(epoch, validator)` accumulator snapshots are reference-counted to support delayed withdrawals and epoch-window correctness.

- `ACTIVE_VALIDATOR_STAKE = 10_000_000 MON`
- `MIN_AUTH_ADDRESS_STAKE = 100_000 MON`
- `WITHDRAWAL_DELAY = 1 epoch`
- `MIN_EXTERNAL_REWARD = 1e9`, `MAX_EXTERNAL_REWARD = 1e25`
- `ACTIVE_VALSET_SIZE = 200`

See implementation constants in `crates/monad-revm/src/staking/constants.rs`.

### Staking API surface in `monad-revm`

The staking precompile is implemented for both read methods and state-mutating user/syscall methods. It is available at `0x1000`.

### Read methods

| Method | Selector | Gas |
|--------|----------|-----|
| `getEpoch()` | `0x757991a8` | `200` |
| `getProposerValId()` | `0xfbacb0be` | `100` |
| `getValidator(uint64)` | `0x2b6d639a` | `97,200` |
| `getDelegator(uint64,address)` | `0x573c1ce0` | `184,900` |
| `getWithdrawalRequest(uint64,address,uint8)` | `0x56fa2045` | `24,300` |
| `getConsensusValidatorSet(uint32)` | `0xfb29b729` | `814,000` |
| `getSnapshotValidatorSet(uint32)` | `0xde66a368` | `814,000` |
| `getExecutionValidatorSet(uint32)` | `0x7cb074df` | `814,000` |
| `getDelegations(address,uint64)` | `0x4fd66050` | `814,000` |
| `getDelegators(uint64,address)` | `0xa0843a26` | `814,000` |

### User write methods

| Method | Selector | Gas | Payable |
|--------|----------|-----|---------|
| `addValidator(bytes,bytes,bytes)` | `0xf145204c` | `505,125` | Yes |
| `delegate(uint64)` | `0x84994fec` | `260,850` | Yes |
| `undelegate(uint64,uint256,uint8)` | `0x5cf41514` | `147,750` | No |
| `withdraw(uint64,uint8)` | `0xaed2ee73` | `68,675` | No |
| `compound(uint64)` | `0xb34fea67` | `289,325` | No |
| `claimRewards(uint64)` | `0xa76e2ca5` | `155,375` | No |
| `changeCommission(uint64,uint256)` | `0x9bdcc3c8` | `39,475` | No |
| `externalReward(uint64)` | `0xe4b3303b` | `66,575` | Yes |

### Syscalls

| Method | Selector | Gas | Caller requirement |
|--------|----------|-----|--------------------|
| `syscallReward(address)` | `0x791bdcf3` | `100,000` | `SYSTEM_ADDRESS` |
| `syscallSnapshot()` | `0x157eeb21` | `500,000` | `SYSTEM_ADDRESS` |
| `syscallOnEpochChange(uint64)` | `0x1d4e9f02` | `50,000` | `SYSTEM_ADDRESS` |

### Execution semantics

- Only direct `CALL` is accepted. `DELEGATECALL`, `CALLCODE`, and `STATICCALL` are rejected.
- Unknown/short selectors route to fallback (`"method not supported"`, 40k fallback cost).
- Read path is dispatch-first for payability, matching C++ behavior (unknown selector fallback bypasses payability guard).
- `getDelegator` is intentionally treated as a write selector in canonical execution because it settles delegator state via `pull_delegator_up_to_date`.

### Important parity note

`monad-revm` tracks C++ staking behavior closely, but there are explicit implementation notes to keep in mind:

- `addValidator` currently skips signature verification and uses simplified key-to-address derivation in `write.rs`. This is intentional in the current implementation and should be considered when writing integration tests.

### How staking is implemented in `monad-revm`

Core modules:

- `crates/monad-revm/src/staking/mod.rs`: top-level precompile dispatcher and read handlers.
- `crates/monad-revm/src/staking/write.rs`: user write handlers, syscalls, and payability logic.
- `crates/monad-revm/src/staking/storage.rs`: storage key derivation for staking namespaces.
- `crates/monad-revm/src/staking/types.rs`: validator, delegator, withdrawal, and list types.
- `crates/monad-revm/src/staking/interface.rs`: ABI definitions and selectors.
- `crates/monad-revm/src/staking/constants.rs`: gas-independent staking constants.

Block lifecycle helpers:

- `crates/monad-revm/src/api/block.rs` exposes `apply_syscall_reward`, `apply_syscall_snapshot`, `apply_syscall_on_epoch_change`, and `apply_epoch_boundary`.
- `syscallReward` supports extended calldata (`selector + blockAuthor + reward`) for `SystemCallEvm` environments that cannot attach `msg.value` to system calls.

Reader integration path:

- `run_staking_with_reader(...)` supports environments that do not expose full `ContextTr`, and is used by `alloy-monad-evm` integration.
- Ordinary direct internal `CALL`s to the native staking address are supported. Top-level and
  internal calls routed through an EIP-7702 delegated address are rejected.

## Reserve Balance Precompile (`0x1001`)

### Activation

- Active on `MonadNine` and above.
- Exposes reserve-balance state during transaction execution.
- The precompile is available at `0x1001` and returns `None` before MonadNine.

### Solidity interface

```solidity
interface IReserveBalance {
    function dippedIntoReserve() external returns (bool);
}
```

- Selector: `0x3a61584e`
- Gas: `100`

### Semantics

- Returns `true` when the current transaction state would violate Monad reserve-balance rules if execution ended at that point.
- Intended for contracts that want to recover, branch, or revert early before transaction end.

### Call restrictions

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

### Chain context and tracker lifecycle

Canonical reserve-balance decisions require a populated `MonadChainContext`: the combined senders
and authorities from the parent and grandparent blocks, current-block senders and authorization
lists, the current transaction index, and the applicable maximum reserve balance. The default
context is intentionally empty and is not sufficient to reproduce historical block execution.
The default maximum reserve balance is 10 MON.

The standard transaction handler initializes and clears `ReserveBalanceTracker` at transaction
boundaries. Embedders that execute a synthetic transaction as an enclosing call must set
`MonadJournalTr::set_preserve_reserve_balance_tracker(true)` first. If active fork or journal state
and chain metadata are replaced, call `ReserveBalanceTracker::rebase` with the replacement state
and `MonadChainContext`; cached thresholds from the previous state must not be reused.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
monad-revm = "0.5.0"
```

To pin directly to the matching immutable Git release:

```toml
[dependencies]
monad-revm = { git = "https://github.com/category-labs/monad-revm", tag = "v0.5.0" }
```

## Usage

### Basic example

```rust
use monad_revm::{monad_context_with_db, MonadBuilder};
use revm::{
    context::TxEnv,
    database::InMemoryDB,
    primitives::{Address, TxKind, U256},
    state::AccountInfo,
    ExecuteEvm,
};

let caller = Address::from([0x11; 20]);
let recipient = Address::from([0x22; 20]);

let mut db = InMemoryDB::default();
db.insert_account_info(
    caller,
    AccountInfo { balance: U256::from(1_000_000), ..Default::default() },
);

let context = monad_context_with_db(db);
let mut evm = context.build_monad();

let tx = TxEnv::builder()
    .caller(caller)
    .kind(TxKind::Call(recipient))
    .gas_limit(21_000)
    .gas_price(0)
    .build_fill();

let result = evm.transact(tx).expect("transaction should execute");
```

The same program is available as [`basic.rs`](crates/monad-revm/examples/basic.rs).

### With inspector

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::{context::Context, inspector::NoOpInspector};

let ctx = Context::monad();
let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
```

### With custom database

```rust
use monad_revm::{monad_context_with_db, MonadBuilder};

let db = MyCustomDatabase::new();
let context = monad_context_with_db(db);
let mut evm = context.build_monad();
```

## Architecture

```text
monad-revm/
├── crates/
│   └── monad-revm/
│       └── src/
│           ├── lib.rs
│           ├── chain.rs
│           ├── cfg.rs
│           ├── evm.rs
│           ├── handler.rs
│           ├── instructions.rs
│           ├── journal.rs
│           ├── memory/
│           │   ├── mod.rs
│           │   └── opcodes.rs
│           ├── precompiles.rs
│           ├── reserve_balance/
│           │   ├── abi.rs
│           │   ├── error.rs
│           │   ├── interface.rs
│           │   ├── mod.rs
│           │   └── tracker.rs
│           ├── spec.rs
│           ├── api/
│           │   ├── block.rs
│           │   ├── builder.rs
│           │   ├── exec.rs
│           │   └── default_ctx.rs
│           └── staking/
│               ├── constants.rs
│               ├── mod.rs
│               ├── write.rs
│               ├── abi.rs
│               ├── interface.rs
│               ├── storage.rs
│               └── types.rs
└── Cargo.toml
```

## Feature flags

- `std`: Enable standard-library support for `monad-revm`, `revm`, and `alloy-sol-types` (default).
  With `default-features = false`, `monad-revm` is `no_std` and uses `alloc`.
- `serde`: Enable serialization for `MonadHardfork` and forward `serde` support to `revm`.
- `memory_limit`: Enable pooled memory accounting and MonadNine's 8 MiB protocol cap (default).
  MIP-3 linear pricing remains active when this feature is disabled, but the cap is not enforced.
- `optional_balance_check`, `optional_block_gas_limit`, `optional_no_base_fee`: Forward the matching optional execution controls to `revm`.
- `c-kzg`, `secp256k1`, `portable`, `blst`: Forward the matching cryptography/portability features to `revm`.
- `dev`: Enable development-oriented optional execution controls used by tests and local integrations.

## Integration layers

- [`alloy-monad-evm`](https://github.com/category-labs/alloy-monad-evm): Alloy `Evm` / `EvmFactory` wrapper over `monad-revm`.
- [Foundry Monad integration](https://github.com/foundry-rs/foundry/pull/15343): Forge, Anvil, Cast, and Chisel support.

## Release coordination

`monad-revm`, `alloy-monad-evm`, and Foundry's Monad integration must move together when porting to a new upstream Foundry/revm/alloy stack. Keep downstream consumers pinned to matching integration refs during a port, then update them to the merged commits or immutable release tags before retiring temporary integration refs.

## References

- [Monad opcode pricing](https://docs.monad.xyz/developer-essentials/opcode-pricing)
- [Monad precompiles](https://docs.monad.xyz/developer-essentials/precompiles)
- [Monad staking API](https://docs.monad.xyz/reference/staking/api)

## License

`monad-revm` is licensed under the MIT License.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in these crates by you, shall be licensed as above, without any additional terms or conditions.
