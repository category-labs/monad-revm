# Monad REVM

[![Crates.io](https://img.shields.io/crates/v/monad-revm.svg)](https://crates.io/crates/monad-revm)
[![Documentation](https://docs.rs/monad-revm/badge.svg)](https://docs.rs/monad-revm)
[![License](https://img.shields.io/crates/l/monad-revm.svg)](LICENSE)

`monad-revm` extends [revm](https://github.com/bluealloy/revm) with Monad-specific execution semantics: gas model changes, repriced precompiles, and full staking precompile support.

## EVM Compatibility

| Component | Version |
|-----------|---------|
| **revm** | v34.0.0 |
| **Monad spec** | `MONAD_EIGHT` (Prague-compatible baseline) |

## What Monad Changes

### Gas model

Monad uses a different cold-access model and no gas refunds.

| Access Type | Ethereum | Monad |
|-------------|----------|-------|
| Cold storage (`SLOAD`) | 2,100 | 8,100 |
| Cold account (`BALANCE`, `EXTCODE*`, `CALL*`) | 2,600 | 10,100 |
| Warm access | 100 | 100 |

### Repriced precompiles

| Precompile | Address | Ethereum | Monad | Multiplier |
|------------|---------|----------|-------|------------|
| `ecRecover` | `0x01` | 3,000 | 6,000 | 2x |
| `ecAdd` | `0x06` | 150 | 300 | 2x |
| `ecMul` | `0x07` | 6,000 | 30,000 | 5x |
| `ecPairing` | `0x08` | 45,000 + 34,000/pt | 225,000 + 170,000/pt | 5x |
| `blake2f` | `0x09` | rounds × 1 | rounds × 2 | 2x |
| KZG point evaluation | `0x0a` | 50,000 | 200,000 | 4x |

### Bytecode and transaction rules

| Rule | Ethereum | Monad |
|------|----------|-------|
| Runtime bytecode limit | 24KB | 128KB |
| Initcode limit | 48KB | 256KB |
| EIP-4844 blob tx | Supported | Rejected (`Eip4844NotSupported`) |

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

Constants (current `MONAD_EIGHT` implementation):

- `ACTIVE_VALIDATOR_STAKE = 10_000_000 MON`
- `MIN_AUTH_ADDRESS_STAKE = 100_000 MON`
- `WITHDRAWAL_DELAY = 1 epoch`
- `MIN_EXTERNAL_REWARD = 1e9`, `MAX_EXTERNAL_REWARD = 1e25`
- `ACTIVE_VALSET_SIZE = 200`

See implementation constants in `src/staking/constants.rs`.

### Staking API surface in `monad-revm`

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

- The Rust implementation currently targets `MONAD_EIGHT` constants.
- `addValidator` currently skips signature verification and uses simplified key-to-address derivation in `write.rs`. This is intentional in the current implementation and should be considered when writing integration tests.

### How staking is implemented in `monad-revm`

Core modules:

- `src/staking/mod.rs`: top-level precompile dispatcher (`run_staking_precompile`) and read handlers.
- `src/staking/write.rs`: all user write handlers + syscall handlers + selector/payability logic.
- `src/staking/storage.rs`: exact storage key derivation for all staking namespaces.
- `src/staking/types.rs`: validator/delegator/withdrawal/list node types.
- `src/staking/interface.rs`: ABI definitions and selectors.
- `src/staking/constants.rs`: gas-independent staking constants.

Block lifecycle helpers:

- `src/api/block.rs` exposes `apply_syscall_reward`, `apply_syscall_snapshot`, `apply_syscall_on_epoch_change`, and `apply_epoch_boundary`.
- `syscallReward` supports extended calldata (`selector + blockAuthor + reward`) for `SystemCallEvm` environments that cannot attach `msg.value` to system calls.

Reader integration path:

- `run_staking_with_reader(...)` supports environments that do not expose full `ContextTr`, and is used by `alloy-monad-evm` integration.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
monad-revm = { git = "https://github.com/category-labs/monad-revm", branch = "main" }
```

Or from crates.io:

```toml
[dependencies]
monad-revm = "0.1"
```

## Usage

### Basic example

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::{
    context::{Context, TxEnv},
    database::InMemoryDB,
    ExecuteEvm,
};

let ctx = Context::monad();
let mut evm = ctx.build_monad();

let tx = TxEnv::builder()
    .caller(caller_address)
    .to(contract_address)
    .value(U256::from(1000))
    .gas_limit(100_000)
    .gas_price(1_000_000_000)
    .build_fill();

let result = evm.transact(tx).expect("Transaction failed");
```

### With inspector

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::{context::Context, inspector::NoOpInspector};

let ctx = Context::monad();
let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
```

### With custom database

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::context::Context;

let db = MyCustomDatabase::new();
let ctx = Context::monad().with_db(db);
let mut evm = ctx.build_monad();
```

## Architecture

```text
monad-revm/
├── crates/
│   └── monad-revm/
│       └── src/
│           ├── lib.rs
│           ├── spec.rs
│           ├── cfg.rs
│           ├── handler.rs
│           ├── instructions.rs
│           ├── precompiles.rs
│           ├── evm.rs
│           ├── api/
│           │   ├── block.rs
│           │   ├── builder.rs
│           │   ├── exec.rs
│           │   └── default_ctx.rs
│           └── staking/
│               ├── mod.rs
│               ├── write.rs
│               ├── abi.rs
│               ├── interface.rs
│               ├── storage.rs
│               └── types.rs
└── Cargo.toml
```

## Feature flags

- `serde`: Enable serialization for `MonadSpecId`.
- `alloy-evm`: Enable integration with `alloy_evm::precompiles::PrecompilesMap`.

## Integration layers

- [`alloy-monad-evm`](https://github.com/category-labs/alloy-monad-evm): Alloy `Evm` / `EvmFactory` wrapper over `monad-revm`.
- [`monad-foundry`](https://github.com/category-labs/foundry/tree/monad): Foundry integration (Forge/Anvil/Cast/Chisel).

## References

- [Monad opcode pricing](https://docs.monad.xyz/developer-essentials/opcode-pricing)
- [Monad precompiles](https://docs.monad.xyz/developer-essentials/precompiles)
- [Monad staking precompile docs](https://docs.monad.xyz/developer-essentials/staking/staking-precompile)

## License

Revm is licensed under MIT License.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in these crates by you, shall be licensed as above, without any additional terms or conditions.
