# Using monad-revm

[Crate overview](../README.md)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
monad-revm = "0.8.0"
revm = "43.0.0"
```

To pin directly to the matching immutable Git release:

```toml
[dependencies]
monad-revm = { git = "https://github.com/category-labs/monad-revm", tag = "v0.8.0" }
revm = "43.0.0"
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
    .gas_price(1)
    .build_fill();

let result = evm.transact(tx).expect("transaction should execute");
```

The same program is available as
[`basic.rs`](../examples/basic.rs).

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

## Feature flags

- `std`: Enable standard-library support for `monad-revm`, `revm`, and `alloy-sol-types` (default).
  With `default-features = false`, `monad-revm` is `no_std` and uses `alloc`.
- `serde`: Enable serialization for `MonadHardfork` and forward `serde` support to `revm`.
- `memory_limit`: Enable pooled memory accounting and MonadNine's 8 MiB protocol cap (default).
  MIP-3 linear pricing remains active when this feature is disabled, but the cap is not enforced.
- `optional_balance_check`, `optional_block_gas_limit`, `optional_no_base_fee`: Forward the matching optional execution controls to `revm`.
- `c-kzg`, `secp256k1`, `portable`, `blst`: Forward the matching cryptography/portability features to `revm`.
- `dev`: Enable development-oriented optional execution controls used by tests and local integrations.

## Source layout

| Modules | Responsibility |
|---------|----------------|
| [`api/`](../src/api/), [`evm.rs`](../src/evm.rs) | Context builders, execution and block lifecycle helpers. |
| [`spec.rs`](../src/spec.rs), [`cfg.rs`](../src/cfg.rs) | Hardfork selection and execution limits. |
| [`handler.rs`](../src/handler.rs), [`journal.rs`](../src/journal.rs) | Transaction charging, state checkpoints and reserve enforcement. |
| [`instructions.rs`](../src/instructions.rs), [`extension.rs`](../src/extension.rs) | Opcode tables and extension decoding. |
| [`memory/`](../src/memory/), [`page.rs`](../src/page.rs), [`page_opcode.rs`](../src/page_opcode.rs) | Memory and storage accounting. |
| [`precompiles.rs`](../src/precompiles.rs), [`staking/`](../src/staking/), [`reserve_balance/`](../src/reserve_balance/) | Repriced and Monad-specific precompiles. |
| [`chain.rs`](../src/chain.rs), [`result.rs`](../src/result.rs) | Chain context and Monad halt reasons. |

## Release coordination

`monad-revm`, `alloy-monad-evm`, and Foundry's Monad integration must move together when porting to a new upstream Foundry/revm/alloy stack. Keep downstream consumers pinned to matching integration refs during a port, then update them to the merged commits or immutable release tags before retiring temporary integration refs.
