# Monad REVM

[![Crates.io](https://img.shields.io/crates/v/monad-revm.svg)](https://crates.io/crates/monad-revm)
[![Documentation](https://docs.rs/monad-revm/badge.svg)](https://docs.rs/monad-revm)
[![License](https://img.shields.io/crates/l/monad-revm.svg)](LICENSE)

Monad-revm extends the [Revm](https://github.com/bluealloy/revm) implementation with Monad-specific customizations.

## EVM Compatibility

| Component | Version |
|-----------|---------|
| **Revm** | v34.0.0 |
| **Monad Spec** | MONAD_EIGHT (Prague compatible)|

## Features

### Gas Model

Monad uses a modified gas model optimized for its execution environment:

**No Gas Refunds** — Users pay `gas_limit × gas_price`, not `gas_used × gas_price`.

**Increased Cold Access Costs** — Cold storage reads are more expensive to reflect the cost of state access from disk:

| Access Type | Ethereum | Monad |
|-------------|----------|-------|
| Cold Storage (SLOAD) | 2,100 | 8,100 |
| Cold Account (BALANCE, EXTCODE*, CALL*) | 2,600 | 10,100 |
| Warm Access | 100 | 100 |

### Repriced Precompiles

Several precompiles are repriced to reflect their relative computational cost:

| Precompile | Address | Ethereum | Monad | Multiplier |
|------------|---------|----------|-------|------------|
| ecRecover | 0x01 | 3,000 | 6,000 | 2x |
| ecAdd | 0x06 | 150 | 300 | 2x |
| ecMul | 0x07 | 6,000 | 30,000 | 5x |
| ecPairing | 0x08 | 45,000 + 34,000/pt | 225,000 + 170,000/pt | 5x |
| blake2f | 0x09 | rounds × 1 | rounds × 2 | 2x |
| KZG point eval | 0x0a | 50,000 | 200,000 | 4x |

### Additional Precompiles

**Staking Precompile** — monad-revm provides read-only access to Monad's native staking state at address `0x1000`:

| Method | Selector | Gas |
|--------|----------|-----|
| `getEpoch()` | 0x757991a8 | 16,200 |
| `getProposerValId()` | 0xfbacb0be | 100 |
| `getValidator(uint64)` | 0x2b6d639a | 97,200 |
| `getDelegator(uint64,address)` | 0x573c1ce0 | 184,900 |
| `getWithdrawalRequest(uint64,address,uint8)` | 0x56fa2045 | 24,300 |
| `getConsensusValidatorSet(uint32)` | 0xfb29b729 | 2,100 + 2,100/elem |
| `getSnapshotValidatorSet(uint32)` | 0xde66a368 | 2,100 + 2,100/elem |
| `getExecutionValidatorSet(uint32)` | 0x7cb074df | 2,100 + 2,100/elem |
| `getDelegations(address,uint64)` | 0xa6a7301c | 814,000 |
| `getDelegators(uint64,address)` | 0x48e327d0 | 814,000 |

### Bytecode Limits

| Limit | Ethereum | Monad |
|-------|----------|-------|
| Max code size | 24KB | 128KB |
| Max initcode size | 48KB | 256KB |

### Transaction Restrictions

**Blob Transactions Rejected** — EIP-4844 blob transactions return `Eip4844NotSupported`.

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

### Basic Example

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::{
    context::{Context, TxEnv},
    database::InMemoryDB,
    ExecuteEvm,
};

// Create a Monad EVM context
let ctx = Context::monad();
let mut evm = ctx.build_monad();

// Execute a transaction
let tx = TxEnv::builder()
    .caller(caller_address)
    .to(contract_address)
    .value(U256::from(1000))
    .gas_limit(100_000)
    .gas_price(1_000_000_000)
    .build_fill();

let result = evm.transact(tx).expect("Transaction failed");
```

### With Inspector

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::{context::Context, inspector::NoOpInspector};

let ctx = Context::monad();
let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
```

### With Custom Database

```rust
use monad_revm::{MonadBuilder, DefaultMonad};
use revm::context::Context;

let db = MyCustomDatabase::new();
let ctx = Context::monad().with_db(db);
let mut evm = ctx.build_monad();
```

## Architecture

```
monad-revm/
├── crates/
│   └── monad-revm/
│       └── src/
│           ├── lib.rs           # Re-exports
│           ├── spec.rs          # MonadSpecId hardfork enum
│           ├── cfg.rs           # MonadCfgEnv 
│           ├── handler.rs       # MonadHandler
│           ├── instructions.rs  # Custom gas parameters
│           ├── precompiles.rs   # Repriced precompiles
│           ├── evm.rs           # MonadEvm type alias
│           ├── api/
│           │   ├── builder.rs   # MonadBuilder trait
│           │   ├── exec.rs      # MonadContextTr trait
│           │   └── default_ctx.rs # DefaultMonad extension
│           └── staking/
│               ├── mod.rs       # Staking precompile dispatcher
│               ├── abi.rs       # Gas constants
│               ├── interface.rs # Solidity ABI definitions
│               ├── storage.rs   # Storage slot calculations
│               └── types.rs     # Validator, Delegator, etc.
└── Cargo.toml
```

### Key Components

**`MonadSpecId`** — Hardfork identifier. Currently only `MonadEight` (based on Prague).

**`MonadHandler`** — Implements Monad's gas model.

**`MonadPrecompiles`** — Provides repriced Ethereum precompiles plus P256VERIFY and the staking precompile.

**`monad_gas_params()`** — Returns gas parameters with Monad's cold access costs.

## Feature Flags

- `serde` — Enable serialization for `MonadSpecId`
- `alloy-evm` — Enable integration with `alloy_evm::precompiles::PrecompilesMap` for Foundry/Anvil

## Integration with Foundry

See [monad-foundry](https://github.com/category-labs/foundry) for the complete Foundry integration.

## Related Projects

- [revm](https://github.com/bluealloy/revm) — Base Ethereum Virtual Machine
- [alloy-monad-evm](https://github.com/haythemsellami/evm) — Alloy EVM wrapper for monad-revm
- [monad-foundry](https://github.com/category-labs/foundry) — Foundry fork with Monad support

## References

- [Monad Opcode Pricing](https://docs.monad.xyz/developer-essentials/opcode-pricing)
- [Monad Precompiles](https://docs.monad.xyz/developer-essentials/precompiles)
- [Monad Changelog](https://docs.monad.xyz/developer-essentials/changelog)

## License

Revm is licensed under MIT License.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in these crates by you, shall be licensed as above, without any additional terms or conditions.