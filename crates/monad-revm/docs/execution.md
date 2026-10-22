# Execution rules

[Crate overview](../README.md)

## EVM Compatibility

| Component | Version |
|-----------|---------|
| **revm** | v43.0.0 |
| **Supported Monad specs** | `MonadEight`, `MonadNine`, `MonadTen`, `MonadNext` |
| **Ethereum foundation** | Prague for `MonadEight`; Osaka for `MonadNine`, `MonadTen`, and `MonadNext` |
| **Default Monad spec** | `MonadTen` |

### Hardfork schedule

| Network | Chain ID | `MonadEight` | `MonadNine` | `MonadTen` | `MonadNext` |
|---------|----------|--------------|-------------|------------|-------------|
| Mainnet | `143` | 2025-11-20 14:30 UTC | 2026-03-19 14:30 UTC | 2026-09-02 14:30 UTC | Unscheduled |
| Testnet | `10143` | 2025-11-19 14:30 UTC | 2026-03-10 14:30 UTC | 2026-08-12 14:30 UTC | Unscheduled |

Use `MonadHardfork::from_chain_and_timestamp(chain_id, timestamp)` to resolve a known network.
Mainnet and testnet timestamps at or after their respective `MonadTen` activations resolve to
`MonadTen`; unknown chain IDs return `None`.
`Context::monad()` deliberately defaults to `MonadTen` and does not perform schedule lookup.

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

Transaction results report the full gas limit through `ExecutionResult::tx_gas_used()` for
receipts and `result.gas().block_regular_gas_used()` for block accounting, including reverts
and halts. Use `result.gas().total_gas_spent()` for execution consumption, including intrinsic
gas, when estimating gas, profiling, or producing gas reports. Monad represents its minimum
transaction charge with `ResultGas::floor_gas()`, set to the gas limit independently of
Ethereum's EIP-7623 calldata floor. This does not change interpreter gas. System calls bypass
transaction charging and retain execution-only gas accounting; they do not produce receipts.

### Repriced precompiles

| Precompile | Address | Ethereum | Monad | Multiplier |
|------------|---------|----------|-------|------------|
| `ecRecover` | `0x01` | 3,000 | 6,000 | 2x |
| `ecAdd` | `0x06` | 150 | 300 | 2x |
| `ecMul` | `0x07` | 6,000 | 30,000 | 5x |
| `ecPairing` | `0x08` | 45,000 + 34,000/pt | 225,000 + 170,000/pt | 5x |
| `blake2f` | `0x09` | rounds × 1 | rounds × 2 | 2x |
| KZG point evaluation | `0x0a` | 50,000 | 200,000 | 4x |
| P256VERIFY | `0x0100` | 6,900 (Osaka) | 6,900 (`MonadEight`+) | Enabled early |

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
for every frame. Nested calls that cross Monad hardfork boundaries restore the parent frame's
behavior on success, revert, error, and immediate precompile completion.

### MIP-8 page-ified storage

`MonadTen` introduces MIP-8 page-level storage warming and gas accounting. Storage pages contain
128 consecutive words. Loading any word warms the page, and the first write plus net state growth
are charged per page. `MonadNine` retains slot-level storage accounting; later hardforks inherit
the `MonadTen` behavior.

### MIP-7 extension opcodes

`MonadNext` enables the [MIP-7](https://mips.monad.xyz/MIPs/MIP-7) `EXTENSION` prefix
(`0xAE`). The following byte selects the extended opcode. MIP-7 defines the encoding;
individual extensions assign selectors, semantics, and gas costs, with no additional charge
for the prefix. `MonadNext` remains unscheduled, and earlier hardforks reject `0xAE`.

Unassigned selectors, the excluded `JUMPDEST` and `PUSH1`–`PUSH32` selectors, and a trailing
prefix halt exceptionally. Ordinary jump-destination analysis is unchanged: a `JUMPDEST`
following `0xAE` remains a valid jump target, and `PUSH` data remains excluded from jump targets.

### MIP-18 call stack introspection

`MonadNext` enables [MIP-18](https://mips.monad.xyz/MIPs/MIP-18), which assigns two
selectors under the MIP-7 `EXTENSION` prefix.

| Encoding | Instruction | Stack input | Stack output | Gas |
|----------|-------------|-------------|--------------|-----|
| `0xAE 0x00` | `CALLSTACKDEPTH` | None | Current frame depth, starting at zero | 2 |
| `0xAE 0x01` | `CALLERN` | Ancestor index `n` | The caller of the frame `n` levels above, or zero if out of range | 2 |

`CALLERN(0)` equals `CALLER`, and `CALLERN(CALLSTACKDEPTH)` equals `ORIGIN`.
Ancestor callers retain the usual `DELEGATECALL` and `CALLCODE` semantics. The gas charge
includes the prefix and selector.

Custom `MonadJournalTr` implementations must implement `call_frame_sender` and
`set_call_frame_sender`. Store callers by zero-based frame depth, overwrite a depth on frame
entry, retain entries across state-checkpoint rollback, and clear them on transaction commit,
discard, and finalization. `MonadEvm` records callers only on `MonadNext`. MIP-18 assumes a
single hardfork throughout each transaction.

## References

- [Monad opcode pricing](https://docs.monad.xyz/developer-essentials/opcode-pricing)
- [Monad precompiles](https://docs.monad.xyz/developer-essentials/precompiles)
