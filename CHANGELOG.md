# Changelog

Notable changes beginning with `monad-revm` 0.5.0 are documented in this file.

## [0.6.0] - 2026-08-14

### Added

- Added transaction-level regressions for exact-once fee and nonce accounting, reserve tracking
  across the initial value transfer, and delegated native-precompile calls created by an EIP-7702
  authorization in the same transaction.
- Expanded CI across the complete feature surface, including `serde` and `memory_limit` on a
  bare-metal `no_std` target, and added a crate publication dry run for pull requests.

### Changed

- Updated the public integration surface from REVM 41 to `revm` 42.0.1 and
  `revm-interpreter` 42.0.0. Downstream crates must update their REVM types in lockstep because
  types from REVM 41 and 42 are not interchangeable.
- Adopted REVM 42's validation, transaction-level `GasTracker`, runtime checkpoint, and fallible
  refund lifecycle while retaining Monad's full gas-limit charging and zero-refund behavior.
- Delegated initial frame construction to REVM so memory, EIP-7702, checkpoint, and runtime
  out-of-gas behavior continue to follow upstream changes. Monad now applies only its required
  delegated native-precompile address selection after upstream frame construction.
- Added the REVM 42 configuration surface while keeping Amsterdam EIP-2780 and EIP-8037 disabled
  for `MonadEight`, `MonadNine`, and `MonadNext`.

### Fixed

- Kept Monad caller fee deduction and call nonce advancement in the single REVM 42 validation
  hook, avoiding duplicate accounting during pre-execution.
- Preserved rejection of EIP-7702 delegated calls into the staking and reserve-balance
  precompiles, including delegations installed by the current transaction.
- Initialized reserve-balance tracking after authorization processing and before the initial
  value transfer, so same-transaction sender delegation and the transfer are both observed.
- Updated reserve-tracker rollback bookkeeping for REVM 42's expanded code-change journal entry.

## [0.5.1] - 2026-08-06

### Changed

- Restored the MIT license used through `0.4.0`. The `0.5.0` crates.io release remains available
  under GPL-3.0-only according to its published package metadata.

### Fixed

- Made staking validator-set bit operations portable to 32-bit targets by using a
  target-sized, bounded shift count.
- Added bare-metal CI coverage for the default `no_std` library configuration and the
  `memory_limit` feature.

## [0.5.0] - 2026-08-06

### Added

- Added mainnet and testnet hardfork schedule lookup for `MonadEight` and `MonadNine`.
- Added `no_std` support when the default `std` feature is disabled.

### Changed

- Relicensed the crate from MIT to GPL-3.0-only, matching the Monad repository.
- Updated the public integration surface to REVM 41.
- Selected Monad instructions, precompiles, warm addresses, and memory limits for each call frame.
- Enforced MonadNine's 8 MiB pooled memory cap against configured execution limits.
- Preserved and rebased reserve-balance tracking for synthetic transactions and replacement state.

### Fixed

- Restored the parent frame's Monad behavior after nested calls and immediate precompile results.
- Reverted preserved reserve-balance tracker mutations when synthetic execution fails.

[0.6.0]: https://github.com/category-labs/monad-revm/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/category-labs/monad-revm/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/category-labs/monad-revm/compare/v0.4.0...v0.5.0
