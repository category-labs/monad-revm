# Changelog

All notable changes to `monad-revm` are documented in this file.

## [0.5.0] - 2026-08-05

### Added

- Added mainnet and testnet hardfork schedule lookup for `MonadEight` and `MonadNine`.
- Added `no_std` support when the default `std` feature is disabled.

### Changed

- Updated the public integration surface to REVM 41.
- Selected Monad instructions, precompiles, warm addresses, and memory limits for each call frame.
- Enforced MonadNine's 8 MiB pooled memory cap against configured execution limits.
- Preserved and rebased reserve-balance tracking for synthetic transactions and replacement state.

### Fixed

- Restored the parent frame's Monad behavior after nested calls and immediate precompile results.
- Reverted preserved reserve-balance tracker mutations when synthetic execution fails.

[0.5.0]: https://github.com/category-labs/monad-revm/compare/v0.4.0...v0.5.0
