//! Reserve-balance precompile ABI constants.

use revm::primitives::{Address, U256};

/// Reserve-balance precompile address.
pub const RESERVE_BALANCE_ADDRESS: Address = revm::precompile::u64_to_address(0x1001);

/// `dippedIntoReserve()` selector.
pub const DIPPED_INTO_RESERVE_SELECTOR: [u8; 4] = [0x3a, 0x61, 0x58, 0x4e];

/// Gas cost for `dippedIntoReserve()`.
pub const DIPPED_INTO_RESERVE_GAS: u64 = 100;

/// Fallback dispatch cost for invalid selectors.
pub const FALLBACK_GAS: u64 = 100;

/// Default reserve balance threshold: 10 MON.
pub const DEFAULT_RESERVE_BALANCE: U256 =
    U256::from_limbs([10_000_000_000_000_000_000u64, 0, 0, 0]);
