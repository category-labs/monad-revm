//! Staking precompile constants.
//!
//! Values match the C++ implementation in `monad/staking/util/constants.hpp`.

use revm::primitives::{Address, U256};

/// 1 MON = 1e18 (same as 1 ETH in wei).
pub const MON: U256 = U256::from_limbs([1_000_000_000_000_000_000, 0, 0, 0]);

/// Accumulator precision: 1e36.
///
/// Used in reward calculations: `reward = (stake * (acc - last_acc)) / UNIT_BIAS`.
pub const UNIT_BIAS: U256 = U256::from_limbs([12919594847110692864, 54210108624275221, 0, 0]);

/// Minimum delegation amount (dust threshold): 1e9 (1 Gwei).
pub const DUST_THRESHOLD: U256 = U256::from_limbs([1_000_000_000, 0, 0, 0]);

/// Maximum commission rate: 1e18 = 100%.
pub const MAX_COMMISSION: U256 = MON;

/// Minimum stake required for validator auth address: 100,000 MON = 1e23.
pub const MIN_AUTH_ADDRESS_STAKE: U256 = U256::from_limbs([200376420520689664, 5421, 0, 0]);

/// Minimum active validator stake (Monad v5+): 10,000,000 MON = 1e25.
pub const ACTIVE_VALIDATOR_STAKE: U256 = U256::from_limbs([1590897978359414784, 542101, 0, 0]);

/// Minimum external reward amount: 1e9 (same as dust threshold).
pub const MIN_EXTERNAL_REWARD: U256 = DUST_THRESHOLD;

/// Maximum external reward amount: 1e25.
pub const MAX_EXTERNAL_REWARD: U256 = U256::from_limbs([1590897978359414784, 542101, 0, 0]);

/// Withdrawal delay in epochs.
pub const WITHDRAWAL_DELAY: u64 = 1;

/// Maximum validators in the consensus set.
pub const ACTIVE_VALSET_SIZE: u32 = 200;

/// System address used for syscalls (canonical Monad SYSTEM_SENDER).
pub const SYSTEM_ADDRESS: Address = Address::new([
    0x6f, 0x49, 0xa8, 0xF6, 0x21, 0x35, 0x3f, 0x12, 0x37, 0x8d, 0x00, 0x46, 0xE7, 0xd7, 0xe4, 0xb9,
    0xB2, 0x49, 0xDC, 0x9e,
]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mon_constant() {
        assert_eq!(MON, U256::from(10u64).pow(U256::from(18)));
    }

    #[test]
    fn test_unit_bias_constant() {
        assert_eq!(UNIT_BIAS, U256::from(10u64).pow(U256::from(36)));
    }

    #[test]
    fn test_dust_threshold() {
        assert_eq!(DUST_THRESHOLD, U256::from(10u64).pow(U256::from(9)));
    }

    #[test]
    fn test_min_auth_address_stake() {
        assert_eq!(
            MIN_AUTH_ADDRESS_STAKE,
            U256::from(100_000u64) * U256::from(10u64).pow(U256::from(18))
        );
    }

    #[test]
    fn test_active_validator_stake() {
        assert_eq!(
            ACTIVE_VALIDATOR_STAKE,
            U256::from(10_000_000u64) * U256::from(10u64).pow(U256::from(18))
        );
    }

    #[test]
    fn test_max_external_reward() {
        assert_eq!(MAX_EXTERNAL_REWARD, U256::from(10u64).pow(U256::from(25)));
    }

    #[test]
    fn test_max_commission() {
        assert_eq!(MAX_COMMISSION, MON);
    }
}
