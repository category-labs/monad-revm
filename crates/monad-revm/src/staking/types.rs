//! Data types for the staking precompile.
//!
//! These types represent the storage layout from C++ monad/staking.

use revm::primitives::{Address, U256};

/// Validator execution view (8 storage slots).
///
/// Layout matches C++ ValExecution struct.
#[derive(Debug, Clone)]
pub struct Validator {
    /// Slot 0: Total stake in the validator pool
    pub stake: U256,
    /// Slot 1: Accumulated reward per token
    pub accumulated_reward_per_token: U256,
    /// Slot 2: Commission rate [0, 1e18]
    pub commission: U256,
    /// Slots 3-5: SECP256k1 public key (33 bytes, compressed)
    pub secp_pubkey: [u8; 33],
    /// Slots 3-5: BLS12-381 public key (48 bytes)
    pub bls_pubkey: [u8; 48],
    /// Slot 6: Authorization address (can change commission)
    pub auth_address: Address,
    /// Slot 6: Validator flags (packed with auth_address)
    pub flags: u64,
    /// Slot 7: Unclaimed rewards in the pool
    pub unclaimed_rewards: U256,
}

impl Default for Validator {
    fn default() -> Self {
        Self {
            stake: U256::ZERO,
            accumulated_reward_per_token: U256::ZERO,
            commission: U256::ZERO,
            secp_pubkey: [0u8; 33],
            bls_pubkey: [0u8; 48],
            auth_address: Address::ZERO,
            flags: 0,
            unclaimed_rewards: U256::ZERO,
        }
    }
}

/// Validator flags
pub mod validator_flags {
    /// Validator has been slashed
    pub const SLASHED: u64 = 1 << 0;
    /// Validator has requested exit
    pub const EXIT_REQUESTED: u64 = 1 << 1;
}

/// Delegator metadata (8 storage slots).
///
/// Layout matches C++ Delegator struct.
#[derive(Debug, Clone, Default)]
pub struct Delegator {
    /// Slot 0: Active stake in consensus
    pub stake: U256,
    /// Slot 1: Last read reward per token accumulator
    pub accumulated_reward_per_token: U256,
    /// Slot 2: Unclaimed rewards
    pub rewards: U256,
    /// Slot 3: Stake activating next epoch
    pub delta_stake: U256,
    /// Slot 4: Stake activating epoch+2
    pub next_delta_stake: U256,
    /// Slot 5: Epoch when delta_stake activates
    pub delta_epoch: u64,
    /// Slot 5: Epoch when next_delta_stake activates (packed)
    pub next_delta_epoch: u64,
    // Slots 6-7: ListNode (linked list pointers) - NOT YET IMPLEMENTED
    //
    // The ListNode structure contains doubly-linked list pointers for:
    // - getDelegations: tracks which validators a delegator has staked with (inext/iprev)
    // - getDelegators: tracks which delegators have staked with a validator (anext/aprev)
    //
    // Layout (2 slots):
    //   Slot 6: inext (u64) | iprev (u64) | anext_part1 (16 bytes)
    //   Slot 7: anext_part2 (4 bytes) | aprev (20 bytes) | padding
}

/// Withdrawal request (3 storage slots).
///
/// Layout matches C++ WithdrawalRequest struct.
#[derive(Debug, Clone, Default)]
pub struct WithdrawalRequest {
    /// Slot 0: Amount being withdrawn
    pub amount: U256,
    /// Slot 1: Accumulator snapshot at undelegation time
    pub accumulator: U256,
    /// Slot 2: Epoch when request was created
    pub epoch: u64,
}

/// Epoch info returned by getEpoch().
#[derive(Debug, Clone, Default)]
pub struct EpochInfo {
    /// Current epoch number
    pub epoch: u64,
    /// Whether in epoch delay period (boundary)
    pub in_delay_period: bool,
}

/// Consensus/Snapshot view (2 storage slots).
#[derive(Debug, Clone, Default)]
pub struct ConsensusView {
    /// Slot 0: Stake at snapshot time
    pub stake: U256,
    /// Slot 1: Commission at snapshot time
    pub commission: U256,
}

impl Validator {
    /// Check if validator exists (has non-zero auth address).
    pub fn exists(&self) -> bool {
        self.auth_address != Address::ZERO
    }

    /// Check if validator has specific flag set.
    pub const fn has_flag(&self, flag: u64) -> bool {
        self.flags & flag != 0
    }
}

impl Delegator {
    /// Check if delegator exists (has any stake or pending stake).
    pub fn exists(&self) -> bool {
        self.stake != U256::ZERO
            || self.delta_stake != U256::ZERO
            || self.next_delta_stake != U256::ZERO
    }

    /// Get total stake including pending.
    #[allow(clippy::missing_const_for_fn)]
    pub fn total_stake(&self) -> U256 {
        self.stake.saturating_add(self.delta_stake).saturating_add(self.next_delta_stake)
    }
}

impl WithdrawalRequest {
    /// Check if withdrawal request exists.
    pub fn exists(&self) -> bool {
        self.amount != U256::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_exists() {
        let mut validator = Validator::default();
        assert!(!validator.exists());

        validator.auth_address = Address::new([0x11; 20]);
        assert!(validator.exists());
    }

    #[test]
    fn test_validator_flags() {
        let mut validator = Validator::default();
        assert!(!validator.has_flag(validator_flags::SLASHED));

        validator.flags = validator_flags::SLASHED;
        assert!(validator.has_flag(validator_flags::SLASHED));
        assert!(!validator.has_flag(validator_flags::EXIT_REQUESTED));

        validator.flags = validator_flags::SLASHED | validator_flags::EXIT_REQUESTED;
        assert!(validator.has_flag(validator_flags::SLASHED));
        assert!(validator.has_flag(validator_flags::EXIT_REQUESTED));
    }

    #[test]
    fn test_delegator_total_stake() {
        let delegator = Delegator {
            stake: U256::from(100),
            delta_stake: U256::from(50),
            next_delta_stake: U256::from(25),
            ..Default::default()
        };
        assert_eq!(delegator.total_stake(), U256::from(175));
    }
}
