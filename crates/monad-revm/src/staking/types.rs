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

/// Validator flags matching C++ `constants.hpp`.
pub mod validator_flags {
    /// Validator is eligible for consensus (no flags set).
    pub const OK: u64 = 0;
    /// Validator stake is below the minimum threshold.
    pub const STAKE_TOO_LOW: u64 = 1 << 0;
    /// Auth address withdrew below minimum stake.
    pub const WITHDRAWN: u64 = 1 << 1;
    /// Validator slashed for double signing.
    pub const DOUBLE_SIGN: u64 = 1 << 2;
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
}

/// Linked list node stored in delegator slots 6-7.
///
/// Each delegator entry contains pointers for two intrusive doubly-linked lists:
/// - **Validator list** (`inext`/`iprev`): tracks which validators a delegator stakes with.
///   Used by `getDelegations`.
/// - **Delegator list** (`anext`/`aprev`): tracks which delegators stake with a validator.
///   Used by `getDelegators`.
///
/// Storage layout (2 slots, 56 bytes used of 64):
/// ```text
/// Slot 6: [inext: u64 (8B)] [iprev: u64 (8B)] [anext bytes 0..16 (16B)]
/// Slot 7: [anext bytes 16..20 (4B)] [aprev: address (20B)] [padding (8B)]
/// ```
#[derive(Debug, Clone, Default)]
pub struct ListNode {
    /// Next validator ID in the delegator's validator list.
    pub inext: u64,
    /// Previous validator ID in the delegator's validator list.
    pub iprev: u64,
    /// Next delegator address in the validator's delegator list.
    pub anext: Address,
    /// Previous delegator address in the validator's delegator list.
    pub aprev: Address,
}

impl ListNode {
    /// Sentinel validator ID (all bits set).
    pub const SENTINEL_VAL_ID: u64 = u64::MAX;

    /// Sentinel delegator address (all 0xFF bytes).
    pub const SENTINEL_ADDRESS: Address = Address::new([0xFF; 20]);

    /// Decode a `ListNode` from two raw storage slots (big-endian U256 values).
    pub fn from_slots(slot6: [u8; 32], slot7: [u8; 32]) -> Self {
        let inext = u64::from_be_bytes(slot6[0..8].try_into().unwrap());
        let iprev = u64::from_be_bytes(slot6[8..16].try_into().unwrap());

        let mut anext_bytes = [0u8; 20];
        anext_bytes[..16].copy_from_slice(&slot6[16..32]);
        anext_bytes[16..].copy_from_slice(&slot7[..4]);
        let anext = Address::from(anext_bytes);

        let aprev = Address::from_slice(&slot7[4..24]);

        Self { inext, iprev, anext, aprev }
    }
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
        assert!(!validator.has_flag(validator_flags::STAKE_TOO_LOW));

        validator.flags = validator_flags::STAKE_TOO_LOW;
        assert!(validator.has_flag(validator_flags::STAKE_TOO_LOW));
        assert!(!validator.has_flag(validator_flags::WITHDRAWN));
        assert!(!validator.has_flag(validator_flags::DOUBLE_SIGN));

        validator.flags = validator_flags::STAKE_TOO_LOW | validator_flags::DOUBLE_SIGN;
        assert!(validator.has_flag(validator_flags::STAKE_TOO_LOW));
        assert!(!validator.has_flag(validator_flags::WITHDRAWN));
        assert!(validator.has_flag(validator_flags::DOUBLE_SIGN));
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
