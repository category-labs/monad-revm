//! Storage key generation for the staking precompile.
//!
//! Storage keys match the C++ implementation in monad/staking/staking_contract.hpp.
//! Keys are 32 bytes constructed from namespace byte + identifiers + padding.

use revm::primitives::{Address, U256};

/// Staking contract address (0x1000)
pub const STAKING_ADDRESS: Address = Address::new([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00,
]);

/// Storage namespace constants matching C++ implementation.
pub mod namespace {
    /// Consensus stake view: val_id => (stake, commission)
    pub const CONSENSUS_STAKE: u8 = 0x04;
    /// Snapshot stake view: previous epoch's stake
    pub const SNAPSHOT_STAKE: u8 = 0x05;
    /// Validator ID by secp address: address => val_id
    pub const VAL_ID_SECP: u8 = 0x06;
    /// Validator ID by BLS address: bls_address => val_id
    pub const VAL_ID_BLS: u8 = 0x07;
    /// Validator existence bitset: bucket => bitset
    pub const VAL_BITSET: u8 = 0x08;
    /// Validator execution view: val_id => Validator struct
    pub const VAL_EXECUTION: u8 = 0x09;
    /// Reward accumulators: epoch => val_id => RefCountedAccumulator
    pub const ACCUMULATOR: u8 = 0x0A;
    /// Delegator metadata: val_id => address => Delegator struct
    pub const DELEGATOR: u8 = 0x0B;
    /// Withdrawal requests: val_id => address => withdrawal_id => WithdrawalRequest
    pub const WITHDRAWAL_REQUEST: u8 = 0x0C;
}

/// Fixed storage slots for global state (namespace 0x00).
pub mod global_slots {
    use revm::primitives::U256;

    /// Current epoch number
    pub const EPOCH: U256 = U256::from_limbs([1, 0, 0, 0]);
    /// Whether in epoch delay period (boundary)
    pub const IN_BOUNDARY: U256 = U256::from_limbs([2, 0, 0, 0]);
    /// Last assigned validator ID
    pub const LAST_VAL_ID: U256 = U256::from_limbs([3, 0, 0, 0]);
    /// Current block proposer validator ID
    pub const PROPOSER_VAL_ID: U256 = U256::from_limbs([4, 0, 0, 0]);
}

/// Base addresses for validator set storage arrays.
/// These are StorageArray<u64_be> where:
/// - Slot 0: length (u64, left-aligned)
/// - Slot 1+i: element[i] (u64, left-aligned)
pub mod valset_slots {
    use revm::primitives::U256;

    /// Execution valset base address (0x0100...)
    /// Changes in real-time with validator stake
    pub const EXECUTION: U256 = U256::from_be_slice(&[
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ]);

    /// Consensus valset base address (0x0200...)
    /// Copy of execution valset with top N stake at snapshot
    pub const CONSENSUS: U256 = U256::from_be_slice(&[
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ]);

    /// Snapshot valset base address (0x0300...)
    /// Copy of consensus valset at snapshot for rewarding
    pub const SNAPSHOT: U256 = U256::from_be_slice(&[
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ]);
}

/// Validator execution storage slot offsets (8 slots total).
pub mod validator_offsets {
    /// Total stake in validator pool
    pub const STAKE: u8 = 0;
    /// Accumulated reward per token
    pub const ACCUMULATED_REWARD_PER_TOKEN: u8 = 1;
    /// Commission rate [0, 1e18]
    pub const COMMISSION: u8 = 2;
    /// Keys (secp33 + bls48) - spans 3 slots
    pub const KEYS: u8 = 3;
    /// Auth address + flags (packed)
    pub const ADDRESS_FLAGS: u8 = 6;
    /// Unclaimed rewards in pool
    pub const UNCLAIMED_REWARDS: u8 = 7;
}

/// Delegator storage slot offsets (8 slots total).
pub mod delegator_offsets {
    /// Active stake in consensus
    pub const STAKE: u8 = 0;
    /// Last read reward per token accumulator
    pub const ACCUMULATED_REWARD_PER_TOKEN: u8 = 1;
    /// Unclaimed rewards
    pub const REWARDS: u8 = 2;
    /// Stake activating next epoch
    pub const DELTA_STAKE: u8 = 3;
    /// Stake activating epoch+2
    pub const NEXT_DELTA_STAKE: u8 = 4;
    /// Epochs (delta_epoch + next_delta_epoch packed)
    pub const EPOCHS: u8 = 5;
    /// List node (linked list pointers) - spans 2 slots
    pub const LIST_NODE: u8 = 6;
}

/// WithdrawalRequest storage slot offsets (3 slots).
pub mod withdrawal_offsets {
    /// Amount being withdrawn
    pub const AMOUNT: u8 = 0;
    /// Accumulator snapshot at undelegation
    pub const ACCUMULATOR: u8 = 1;
    /// Epoch when request created
    pub const EPOCH: u8 = 2;
}

/// Generate storage key for validator execution data.
///
/// Base key format: [namespace(1)][val_id(8)][padding(23)]
/// Slot is added numerically to the base key (goes to least significant position).
pub fn validator_key(val_id: u64, slot: u8) -> U256 {
    let mut key = [0u8; 32];
    key[0] = namespace::VAL_EXECUTION;
    key[1..9].copy_from_slice(&val_id.to_be_bytes());
    // slot is added as offset to the uint256 value (at the end)
    U256::from_be_bytes(key) + U256::from(slot)
}

/// Generate storage key for delegator data.
///
/// Base key format: [namespace(1)][val_id(8)][address(20)][padding(3)]
/// Slot is added numerically to the base key.
pub fn delegator_key(val_id: u64, delegator: &Address, slot: u8) -> U256 {
    let mut key = [0u8; 32];
    key[0] = namespace::DELEGATOR;
    key[1..9].copy_from_slice(&val_id.to_be_bytes());
    key[9..29].copy_from_slice(delegator.as_slice());
    // slot is added as offset to the uint256 value
    U256::from_be_bytes(key) + U256::from(slot)
}

/// Generate storage key for withdrawal request.
///
/// Base key format: [namespace(1)][val_id(8)][address(20)][withdrawal_id(1)][padding(2)]
/// Slot is added numerically to the base key.
pub fn withdrawal_key(val_id: u64, delegator: &Address, withdrawal_id: u8, slot: u8) -> U256 {
    let mut key = [0u8; 32];
    key[0] = namespace::WITHDRAWAL_REQUEST;
    key[1..9].copy_from_slice(&val_id.to_be_bytes());
    key[9..29].copy_from_slice(delegator.as_slice());
    key[29] = withdrawal_id;
    // slot is added as offset to the uint256 value
    U256::from_be_bytes(key) + U256::from(slot)
}

/// Generate storage key for consensus view (stake/commission snapshot).
///
/// Base key format: [namespace(1)][val_id(8)][padding(23)]
/// Slot is added numerically to the base key.
pub fn consensus_view_key(val_id: u64, slot: u8) -> U256 {
    let mut key = [0u8; 32];
    key[0] = namespace::CONSENSUS_STAKE;
    key[1..9].copy_from_slice(&val_id.to_be_bytes());
    // slot is added as offset to the uint256 value
    U256::from_be_bytes(key) + U256::from(slot)
}

/// Generate storage key for snapshot view.
///
/// Base key format: [namespace(1)][val_id(8)][padding(23)]
/// Slot is added numerically to the base key.
pub fn snapshot_view_key(val_id: u64, slot: u8) -> U256 {
    let mut key = [0u8; 32];
    key[0] = namespace::SNAPSHOT_STAKE;
    key[1..9].copy_from_slice(&val_id.to_be_bytes());
    // slot is added as offset to the uint256 value
    U256::from_be_bytes(key) + U256::from(slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staking_address() {
        // Verify staking address is 0x1000
        let expected =
            Address::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x00]);
        assert_eq!(STAKING_ADDRESS, expected);
    }

    #[test]
    fn test_validator_key_format() {
        // With slot 0, key should be just base key
        let key = validator_key(1, 0);
        let bytes = key.to_be_bytes::<32>();

        // Check namespace byte
        assert_eq!(bytes[0], namespace::VAL_EXECUTION);
        // Check val_id
        assert_eq!(&bytes[1..9], &1u64.to_be_bytes());
        // Check slot 0 means all trailing bytes are zero
        assert_eq!(&bytes[9..32], &[0u8; 23]);

        // With slot 6, it should be added to the least significant position
        let key_slot6 = validator_key(1, 6);
        let bytes6 = key_slot6.to_be_bytes::<32>();
        assert_eq!(bytes6[31], 6); // slot at the end
    }

    #[test]
    fn test_delegator_key_format() {
        let delegator = Address::new([0x11; 20]);
        let key = delegator_key(42, &delegator, 2);
        let bytes = key.to_be_bytes::<32>();

        // Check namespace byte
        assert_eq!(bytes[0], namespace::DELEGATOR);
        // Check val_id
        assert_eq!(&bytes[1..9], &42u64.to_be_bytes());
        // Check delegator address
        assert_eq!(&bytes[9..29], delegator.as_slice());
        // Check slot at the end (least significant position)
        assert_eq!(bytes[31], 2);
    }

    #[test]
    fn test_withdrawal_key_format() {
        let delegator = Address::new([0x22; 20]);
        let key = withdrawal_key(100, &delegator, 5, 1);
        let bytes = key.to_be_bytes::<32>();

        // Check namespace byte
        assert_eq!(bytes[0], namespace::WITHDRAWAL_REQUEST);
        // Check val_id
        assert_eq!(&bytes[1..9], &100u64.to_be_bytes());
        // Check delegator address
        assert_eq!(&bytes[9..29], delegator.as_slice());
        // Check withdrawal_id
        assert_eq!(bytes[29], 5);
        // Check slot at the end (least significant position)
        assert_eq!(bytes[31], 1);
    }
}
