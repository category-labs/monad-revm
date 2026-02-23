//! ABI constants for staking precompile.
//!
//! Gas costs for staking operations. Selectors and encoding/decoding
//! are handled by alloy-sol-types generated types in [`super::interface`].

/// Gas costs for staking operations.
pub mod gas {
    /// getEpoch view
    pub const GET_EPOCH: u64 = 200;
    /// getProposerValId view
    pub const GET_PROPOSER_VAL_ID: u64 = 100;
    /// getValidator view
    pub const GET_VALIDATOR: u64 = 97_200;
    /// getDelegator view (includes warm_sstores for pull_up persistence)
    pub const GET_DELEGATOR: u64 = 184_900;
    /// getWithdrawalRequest view
    pub const GET_WITHDRAWAL_REQUEST: u64 = 24_300;
    /// getConsensusValidatorSet view (flat cost)
    pub const GET_CONSENSUS_VALIDATOR_SET: u64 = 814_000;
    /// getSnapshotValidatorSet view (flat cost)
    pub const GET_SNAPSHOT_VALIDATOR_SET: u64 = 814_000;
    /// getExecutionValidatorSet view (flat cost)
    pub const GET_EXECUTION_VALIDATOR_SET: u64 = 814_000;

    /// getDelegations view (linked list traversal, paginated to 50 entries)
    pub const GET_DELEGATIONS: u64 = 814_000;
    /// getDelegators view (linked list traversal, paginated to 50 entries)
    pub const GET_DELEGATORS: u64 = 814_000;

    /// Fallback for unknown/short selectors (C++ staking_contract.cpp:773)
    pub const FALLBACK: u64 = 40_000;

    // ═══════════════════════════════════════════════════════════════════
    // State-modifying function gas costs
    // ═══════════════════════════════════════════════════════════════════

    /// delegate(uint64) payable
    pub const DELEGATE: u64 = 260_850;
    /// undelegate(uint64, uint256, uint8)
    pub const UNDELEGATE: u64 = 147_750;
    /// withdraw(uint64, uint8)
    pub const WITHDRAW: u64 = 68_675;
    /// compound(uint64)
    pub const COMPOUND: u64 = 289_325;
    /// claimRewards(uint64)
    pub const CLAIM_REWARDS: u64 = 155_375;
    /// addValidator(bytes, bytes, bytes) payable
    pub const ADD_VALIDATOR: u64 = 505_125;
    /// changeCommission(uint64, uint256)
    pub const CHANGE_COMMISSION: u64 = 39_475;
    /// externalReward(uint64) payable
    pub const EXTERNAL_REWARD: u64 = 66_575;

    // ═══════════════════════════════════════════════════════════════════
    // Syscall gas costs (system calls are gas-free but need a budget)
    // ═══════════════════════════════════════════════════════════════════

    /// syscallReward(address) — called every block
    pub const SYSCALL_REWARD: u64 = 100_000;
    /// syscallSnapshot() — called at epoch boundary
    pub const SYSCALL_SNAPSHOT: u64 = 500_000;
    /// syscallOnEpochChange(uint64) — called at epoch boundary
    pub const SYSCALL_ON_EPOCH_CHANGE: u64 = 50_000;
}
