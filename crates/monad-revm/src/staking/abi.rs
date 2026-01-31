//! ABI constants for staking precompile.
//!
//! Gas costs for staking operations. Selectors and encoding/decoding
//! are handled by alloy-sol-types generated types in [`super::interface`].

/// Gas costs for staking operations.
pub mod gas {
    /// getEpoch view
    pub const GET_EPOCH: u64 = 16_200;
    /// getProposerValId view
    pub const GET_PROPOSER_VAL_ID: u64 = 100;
    /// getValidator view
    pub const GET_VALIDATOR: u64 = 97_200;
    /// getDelegator view
    pub const GET_DELEGATOR: u64 = 184_900;
    /// getWithdrawalRequest view
    pub const GET_WITHDRAWAL_REQUEST: u64 = 24_300;
    /// getConsensusValidatorSet view (base cost + per element)
    pub const GET_CONSENSUS_VALIDATOR_SET_BASE: u64 = 2_100;
    /// getSnapshotValidatorSet view (base cost + per element)
    pub const GET_SNAPSHOT_VALIDATOR_SET_BASE: u64 = 2_100;
    /// getExecutionValidatorSet view (base cost + per element)
    pub const GET_EXECUTION_VALIDATOR_SET_BASE: u64 = 2_100;
    /// Per-element gas cost for validator set reads
    pub const VALIDATOR_SET_PER_ELEMENT: u64 = 2_100;

    /// getDelegations view (linked list traversal, paginated to 50 entries)
    pub const GET_DELEGATIONS: u64 = 814_000;
    /// getDelegators view (linked list traversal, paginated to 50 entries)
    pub const GET_DELEGATORS: u64 = 814_000;
}
