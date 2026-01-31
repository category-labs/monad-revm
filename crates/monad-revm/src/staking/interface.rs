//! Monad staking precompile Solidity interface.
//!
//! Uses alloy-sol-types' `sol!` macro to generate type-safe Call/Return
//! structs and selectors for all staking precompile functions.
//!
//! All view functions are fully implemented, including `getDelegations` and
//! `getDelegators` which traverse doubly-linked lists stored in Delegator
//! storage slots 6-7 (`ListNode` structure).

alloy_sol_types::sol! {
    /// Monad staking precompile interface at address 0x1000.
    /// Reference: https://docs.monad.xyz/developer-essentials/staking/staking-precompile
    interface IMonadStaking {
        // View functions
        function getEpoch() external returns (uint64 epoch, bool inEpochDelayPeriod);
        function getProposerValId() external returns (uint64 val_id);
        function getValidator(uint64 validatorId) external view returns (
            address authAddress, uint64 flags, uint256 stake,
            uint256 accRewardPerToken, uint256 commission, uint256 unclaimedRewards,
            uint256 consensusStake, uint256 consensusCommission,
            uint256 snapshotStake, uint256 snapshotCommission,
            bytes memory secpPubkey, bytes memory blsPubkey);
        function getDelegator(uint64 validatorId, address delegator) external returns (
            uint256 stake, uint256 accRewardPerToken, uint256 unclaimedRewards,
            uint256 deltaStake, uint256 nextDeltaStake, uint64 deltaEpoch, uint64 nextDeltaEpoch);
        function getWithdrawalRequest(uint64 validatorId, address delegator, uint8 withdrawId)
            external returns (uint256 withdrawalAmount, uint256 accRewardPerToken, uint64 withdrawEpoch);
        function getConsensusValidatorSet(uint32 startIndex)
            external returns (bool isDone, uint32 nextIndex, uint64[] memory valIds);
        function getSnapshotValidatorSet(uint32 startIndex)
            external returns (bool isDone, uint32 nextIndex, uint64[] memory valIds);
        function getExecutionValidatorSet(uint32 startIndex)
            external returns (bool isDone, uint32 nextIndex, uint64[] memory valIds);
        function getDelegations(address delegator, uint64 startValId)
            external returns (bool isDone, uint64 nextValId, uint64[] memory valIds);
        function getDelegators(uint64 validatorId, address startDelegator)
            external returns (bool isDone, address nextDelegator, address[] memory delegators);

        // State-changing functions
        function addValidator(bytes calldata payload, bytes calldata signedSecpMessage, bytes calldata signedBlsMessage)
            external payable returns (uint64 validatorId);
        function delegate(uint64 validatorId) external payable returns (bool success);
        function undelegate(uint64 validatorId, uint256 amount, uint8 withdrawId) external returns (bool success);
        function withdraw(uint64 validatorId, uint8 withdrawId) external returns (bool success);
        function compound(uint64 validatorId) external returns (bool success);
        function claimRewards(uint64 validatorId) external returns (bool success);
        function changeCommission(uint64 validatorId, uint256 commission) external returns (bool success);
        function externalReward(uint64 validatorId) external returns (bool success);

        // Syscalls
        function syscallOnEpochChange(uint64 epoch) external;
        function syscallReward(address blockAuthor) external;
        function syscallSnapshot() external;

        // Events
        event ClaimRewards(uint64 indexed validatorId, address indexed delegator, uint256 amount, uint64 epoch);
        event CommissionChanged(uint64 indexed validatorId, uint256 oldCommission, uint256 newCommission);
        event Delegate(uint64 indexed validatorId, address indexed delegator, uint256 amount, uint64 activationEpoch);
        event EpochChanged(uint64 oldEpoch, uint64 newEpoch);
        event Undelegate(uint64 indexed validatorId, address indexed delegator, uint8 withdrawId, uint256 amount, uint64 activationEpoch);
        event ValidatorCreated(uint64 indexed validatorId, address indexed authAddress, uint256 commission);
        event ValidatorRewarded(uint64 indexed validatorId, address indexed from, uint256 amount, uint64 epoch);
        event ValidatorStatusChanged(uint64 indexed validatorId, uint64 flags);
        event Withdraw(uint64 indexed validatorId, address indexed delegator, uint8 withdrawId, uint256 amount, uint64 withdrawEpoch);
    }
}
