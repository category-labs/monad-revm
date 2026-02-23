//! Block lifecycle helpers for Monad staking syscalls.
//!
//! Provides convenience functions for invoking staking syscalls at the
//! appropriate block lifecycle points. The block executor (e.g., reth block
//! builder or Foundry's EVM) should call these at the right times.
//!
//! ## Block Lifecycle
//!
//! ```text
//! ┌─ Block Start ─────────────────────────────────────────────┐
//! │ apply_syscall_reward(evm, block_author, block_reward)     │
//! ├─ Transactions ────────────────────────────────────────────┤
//! │ evm.transact(tx1)                                         │
//! │ evm.transact(tx2)                                         │
//! │ ...                                                       │
//! ├─ Epoch Boundary (if applicable) ──────────────────────────┤
//! │ apply_epoch_boundary(evm, new_epoch)                      │
//! │   ├─ syscallSnapshot()                                    │
//! │   └─ syscallOnEpochChange(new_epoch)                      │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Note on `syscallReward`
//!
//! In the Monad C++ execution client, `syscallReward` is invoked as a system
//! transaction with `msg.value` = block reward (token minting). REVM's
//! [`SystemCallEvm`] does not support value on system calls, so this helper
//! appends the block reward amount to the calldata as a 32-byte extension.
//! The precompile handler detects this extended format and uses it for reward
//! distribution.
//!
//! When called directly in Foundry with `vm.prank(SYSTEM_ADDRESS)` and
//! `{value: reward}`, the handler falls back to `msg.value`.
//!
//! The caller is also responsible for **minting** the block reward amount
//! (adding balance to `STAKING_ADDRESS`) since REVM system calls don't
//! transfer value.

use crate::staking::{constants::SYSTEM_ADDRESS, interface::IMonadStaking::*, STAKING_ADDRESS};
use alloy_sol_types::SolCall;
use revm::{
    handler::system_call::SystemCallEvm,
    primitives::{Address, Bytes, U256},
};

// ═══════════════════════════════════════════════════════════════════════════════
// Calldata Encoding
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode `syscallReward(address blockAuthor)` calldata with block reward
/// appended.
///
/// The standard ABI-encoded calldata (4 + 32 = 36 bytes) is extended with
/// 32 bytes of big-endian block reward. The precompile handler reads this
/// extension when the reward is not available via `msg.value`.
pub fn syscall_reward_calldata(block_author: Address, block_reward: U256) -> Bytes {
    let mut data = syscallRewardCall { blockAuthor: block_author }.abi_encode();
    // Append block reward as 32-byte big-endian extension
    data.extend_from_slice(&block_reward.to_be_bytes::<32>());
    data.into()
}

/// Encode `syscallSnapshot()` calldata.
pub fn syscall_snapshot_calldata() -> Bytes {
    syscallSnapshotCall {}.abi_encode().into()
}

/// Encode `syscallOnEpochChange(uint64 epoch)` calldata.
pub fn syscall_on_epoch_change_calldata(new_epoch: u64) -> Bytes {
    syscallOnEpochChangeCall { epoch: new_epoch }.abi_encode().into()
}

// ═══════════════════════════════════════════════════════════════════════════════
// SystemCallEvm Wrappers
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply staking block reward via system call.
///
/// Should be called **once per block** to distribute the block reward to the
/// block author's validator. The block reward amount is encoded in the
/// extended calldata.
///
/// **Important**: The caller is responsible for minting `block_reward` to
/// `STAKING_ADDRESS` balance beforehand (REVM system calls don't carry value).
pub fn apply_syscall_reward<EVM: SystemCallEvm>(
    evm: &mut EVM,
    block_author: Address,
    block_reward: U256,
) -> Result<EVM::ExecutionResult, EVM::Error> {
    let data = syscall_reward_calldata(block_author, block_reward);
    evm.system_call_one_with_caller(SYSTEM_ADDRESS, STAKING_ADDRESS, data)
}

/// Apply epoch snapshot via system call.
///
/// Should be called at the **start of an epoch boundary period** to snapshot
/// the current consensus validator set for reward distribution during the
/// transition.
pub fn apply_syscall_snapshot<EVM: SystemCallEvm>(
    evm: &mut EVM,
) -> Result<EVM::ExecutionResult, EVM::Error> {
    let data = syscall_snapshot_calldata();
    evm.system_call_one_with_caller(SYSTEM_ADDRESS, STAKING_ADDRESS, data)
}

/// Apply epoch change via system call.
///
/// Should be called at the **end of an epoch boundary period** to finalize the
/// epoch transition and advance the epoch counter.
pub fn apply_syscall_on_epoch_change<EVM: SystemCallEvm>(
    evm: &mut EVM,
    new_epoch: u64,
) -> Result<EVM::ExecutionResult, EVM::Error> {
    let data = syscall_on_epoch_change_calldata(new_epoch);
    evm.system_call_one_with_caller(SYSTEM_ADDRESS, STAKING_ADDRESS, data)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Composite Lifecycle Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply full epoch boundary transition.
///
/// Combines `syscallSnapshot()` and `syscallOnEpochChange(newEpoch)` in the
/// correct order. Should be called when the consensus layer signals an epoch
/// boundary.
///
/// Returns the results of both system calls.
pub fn apply_epoch_boundary<EVM: SystemCallEvm>(
    evm: &mut EVM,
    new_epoch: u64,
) -> Result<(EVM::ExecutionResult, EVM::ExecutionResult), EVM::Error> {
    let snapshot_result = apply_syscall_snapshot(evm)?;
    let epoch_change_result = apply_syscall_on_epoch_change(evm, new_epoch)?;
    Ok((snapshot_result, epoch_change_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_sol_types::SolCall;

    #[test]
    fn test_syscall_reward_calldata_selector() {
        let data = syscall_reward_calldata(Address::ZERO, U256::ZERO);
        assert_eq!(&data[..4], &syscallRewardCall::SELECTOR);
    }

    #[test]
    fn test_syscall_reward_calldata_length() {
        // Standard ABI (36) + extended reward (32) = 68 bytes
        let data = syscall_reward_calldata(Address::ZERO, U256::ZERO);
        assert_eq!(data.len(), 68);
    }

    #[test]
    fn test_syscall_reward_calldata_decodes_author() {
        let author = Address::new([0xAB; 20]);
        let data = syscall_reward_calldata(author, U256::ZERO);

        let call = syscallRewardCall::abi_decode_raw(&data[4..])
            .expect("should decode standard ABI portion");
        assert_eq!(call.blockAuthor, author);
    }

    #[test]
    fn test_syscall_reward_calldata_extended_reward() {
        let reward = U256::from(1_000_000_000_000_000_000u128); // 1 MON
        let data = syscall_reward_calldata(Address::ZERO, reward);

        // Extended reward is bytes [36..68]
        let decoded = U256::from_be_slice(&data[36..68]);
        assert_eq!(decoded, reward);
    }

    #[test]
    fn test_syscall_snapshot_calldata_selector() {
        let data = syscall_snapshot_calldata();
        assert_eq!(&data[..4], &syscallSnapshotCall::SELECTOR);
    }

    #[test]
    fn test_syscall_snapshot_calldata_length() {
        // Just selector, no params
        let data = syscall_snapshot_calldata();
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_syscall_on_epoch_change_calldata_selector() {
        let data = syscall_on_epoch_change_calldata(42);
        assert_eq!(&data[..4], &syscallOnEpochChangeCall::SELECTOR);
    }

    #[test]
    fn test_syscall_on_epoch_change_calldata_decodes_epoch() {
        let data = syscall_on_epoch_change_calldata(42);
        let call =
            syscallOnEpochChangeCall::abi_decode_raw(&data[4..]).expect("should decode epoch");
        assert_eq!(call.epoch, 42);
    }
}
