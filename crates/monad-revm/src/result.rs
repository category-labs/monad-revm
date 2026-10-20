//! Monad transaction execution outcomes.
//!
//! Ordinary transaction results retain REVM's [`ExecutionResult`] and [`ResultGas`] types.
//! [`ExecutionResult::tx_gas_used`] and [`ResultGas::block_regular_gas_used`] report the full
//! transaction gas limit for receipt and block accounting, including reverts and halts.
//! [`ResultGas::total_gas_spent`] retains execution consumption, including intrinsic gas,
//! for estimation, profiling, and gas reports.
//!
//! Monad represents its minimum transaction charge with [`ResultGas::floor_gas`]. This floor
//! is the gas limit, independently of Ethereum's EIP-7623 calldata floor, and does not change
//! interpreter gas or enable an Ethereum fork rule. System calls bypass transaction charging
//! and retain their execution-only gas accounting; their results do not belong in receipts.
//!
//! [`ExecutionResult`]: revm::context_interface::result::ExecutionResult
//! [`ExecutionResult::tx_gas_used`]: revm::context_interface::result::ExecutionResult::tx_gas_used
//! [`ResultGas`]: revm::context_interface::result::ResultGas
//! [`ResultGas::block_regular_gas_used`]: revm::context_interface::result::ResultGas::block_regular_gas_used
//! [`ResultGas::total_gas_spent`]: revm::context_interface::result::ResultGas::total_gas_spent
//! [`ResultGas::floor_gas`]: revm::context_interface::result::ResultGas::floor_gas

use core::fmt;
use revm::context_interface::result::HaltReason;

/// Exceptional transaction outcomes, including Monad's reserve-balance rejection.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MonadHaltReason {
    /// An Ethereum execution halt.
    Ethereum(HaltReason),
    /// Execution ended below an account's protected reserve balance.
    ReserveBalanceViolation,
}

impl From<HaltReason> for MonadHaltReason {
    fn from(reason: HaltReason) -> Self {
        Self::Ethereum(reason)
    }
}

impl fmt::Display for MonadHaltReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ethereum(reason) => reason.fmt(f),
            Self::ReserveBalanceViolation => f.write_str("reserve balance violation"),
        }
    }
}

impl core::error::Error for MonadHaltReason {}
