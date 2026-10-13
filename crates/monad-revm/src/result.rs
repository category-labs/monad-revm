//! Monad transaction execution outcomes.

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
