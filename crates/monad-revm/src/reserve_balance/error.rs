//! Reserve-balance precompile errors.

use core::fmt;
use revm::precompile::PrecompileHalt;

/// Reserve-balance precompile errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReserveBalanceError {
    /// Unknown or short selector.
    MethodNotSupported,
    /// Nonzero value supplied to a non-payable method.
    ValueNonZero,
    /// Extra calldata past the selector.
    InvalidInput,
}

impl ReserveBalanceError {
    /// Converts the error into a precompile error.
    pub fn into_precompile_error(self) -> PrecompileHalt {
        match self {
            Self::MethodNotSupported => PrecompileHalt::Other("method not supported".into()),
            Self::ValueNonZero => PrecompileHalt::Other("value is nonzero".into()),
            Self::InvalidInput => PrecompileHalt::Other("input is invalid".into()),
        }
    }
}

impl fmt::Display for ReserveBalanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MethodNotSupported => f.write_str("method not supported"),
            Self::ValueNonZero => f.write_str("value is nonzero"),
            Self::InvalidInput => f.write_str("input is invalid"),
        }
    }
}
