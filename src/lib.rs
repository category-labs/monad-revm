//! Monad-specific EVM implementation.
//!
//! This crate provides Monad-specific customizations for REVM:
//! - Gas limit charging (no refunds)
//! - Custom precompiles
//! - Custom gas costs
//! - Custom code size limits (128KB max code, 256KB max initcode)

/// Configuration module for Monad-specific settings.
pub mod cfg;
/// Monad-specific instruction set with custom gas costs.
pub mod instructions;
/// Monad precompiles with custom gas pricing.
pub mod precompiles;
/// Monad specification identifiers and hardfork definitions.
pub mod spec;

pub use cfg::{MonadCfgEnv, MONAD_MAX_CODE_SIZE, MONAD_MAX_INITCODE_SIZE};
pub use spec::*;
