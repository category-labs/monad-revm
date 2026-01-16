//! Monad-specific EVM implementation.
//!
//! This crate provides Monad-specific customizations for REVM:
//! - Gas limit charging (no refunds)
//! - Custom precompiles
//! - Custom gas costs
//! - Custom code size limits (128KB max code, 256KB max initcode)

/// API module for building and executing Monad EVM.
pub mod api;
/// Configuration module for Monad-specific settings.
pub mod cfg;
/// EVM type aliases and builders for Monad.
pub mod evm;
/// Handler customizations for Monad execution.
pub mod handler;
/// Monad-specific instruction set with custom gas costs.
pub mod instructions;
/// Monad precompiles with custom gas pricing.
pub mod precompiles;
/// Monad specification identifiers and hardfork definitions.
pub mod spec;

pub use api::*;
pub use cfg::{MonadCfgEnv, MONAD_MAX_CODE_SIZE, MONAD_MAX_INITCODE_SIZE};
pub use evm::MonadEvm;
pub use spec::*;
