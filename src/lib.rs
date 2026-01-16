//! Monad-specific EVM implementation.
//!
//! This crate provides Monad-specific customizations for REVM:
//! - Gas limit charging (no refunds)
//! - Custom precompiles
//! - Custom gas costs
//! - Custom code size limits (128KB max code, 256KB max initcode)

/// Monad specification identifiers and hardfork definitions.
pub mod spec;

pub use spec::*;
