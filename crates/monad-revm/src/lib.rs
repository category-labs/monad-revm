//! Monad-specific EVM implementation.
//!
//! This crate provides Monad-specific customizations for REVM:
//! - Full gas-limit charging without refunds
//! - Monad opcode gas and bytecode size limits
//! - MIP-3 memory pricing and feature-gated pooled memory limits
//! - Per-frame Monad hardfork instruction and precompile selection
//! - Repriced and Monad-specific protocol precompiles

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(test)]
extern crate std;

/// API module for building and executing Monad EVM.
pub mod api;
/// Configuration module for Monad-specific settings.
pub mod cfg;
/// Monad chain context used for reserve-balance decisions.
pub mod chain;
/// EVM type aliases and builders for Monad.
pub mod evm;
/// Handler customizations for Monad execution.
pub mod handler;
/// Monad-specific instruction set with custom gas costs.
pub mod instructions;
/// Monad journal wrapper with reserve-balance tracking.
pub mod journal;
/// MIP-3 linear memory costs with a feature-gated 8 MiB pooled limit.
pub mod memory;
/// Monad precompiles with custom gas pricing.
pub mod precompiles;
/// Monad reserve-balance precompile (0x1001).
pub mod reserve_balance;
/// Monad specification identifiers and hardfork definitions.
pub mod spec;
/// Monad staking precompile (0x1000).
pub mod staking;

pub use api::*;
pub use cfg::{MonadCfgEnv, MONAD_MAX_CODE_SIZE, MONAD_MAX_INITCODE_SIZE};
pub use chain::MonadChainContext;
pub use evm::MonadEvm;
pub use journal::{MonadJournal, MonadJournalTr};
pub use spec::*;
