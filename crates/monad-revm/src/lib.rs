//! Monad-specific EVM implementation.
//!
//! This crate provides Monad-specific customizations for REVM:
//! - Gas limit charging (no refunds)
//! - Custom precompiles (including staking at 0x1000)
//! - Custom gas costs
//! - Custom code size limits (128KB max code, 256KB max initcode)

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
/// MIP-3: Linear memory cost model with 8 MB pooled limit.
pub mod memory;
/// Helpers for paged storage demo semantics.
pub mod page;
/// MIP-8 storage opcode overrides.
pub mod page_opcode;
/// Monad precompiles with custom gas pricing.
pub mod precompiles;
/// Monad reserve-balance precompile (0x1001).
pub mod reserve_balance;
/// Monad specification identifiers and hardfork definitions.
pub mod spec;
/// Monad staking precompile (0x1000) - read-only view methods.
pub mod staking;

pub use api::*;
pub use cfg::{MonadCfgEnv, MONAD_MAX_CODE_SIZE, MONAD_MAX_INITCODE_SIZE};
pub use chain::MonadChainContext;
pub use evm::MonadEvm;
pub use journal::{MonadJournal, MonadJournalTr};
pub use spec::*;
