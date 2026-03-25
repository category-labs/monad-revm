//! Monad-specific EVM configuration.
//!
//! This module provides [`MonadCfgEnv`], a wrapper around `CfgEnv<MonadHardfork>` that
//! implements the `Cfg` trait with Monad-specific defaults.

use crate::{instructions::monad_gas_params, MonadHardfork};
use core::ops::{Deref, DerefMut};
use revm::context::{Cfg, CfgEnv};
use revm::context_interface::cfg::GasParams;

/// Monad transaction gas limit cap (30M gas).
pub const MONAD_TX_GAS_LIMIT_CAP: u64 = 30_000_000;

/// MIP-3: Maximum memory per transaction (8 MB), pooled across the call stack.
pub const MONAD_MEMORY_LIMIT: u64 = 8 * 1024 * 1024;

/// REVM's default memory limit (`2^32 - 1` bytes).
///
/// Used to recognize unnormalized `CfgEnv` values that should receive Monad's
/// chain default.
#[cfg(feature = "memory_limit")]
const REVM_DEFAULT_MEMORY_LIMIT: u64 = (1 << 32) - 1;

/// Monad maximum contract code size.
///
/// Monad uses a larger code size limit than Ethereum's EIP-170 (24KB).
/// Set to 128KB (0x20000) to allow larger contracts.
pub const MONAD_MAX_CODE_SIZE: usize = 0x20000; // 128KB

/// Monad maximum initcode size.
///
/// Following EIP-3860 pattern (2x code size), this is 256KB.
pub const MONAD_MAX_INITCODE_SIZE: usize = MONAD_MAX_CODE_SIZE * 2; // 256KB

/// Monad-specific EVM configuration.
///
/// This is a newtype wrapper around `CfgEnv<MonadHardfork>` that implements
/// the `Cfg` trait with Monad-specific defaults for:
/// - `max_code_size()`: Returns [`MONAD_MAX_CODE_SIZE`] (128KB) instead of EIP-170's 24KB
/// - `max_initcode_size()`: Returns [`MONAD_MAX_INITCODE_SIZE`] (256KB) instead of EIP-3860's 48KB
///
/// All other configuration options are delegated to the inner `CfgEnv`.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MonadCfgEnv(pub CfgEnv<MonadHardfork>);

impl MonadCfgEnv {
    /// Creates a new `MonadCfgEnv` with default Monad spec and Monad gas params.
    pub fn new() -> Self {
        let spec = MonadHardfork::default();
        let mut cfg = CfgEnv::new_with_spec_and_gas_params(spec, monad_gas_params(spec));
        Self::apply_memory_limit_default(&mut cfg);
        Self(cfg)
    }

    /// Creates a new `MonadCfgEnv` with the specified spec and Monad gas params.
    pub fn new_with_spec(spec: MonadHardfork) -> Self {
        let mut cfg = CfgEnv::new_with_spec_and_gas_params(spec, monad_gas_params(spec));
        Self::apply_memory_limit_default(&mut cfg);
        Self(cfg)
    }

    /// Returns a reference to the inner `CfgEnv`.
    pub const fn inner(&self) -> &CfgEnv<MonadHardfork> {
        &self.0
    }

    /// Returns a mutable reference to the inner `CfgEnv`.
    pub const fn inner_mut(&mut self) -> &mut CfgEnv<MonadHardfork> {
        &mut self.0
    }

    /// Consumes self and returns the inner `CfgEnv`.
    pub fn into_inner(self) -> CfgEnv<MonadHardfork> {
        self.0
    }

    /// Sets the chain ID.
    pub const fn with_chain_id(mut self, chain_id: u64) -> Self {
        self.0.chain_id = chain_id;
        self
    }

    fn apply_memory_limit_default(_cfg: &mut CfgEnv<MonadHardfork>) {
        #[cfg(feature = "memory_limit")]
        {
            let cfg = _cfg;
            let inner_limit = <CfgEnv<MonadHardfork> as Cfg>::memory_limit(cfg);
            if MonadHardfork::MonadNine.is_enabled_in(cfg.spec)
                && inner_limit == REVM_DEFAULT_MEMORY_LIMIT
            {
                cfg.memory_limit = MONAD_MEMORY_LIMIT;
            }
        }
    }
}

impl Default for MonadCfgEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl From<CfgEnv<MonadHardfork>> for MonadCfgEnv {
    fn from(mut cfg: CfgEnv<MonadHardfork>) -> Self {
        // Inject Monad-specific gas params when converting from CfgEnv.
        // This ensures downstream consumers (alloy-monad-evm, monad-foundry)
        // automatically get Monad gas costs when converting.
        cfg.set_gas_params(monad_gas_params(cfg.spec));
        Self::apply_memory_limit_default(&mut cfg);
        Self(cfg)
    }
}

impl From<MonadCfgEnv> for CfgEnv<MonadHardfork> {
    fn from(cfg: MonadCfgEnv) -> Self {
        cfg.0
    }
}

impl Deref for MonadCfgEnv {
    type Target = CfgEnv<MonadHardfork>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MonadCfgEnv {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Cfg for MonadCfgEnv {
    type Spec = MonadHardfork;

    #[inline]
    fn chain_id(&self) -> u64 {
        self.0.chain_id
    }

    #[inline]
    fn spec(&self) -> Self::Spec {
        self.0.spec
    }

    #[inline]
    fn tx_chain_id_check(&self) -> bool {
        self.0.tx_chain_id_check
    }

    #[inline]
    fn tx_gas_limit_cap(&self) -> u64 {
        self.0.tx_gas_limit_cap.unwrap_or(MONAD_TX_GAS_LIMIT_CAP)
    }

    #[inline]
    fn max_blobs_per_tx(&self) -> Option<u64> {
        self.0.max_blobs_per_tx
    }

    /// Returns Monad's max code size.
    ///
    /// Uses [`MONAD_MAX_CODE_SIZE`] as default instead of EIP-170's 24KB.
    /// Can still be overridden via `limit_contract_code_size`.
    fn max_code_size(&self) -> usize {
        self.0.limit_contract_code_size.unwrap_or(MONAD_MAX_CODE_SIZE)
    }

    /// Returns Monad's max initcode size.
    ///
    /// Uses [`MONAD_MAX_INITCODE_SIZE`] as default instead of EIP-3860's 48KB.
    /// Can still be overridden via `limit_contract_initcode_size`.
    fn max_initcode_size(&self) -> usize {
        self.0
            .limit_contract_initcode_size
            .or_else(|| self.0.limit_contract_code_size.map(|size| size.saturating_mul(2)))
            .unwrap_or(MONAD_MAX_INITCODE_SIZE)
    }

    fn is_eip3541_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_eip3541_disabled(&self.0)
    }

    fn is_eip3607_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_eip3607_disabled(&self.0)
    }

    fn is_eip7623_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_eip7623_disabled(&self.0)
    }

    fn is_balance_check_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_balance_check_disabled(&self.0)
    }

    fn is_block_gas_limit_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_block_gas_limit_disabled(&self.0)
    }

    fn is_nonce_check_disabled(&self) -> bool {
        self.0.disable_nonce_check
    }

    fn is_base_fee_check_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_base_fee_check_disabled(&self.0)
    }

    fn is_priority_fee_check_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_priority_fee_check_disabled(&self.0)
    }

    fn is_fee_charge_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_fee_charge_disabled(&self.0)
    }

    fn is_eip7708_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_eip7708_disabled(&self.0)
    }

    fn is_eip7708_delayed_burn_disabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_eip7708_delayed_burn_disabled(&self.0)
    }

    fn memory_limit(&self) -> u64 {
        let inner_limit = <CfgEnv<MonadHardfork> as Cfg>::memory_limit(&self.0);
        #[cfg(feature = "memory_limit")]
        {
            // `MonadCfgEnv` is a public tuple wrapper and can also be deserialized,
            // so callers can bypass constructors that normalize MonadNine+ defaults.
            if MonadHardfork::MonadNine.is_enabled_in(self.0.spec)
                && inner_limit == REVM_DEFAULT_MEMORY_LIMIT
            {
                return MONAD_MEMORY_LIMIT;
            }
        }
        inner_limit
    }

    fn gas_params(&self) -> &GasParams {
        &self.0.gas_params
    }

    fn is_amsterdam_eip8037_enabled(&self) -> bool {
        <CfgEnv<MonadHardfork> as Cfg>::is_amsterdam_eip8037_enabled(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_defaults() {
        let cfg = MonadCfgEnv::new();

        // Verify Monad-specific defaults
        assert_eq!(cfg.max_code_size(), MONAD_MAX_CODE_SIZE);
        assert_eq!(cfg.max_initcode_size(), MONAD_MAX_INITCODE_SIZE);

        // Verify we can still override
        let mut cfg = MonadCfgEnv::new();
        cfg.0.limit_contract_code_size = Some(100_000);
        assert_eq!(cfg.max_code_size(), 100_000);
        assert_eq!(cfg.max_initcode_size(), 200_000);
    }

    #[test]
    fn test_from_cfg_env() {
        let cfg_env = CfgEnv::new_with_spec(MonadHardfork::default());
        let monad_cfg: MonadCfgEnv = cfg_env.into();

        // Should now use Monad defaults
        assert_eq!(monad_cfg.max_code_size(), MONAD_MAX_CODE_SIZE);
    }

    #[test]
    fn test_tx_gas_limit_cap_is_monad_cap() {
        // Monad uses a 30M tx gas limit, not EIP-7825's 16.7M.
        // This must hold for all specs, including MonadNine which maps to OSAKA.
        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
            let cfg = MonadCfgEnv::new_with_spec(spec);
            assert_eq!(
                cfg.tx_gas_limit_cap(),
                MONAD_TX_GAS_LIMIT_CAP,
                "tx_gas_limit_cap should be 30M for {spec:?}, not EIP-7825's 16.7M"
            );
        }
    }

    #[test]
    fn test_eip8037_is_disabled_for_current_monad_specs() {
        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
            let cfg = MonadCfgEnv::new_with_spec(spec);
            assert!(
                !cfg.is_amsterdam_eip8037_enabled(),
                "EIP-8037 must remain disabled for {spec:?}"
            );
        }
    }

    #[test]
    fn test_tx_gas_limit_cap_respects_explicit_override() {
        let mut cfg = MonadCfgEnv::new_with_spec(MonadHardfork::MonadNine);
        cfg.0.tx_gas_limit_cap = Some(u64::MAX);
        assert_eq!(cfg.tx_gas_limit_cap(), u64::MAX);

        cfg.0.tx_gas_limit_cap = Some(12_345_678);
        assert_eq!(cfg.tx_gas_limit_cap(), 12_345_678);
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_memory_limit_defaults_to_monad_limit_for_monad_nine_and_next() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let cfg = MonadCfgEnv::new_with_spec(spec);
            assert_eq!(cfg.0.memory_limit, MONAD_MEMORY_LIMIT);
            assert_eq!(cfg.memory_limit(), MONAD_MEMORY_LIMIT);
        }
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_memory_limit_respects_explicit_override_for_monad_nine_and_next() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let mut cfg = MonadCfgEnv::new_with_spec(spec);
            cfg.0.memory_limit = 16_000;
            assert_eq!(cfg.memory_limit(), 16_000);
        }
    }

    #[test]
    fn test_memory_limit_uses_inner_value_before_monad_nine() {
        let cfg = MonadCfgEnv::new_with_spec(MonadHardfork::MonadEight);
        let inner_default = <CfgEnv<MonadHardfork> as Cfg>::memory_limit(&cfg.0);
        assert_eq!(cfg.memory_limit(), inner_default);
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_from_cfg_env_applies_monad_memory_limit_default() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let cfg_env = CfgEnv::new_with_spec(spec);
            let monad_cfg: MonadCfgEnv = cfg_env.into();
            assert_eq!(monad_cfg.0.memory_limit, MONAD_MEMORY_LIMIT);
            assert_eq!(monad_cfg.memory_limit(), MONAD_MEMORY_LIMIT);
        }
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_memory_limit_applies_monad_default_for_tuple_constructor() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let cfg = MonadCfgEnv(CfgEnv::new_with_spec(spec));
            assert_eq!(cfg.memory_limit(), MONAD_MEMORY_LIMIT);
        }
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_from_cfg_env_respects_explicit_memory_limit_override() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let mut cfg_env = CfgEnv::new_with_spec(spec);
            cfg_env.memory_limit = 16_000;

            let monad_cfg: MonadCfgEnv = cfg_env.into();
            assert_eq!(monad_cfg.0.memory_limit, 16_000);
            assert_eq!(monad_cfg.memory_limit(), 16_000);
        }
    }
}
