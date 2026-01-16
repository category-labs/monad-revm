use crate::MonadSpecId;
use revm::{
    context_interface::cfg::{GasId, GasParams},
    handler::instructions::EthInstructions,
    interpreter::{
        instructions::instruction_table_gas_changes_spec, interpreter::EthInterpreter, Host,
    },
};

/// Type alias for Monad instructions.
pub type MonadInstructions<CTX> = EthInstructions<EthInterpreter, CTX>;

/// Monad-specific gas parameters for a given hardfork.
/// Override Ethereum defaults with Monad's gas costs.
///
/// Monad increases cold access costs to account for the relatively higher cost
/// of state reads from disk. See: <https://docs.monad.xyz/developer-essentials/opcode-pricing#cold-access-cost>
///
/// | Access Type | Ethereum | Monad |
/// |-------------|----------|-------|
/// | Account     | 2600     | 10100 |
/// | Storage     | 2100     | 8100  |
///
/// Warm access costs (100 gas) remain the same as Ethereum.
pub fn monad_gas_params(spec: MonadSpecId) -> GasParams {
    let eth_spec = spec.into_eth_spec();
    let mut params = GasParams::new_spec(eth_spec);

    if MonadSpecId::MonadEight.is_enabled_in(spec) {
        params.override_gas([
            // SSTORE uses full cold storage cost
            (GasId::cold_storage_cost(), COLD_SLOAD_COST),
            // SLOAD uses additional cost (cold - warm)
            (
                GasId::cold_storage_additional_cost(),
                COLD_SLOAD_COST - WARM_STORAGE_READ_COST,
            ),
            // Account access opcodes (BALANCE, EXTCODESIZE, EXTCODECOPY, EXTCODEHASH,
            // CALL, CALLCODE, DELEGATECALL, STATICCALL, SELFDESTRUCT) use additional cost
            (
                GasId::cold_account_additional_cost(),
                COLD_ACCOUNT_ACCESS_COST - WARM_STORAGE_READ_COST,
            ),
        ]);
    }

    params
}

// Create Monad instructions table with custom gas costs.
/// This function combines:
/// 1. Standard instruction table for the underlying Ethereum spec
/// 2. Any custom Monad opcodes (future)
///
/// Note: Gas params are now stored in CfgEnv, not in instructions.
/// Use `monad_gas_params()` with `MonadCfgEnv` for custom gas costs.
pub fn monad_instructions<CTX: Host>(spec: MonadSpecId) -> MonadInstructions<CTX> {
    let eth_spec = spec.into_eth_spec();
    EthInstructions::new(instruction_table_gas_changes_spec(eth_spec), eth_spec)
}

/// Monad cold storage access cost (SLOAD, SSTORE).
/// Ethereum: 2100, Monad: 8100
pub const COLD_SLOAD_COST: u64 = 8100;

/// Monad cold account access cost (BALANCE, EXTCODE*, CALL*, SELFDESTRUCT).
/// Ethereum: 2600, Monad: 10100
pub const COLD_ACCOUNT_ACCESS_COST: u64 = 10100;

/// Warm storage read cost - same as Ethereum.
pub const WARM_STORAGE_READ_COST: u64 = 100;

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::hardfork::SpecId;

    #[test]
    fn test_monad_gas_params_cold_storage_cost() {
        let params = monad_gas_params(MonadSpecId::MonadEight);
        assert_eq!(params.get(GasId::cold_storage_cost()), COLD_SLOAD_COST);
    }

    #[test]
    fn test_monad_gas_params_cold_storage_additional_cost() {
        let params = monad_gas_params(MonadSpecId::MonadEight);
        assert_eq!(
            params.get(GasId::cold_storage_additional_cost()),
            COLD_SLOAD_COST - WARM_STORAGE_READ_COST
        );
    }

    #[test]
    fn test_monad_gas_params_cold_account_additional_cost() {
        let params = monad_gas_params(MonadSpecId::MonadEight);
        assert_eq!(
            params.get(GasId::cold_account_additional_cost()),
            COLD_ACCOUNT_ACCESS_COST - WARM_STORAGE_READ_COST
        );
    }

    #[test]
    fn test_monad_gas_params_warm_storage_unchanged() {
        let params = monad_gas_params(MonadSpecId::MonadEight);
        assert_eq!(
            params.get(GasId::warm_storage_read_cost()),
            WARM_STORAGE_READ_COST
        );
    }

    #[test]
    fn test_monad_vs_ethereum_cold_costs() {
        let monad = monad_gas_params(MonadSpecId::MonadEight);
        let eth = GasParams::new_spec(SpecId::PRAGUE);

        // Monad cold storage: 8100 vs Ethereum: 2100
        assert_eq!(monad.get(GasId::cold_storage_cost()), 8100);
        assert_eq!(eth.get(GasId::cold_storage_cost()), 2100);

        // Monad cold account additional: 10000 vs Ethereum: 2500
        assert_eq!(monad.get(GasId::cold_account_additional_cost()), 10000);
        assert_eq!(eth.get(GasId::cold_account_additional_cost()), 2500);
    }
}
