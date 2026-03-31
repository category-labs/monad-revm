use crate::{api::exec::MonadContextTr, MonadSpecId};
use revm::{
    context_interface::cfg::{GasId, GasParams},
    handler::instructions::EthInstructions,
    interpreter::{
        instructions::{instruction_table_gas_changes_spec, Instruction},
        interpreter::EthInterpreter,
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
            (GasId::cold_storage_additional_cost(), COLD_SLOAD_COST - WARM_STORAGE_READ_COST),
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

/// Create Monad instructions table with custom gas costs.
///
/// For MonadNine+ (MIP-3), all memory-expanding opcodes are replaced with
/// Monad-local handlers that use the linear cost model (`words / 2`).
pub fn monad_instructions<CTX: MonadContextTr>(spec: MonadSpecId) -> MonadInstructions<CTX> {
    let eth_spec = spec.into_eth_spec();
    let mut instructions =
        EthInstructions::new(instruction_table_gas_changes_spec(eth_spec), eth_spec);

    // MIP-3: Replace memory-expanding opcodes with linear-cost variants.
    if MonadSpecId::MonadNine.is_enabled_in(spec) {
        use crate::memory::opcodes;
        use revm::bytecode::opcode::*;
        use revm::interpreter::instructions::gas;

        // Memory opcodes
        instructions.insert_instruction(MLOAD, Instruction::new(opcodes::mload, 3));
        instructions.insert_instruction(MSTORE, Instruction::new(opcodes::mstore, 3));
        instructions.insert_instruction(MSTORE8, Instruction::new(opcodes::mstore8, 3));
        instructions.insert_instruction(MCOPY, Instruction::new(opcodes::mcopy, 3));

        // Hash
        instructions
            .insert_instruction(KECCAK256, Instruction::new(opcodes::keccak256, gas::KECCAK256));

        // Copy opcodes
        instructions.insert_instruction(CALLDATACOPY, Instruction::new(opcodes::calldatacopy, 3));
        instructions.insert_instruction(CODECOPY, Instruction::new(opcodes::codecopy, 3));
        instructions
            .insert_instruction(RETURNDATACOPY, Instruction::new(opcodes::returndatacopy, 3));
        instructions.insert_instruction(
            EXTCODECOPY,
            Instruction::new(opcodes::extcodecopy, gas::WARM_STORAGE_READ_COST),
        );

        // Log opcodes
        instructions.insert_instruction(LOG0, Instruction::new(opcodes::log::<0, _>, gas::LOG));
        instructions.insert_instruction(LOG1, Instruction::new(opcodes::log::<1, _>, gas::LOG));
        instructions.insert_instruction(LOG2, Instruction::new(opcodes::log::<2, _>, gas::LOG));
        instructions.insert_instruction(LOG3, Instruction::new(opcodes::log::<3, _>, gas::LOG));
        instructions.insert_instruction(LOG4, Instruction::new(opcodes::log::<4, _>, gas::LOG));

        // Create opcodes
        instructions
            .insert_instruction(CREATE, Instruction::new(opcodes::create::<_, false, _>, 0));
        instructions
            .insert_instruction(CREATE2, Instruction::new(opcodes::create::<_, true, _>, 0));

        // Call opcodes
        instructions
            .insert_instruction(CALL, Instruction::new(opcodes::call, gas::WARM_STORAGE_READ_COST));
        instructions.insert_instruction(
            CALLCODE,
            Instruction::new(opcodes::call_code, gas::WARM_STORAGE_READ_COST),
        );
        instructions.insert_instruction(
            DELEGATECALL,
            Instruction::new(opcodes::delegate_call, gas::WARM_STORAGE_READ_COST),
        );
        instructions.insert_instruction(
            STATICCALL,
            Instruction::new(opcodes::static_call, gas::WARM_STORAGE_READ_COST),
        );

        // Return opcodes
        instructions.insert_instruction(RETURN, Instruction::new(opcodes::ret, 0));
        instructions.insert_instruction(REVERT, Instruction::new(opcodes::revert, 0));
    }

    if MonadSpecId::MonadNext.is_enabled_in(spec) {
        use crate::page_opcode;
        use revm::bytecode::opcode::SSTORE;

        instructions.insert_instruction(SSTORE, Instruction::new(page_opcode::sstore, 0));
    }

    instructions
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
        assert_eq!(params.get(GasId::warm_storage_read_cost()), WARM_STORAGE_READ_COST);
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
