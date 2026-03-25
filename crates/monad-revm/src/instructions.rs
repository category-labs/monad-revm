use crate::MonadSpecId;
use revm::{
    context_interface::cfg::{GasId, GasParams},
    handler::instructions::EthInstructions,
    interpreter::{
        instructions::{instruction_table_gas_changes_spec, Instruction},
        interpreter::EthInterpreter,
        Host,
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
/// For all supported Monad specs, CREATE/CREATE2 use Monad-local handlers so
/// delegated accounts cannot create contracts. MonadNine+ additionally replaces
/// memory-expanding opcodes with linear-cost MIP-3 handlers (`words / 2`).
pub fn monad_instructions<CTX: Host>(spec: MonadSpecId) -> MonadInstructions<CTX> {
    let eth_spec = spec.into_eth_spec();
    let mut instructions =
        EthInstructions::new(instruction_table_gas_changes_spec(eth_spec), eth_spec);

    // All supported Monad specs forbid CREATE/CREATE2 while executing on behalf of
    // an EIP-7702 delegated account.
    use crate::memory::opcodes;
    use revm::bytecode::opcode::*;
    instructions.insert_instruction(CREATE, Instruction::new(opcodes::create::<_, false, _>, 0));
    instructions.insert_instruction(CREATE2, Instruction::new(opcodes::create::<_, true, _>, 0));
    instructions.insert_instruction(CALL, Instruction::new(opcodes::call, WARM_STORAGE_READ_COST));
    instructions
        .insert_instruction(CALLCODE, Instruction::new(opcodes::call_code, WARM_STORAGE_READ_COST));
    instructions.insert_instruction(
        DELEGATECALL,
        Instruction::new(opcodes::delegate_call, WARM_STORAGE_READ_COST),
    );
    instructions.insert_instruction(
        STATICCALL,
        Instruction::new(opcodes::static_call, WARM_STORAGE_READ_COST),
    );

    // MIP-3: Replace memory-expanding opcodes with linear-cost variants.
    if MonadSpecId::MonadNine.is_enabled_in(spec) {
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

        // Return opcodes
        instructions.insert_instruction(RETURN, Instruction::new(opcodes::ret, 0));
        instructions.insert_instruction(REVERT, Instruction::new(opcodes::revert, 0));
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
    use crate::{
        api::{builder::MonadBuilder, default_ctx::monad_context_with_db},
        reserve_balance::{
            abi::RESERVE_BALANCE_ADDRESS, interface::IReserveBalance::dippedIntoReserveCall,
        },
        staking::{interface::IMonadStaking::getEpochCall, storage::STAKING_ADDRESS},
        MonadCfgEnv,
    };
    use alloy_sol_types::SolCall;
    use revm::primitives::hardfork::SpecId;
    use revm::{
        bytecode::opcode,
        context::TxEnv,
        context_interface::result::{ExecutionResult, HaltReason},
        database::InMemoryDB,
        handler::EvmTr,
        primitives::{Address, Bytes, TxKind, U256},
        state::{AccountInfo, Bytecode},
        ExecuteEvm,
    };

    const DUPN_OPCODE: u8 = 0xE6;
    const SWAPN_OPCODE: u8 = 0xE7;
    const EXCHANGE_OPCODE: u8 = 0xE8;

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

    fn run_contract(spec: MonadSpecId, code: Vec<u8>) -> ExecutionResult<HaltReason> {
        let caller = Address::from([0x11; 20]);
        let contract = Address::from([0x22; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            contract,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(code))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
        let mut evm = ctx.build_monad();
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(contract))
            .gas_limit(100_000)
            .gas_price(0)
            .build_fill();

        evm.transact(tx).expect("contract call should execute").result
    }

    fn run_delegated_contract(
        spec: MonadSpecId,
        target_code: Bytecode,
        delegated_address: Address,
        delegated_code: Vec<u8>,
        extra_accounts: &[(Address, Bytecode)],
    ) -> ExecutionResult<HaltReason> {
        let caller = Address::from([0x11; 20]);
        let target = Address::from([0x22; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(target, AccountInfo::default().with_code(target_code));
        db.insert_account_info(
            delegated_address,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(delegated_code))),
        );
        for (address, code) in extra_accounts {
            db.insert_account_info(*address, AccountInfo::default().with_code(code.clone()));
        }

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
        let mut evm = ctx.build_monad();
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(target))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();

        evm.transact(tx).expect("delegated contract call should execute").result
    }

    fn run_contract_with_input_and_accounts(
        spec: MonadSpecId,
        target_code: Bytecode,
        input: Bytes,
        extra_accounts: &[(Address, Bytecode)],
    ) -> ExecutionResult<HaltReason> {
        let caller = Address::from([0x11; 20]);
        let target = Address::from([0x22; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(target, AccountInfo::default().with_code(target_code));
        for (address, code) in extra_accounts {
            db.insert_account_info(*address, AccountInfo::default().with_code(code.clone()));
        }

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
        let mut evm = ctx.build_monad();
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(target))
            .gas_limit(1_000_000)
            .gas_price(0)
            .data(input)
            .build_fill();

        evm.transact(tx).expect("contract call should execute").result
    }

    fn call_returns_success_flag_contract(target: Address, selector: [u8; 4]) -> Vec<u8> {
        let mut code = vec![opcode::PUSH4];
        code.extend_from_slice(&selector);
        code.extend_from_slice(&[
            opcode::PUSH1,
            0x1c,
            opcode::MSTORE,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH1,
            0x04,
            opcode::PUSH1,
            0x1c,
            opcode::PUSH0,
            opcode::PUSH20,
        ]);
        code.extend_from_slice(target.as_slice());
        code.extend_from_slice(&[
            opcode::GAS,
            opcode::CALL,
            opcode::PUSH0,
            opcode::MSTORE,
            opcode::PUSH1,
            0x20,
            opcode::PUSH0,
            opcode::RETURN,
        ]);
        code
    }

    #[test]
    fn test_clz_is_only_available_on_monad_nine() {
        let clz_contract = vec![
            opcode::PUSH1,
            0x01,
            opcode::CLZ,
            opcode::PUSH1,
            0x00,
            opcode::MSTORE,
            opcode::PUSH1,
            0x20,
            opcode::PUSH1,
            0x00,
            opcode::RETURN,
        ];

        let monad_eight_result = run_contract(MonadSpecId::MonadEight, clz_contract.clone());
        assert!(
            matches!(
                monad_eight_result,
                ExecutionResult::Halt { reason: HaltReason::NotActivated, .. }
            ),
            "CLZ should be unavailable before MonadNine, got {monad_eight_result:?}"
        );

        let monad_nine_result = run_contract(MonadSpecId::MonadNine, clz_contract);
        let output = monad_nine_result.output().expect("CLZ should return data on MonadNine");
        assert_eq!(
            U256::from_be_slice(output.as_ref()),
            U256::from(255),
            "CLZ(1) should return 255 on MonadNine"
        );
    }

    #[test]
    fn test_extended_stack_opcode_bytes_are_unknown_on_monad_nine_and_next() {
        for spec in [MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            for opcode in [DUPN_OPCODE, SWAPN_OPCODE, EXCHANGE_OPCODE] {
                let result = run_contract(spec, vec![opcode]);
                assert!(
                    matches!(
                        result,
                        ExecutionResult::Halt { reason: HaltReason::OpcodeNotFound, .. }
                    ),
                    "opcode 0x{opcode:02x} should be unavailable on {spec:?}, got {result:?}"
                );
            }
        }
    }

    #[test]
    fn test_jumpdest_after_unknown_extended_stack_opcode_byte_is_reachable() {
        let contract = vec![
            opcode::PUSH1,
            0x04,
            opcode::JUMP,
            DUPN_OPCODE,
            opcode::JUMPDEST,
            opcode::PUSH1,
            0x2a,
            opcode::PUSH1,
            0x00,
            opcode::MSTORE,
            opcode::PUSH1,
            0x20,
            opcode::PUSH1,
            0x00,
            opcode::RETURN,
        ];

        for spec in [MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_contract(spec, contract.clone());
            let output = result.output().expect("jump target should execute successfully");
            assert_eq!(
                U256::from_be_slice(output.as_ref()),
                U256::from(42),
                "jumpdest after 0xE6 should remain reachable on {spec:?}"
            );
        }
    }

    #[test]
    fn test_create_is_rejected_for_delegated_accounts() {
        let delegated_address = Address::from([0x33; 20]);
        let delegated_code = vec![opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::CREATE];

        for spec in [MonadSpecId::MonadEight, MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_delegated_contract(
                spec,
                Bytecode::new_eip7702(delegated_address),
                delegated_address,
                delegated_code.clone(),
                &[],
            );
            assert!(
                matches!(
                    result,
                    ExecutionResult::Halt { reason: HaltReason::NotActivated, .. }
                ),
                "CREATE should halt with NotActivated for delegated accounts on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_create2_is_rejected_for_delegated_accounts() {
        let delegated_address = Address::from([0x33; 20]);
        let delegated_code =
            vec![opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::CREATE2];

        for spec in [MonadSpecId::MonadEight, MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_delegated_contract(
                spec,
                Bytecode::new_eip7702(delegated_address),
                delegated_address,
                delegated_code.clone(),
                &[],
            );
            assert!(
                matches!(
                    result,
                    ExecutionResult::Halt { reason: HaltReason::NotActivated, .. }
                ),
                "CREATE2 should halt with NotActivated for delegated accounts on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_nested_delegatecall_to_create2_only_fails_for_delegated_accounts() {
        let delegated_address = Address::from([0x33; 20]);
        let creator = Address::from([0x44; 20]);

        let mut delegated_code =
            vec![opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::PUSH20];
        delegated_code.extend_from_slice(creator.as_slice());
        delegated_code.extend_from_slice(&[
            opcode::GAS,
            opcode::DELEGATECALL,
            opcode::PUSH1,
            0x1f,
            opcode::JUMPI,
            opcode::INVALID,
            opcode::JUMPDEST,
            opcode::STOP,
        ]);

        let creator_code = Bytecode::new_raw(Bytes::from(vec![
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::CREATE2,
        ]));

        for spec in [MonadSpecId::MonadEight, MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let delegated_result = run_delegated_contract(
                spec,
                Bytecode::new_eip7702(delegated_address),
                delegated_address,
                delegated_code.clone(),
                &[(creator, creator_code.clone())],
            );
            assert!(
                matches!(
                    delegated_result,
                    ExecutionResult::Halt { reason: HaltReason::InvalidFEOpcode, .. }
                ),
                "nested delegatecall should hit the INVALID sentinel when delegated CREATE2 fails on {spec:?}, got {delegated_result:?}"
            );

            let regular_result = run_delegated_contract(
                spec,
                Bytecode::new_raw(Bytes::from(delegated_code.clone())),
                delegated_address,
                delegated_code.clone(),
                &[(creator, creator_code.clone())],
            );
            assert!(
                matches!(regular_result, ExecutionResult::Success { .. }),
                "nested delegatecall should succeed for a regular contract on {spec:?}, got {regular_result:?}"
            );
        }
    }

    #[test]
    fn test_top_level_delegated_staking_precompile_call_reverts() {
        let input = Bytes::from(getEpochCall::SELECTOR.to_vec());

        for spec in [MonadSpecId::MonadEight, MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_eip7702(STAKING_ADDRESS),
                input.clone(),
                &[],
            );
            assert!(
                matches!(result, ExecutionResult::Revert { ref output, .. } if output.is_empty()),
                "delegated top-level staking call should revert with empty output on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_internal_call_to_delegated_staking_precompile_reverts() {
        let delegated_target = Address::from([0x55; 20]);
        let caller_code =
            call_returns_success_flag_contract(delegated_target, getEpochCall::SELECTOR);

        for spec in [MonadSpecId::MonadEight, MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_raw(Bytes::from(caller_code.clone())),
                Bytes::new(),
                &[(delegated_target, Bytecode::new_eip7702(STAKING_ADDRESS))],
            );
            let output = result.output().expect("CALL result contract should return output");
            assert_eq!(
                U256::from_be_slice(output.as_ref()),
                U256::ZERO,
                "internal CALL into delegated staking precompile should fail on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_top_level_delegated_reserve_balance_precompile_call_reverts() {
        let input = Bytes::from(dippedIntoReserveCall::SELECTOR.to_vec());

        for spec in [MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_eip7702(RESERVE_BALANCE_ADDRESS),
                input.clone(),
                &[],
            );
            assert!(
                matches!(result, ExecutionResult::Revert { ref output, .. } if output.is_empty()),
                "delegated top-level reserve-balance call should revert with empty output on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_internal_call_to_delegated_reserve_balance_precompile_reverts() {
        let delegated_target = Address::from([0x66; 20]);
        let caller_code =
            call_returns_success_flag_contract(delegated_target, dippedIntoReserveCall::SELECTOR);

        for spec in [MonadSpecId::MonadNine, MonadSpecId::MonadNext] {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_raw(Bytes::from(caller_code.clone())),
                Bytes::new(),
                &[(delegated_target, Bytecode::new_eip7702(RESERVE_BALANCE_ADDRESS))],
            );
            let output = result.output().expect("CALL result contract should return output");
            assert_eq!(
                U256::from_be_slice(output.as_ref()),
                U256::ZERO,
                "internal CALL into delegated reserve-balance precompile should fail on {spec:?}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_create_still_succeeds_for_regular_contracts() {
        let result = run_contract(
            MonadSpecId::MonadNine,
            vec![opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::CREATE, opcode::STOP],
        );
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "regular CREATE should still succeed, got {result:?}"
        );
    }
}
