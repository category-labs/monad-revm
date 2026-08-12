use crate::MonadHardfork;
use core::ops::Deref;
use revm::{
    context_interface::cfg::{GasId, GasParams},
    handler::instructions::{EthInstructions, InstructionProvider},
    interpreter::{
        instructions::{gas_table_spec, instruction_table, GasTable, Instruction},
        interpreter::EthInterpreter,
        Host, InstructionTable,
    },
};

/// Monad instruction provider with exact hardfork identity.
#[derive(Debug)]
pub struct MonadInstructions<CTX> {
    hardfork: MonadHardfork,
    inner: EthInstructions<EthInterpreter, CTX>,
}

impl<CTX: Host> Clone for MonadInstructions<CTX> {
    fn clone(&self) -> Self {
        Self { hardfork: self.hardfork, inner: self.inner.clone() }
    }
}

impl<CTX: Host> MonadInstructions<CTX> {
    /// Returns the selected Monad hardfork.
    pub const fn hardfork(&self) -> MonadHardfork {
        self.hardfork
    }

    /// Wraps a custom Ethereum instruction provider for an exact Monad hardfork.
    pub fn from_inner(
        hardfork: MonadHardfork,
        inner: EthInstructions<EthInterpreter, CTX>,
    ) -> Self {
        assert_eq!(
            inner.spec,
            hardfork.into_eth_spec(),
            "Ethereum and Monad instruction specs must match"
        );
        Self { hardfork, inner }
    }

    /// Consumes this provider and returns its Ethereum instruction tables.
    pub fn into_inner(self) -> EthInstructions<EthInterpreter, CTX> {
        self.inner
    }

    /// Inserts an instruction and its static gas cost.
    pub fn insert_instruction(
        &mut self,
        opcode: u8,
        instruction: Instruction<EthInterpreter, CTX>,
        gas: u16,
    ) {
        self.inner.insert_instruction(opcode, instruction, gas);
    }

    /// Inserts a static gas cost.
    pub fn insert_gas(&mut self, opcode: u8, gas: u16) {
        self.inner.insert_gas(opcode, gas);
    }

    /// Returns the mutable instruction table.
    pub fn instruction_table_mut(&mut self) -> &mut InstructionTable<EthInterpreter, CTX> {
        self.inner.instruction_table_mut()
    }

    /// Returns the mutable static gas table.
    pub fn gas_table_mut(&mut self) -> &mut GasTable {
        self.inner.gas_table_mut()
    }
}

impl<CTX> Deref for MonadInstructions<CTX> {
    type Target = EthInstructions<EthInterpreter, CTX>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<CTX: Host> InstructionProvider for MonadInstructions<CTX> {
    type Context = CTX;
    type InterpreterTypes = EthInterpreter;

    fn instruction_table(&self) -> &InstructionTable<Self::InterpreterTypes, Self::Context> {
        self.inner.instruction_table()
    }

    fn gas_table(&self) -> &GasTable {
        self.inner.gas_table()
    }
}

/// Instruction provider that follows Monad hardfork changes between frames.
#[auto_impl::auto_impl(&mut, Box)]
pub trait MonadInstructionProvider: InstructionProvider {
    /// Selects the instructions for a Monad hardfork.
    fn set_spec(&mut self, spec: MonadHardfork);
}

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
pub fn monad_gas_params(spec: MonadHardfork) -> GasParams {
    let eth_spec = spec.into_eth_spec();
    let mut params = GasParams::new_spec(eth_spec);

    if MonadHardfork::MonadEight.is_enabled_in(spec) {
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
pub fn monad_instructions<CTX: Host>(spec: MonadHardfork) -> MonadInstructions<CTX> {
    let eth_spec = spec.into_eth_spec();
    let mut instructions =
        EthInstructions::new(instruction_table(), gas_table_spec(eth_spec), eth_spec);

    // All supported Monad specs forbid CREATE/CREATE2 while executing on behalf of
    // an EIP-7702 delegated account.
    use crate::memory::opcodes;
    use revm::bytecode::opcode::*;
    instructions.insert_instruction(CREATE, Instruction::new(opcodes::create::<_, false, _>), 0);
    instructions.insert_instruction(CREATE2, Instruction::new(opcodes::create::<_, true, _>), 0);
    instructions.insert_instruction(
        CALL,
        Instruction::new(opcodes::call),
        WARM_STORAGE_READ_COST as u16,
    );
    instructions.insert_instruction(
        CALLCODE,
        Instruction::new(opcodes::call_code),
        WARM_STORAGE_READ_COST as u16,
    );
    instructions.insert_instruction(
        DELEGATECALL,
        Instruction::new(opcodes::delegate_call),
        WARM_STORAGE_READ_COST as u16,
    );
    instructions.insert_instruction(
        STATICCALL,
        Instruction::new(opcodes::static_call),
        WARM_STORAGE_READ_COST as u16,
    );

    // MIP-3: Replace memory-expanding opcodes with linear-cost variants.
    if MonadHardfork::MonadNine.is_enabled_in(spec) {
        use revm::interpreter::instructions::gas;

        // Memory opcodes
        instructions.insert_instruction(MLOAD, Instruction::new(opcodes::mload), 3);
        instructions.insert_instruction(MSTORE, Instruction::new(opcodes::mstore), 3);
        instructions.insert_instruction(MSTORE8, Instruction::new(opcodes::mstore8), 3);
        instructions.insert_instruction(MCOPY, Instruction::new(opcodes::mcopy), 3);

        // Hash
        instructions.insert_instruction(
            KECCAK256,
            Instruction::new(opcodes::keccak256),
            gas::KECCAK256 as u16,
        );

        // Copy opcodes
        instructions.insert_instruction(CALLDATACOPY, Instruction::new(opcodes::calldatacopy), 3);
        instructions.insert_instruction(CODECOPY, Instruction::new(opcodes::codecopy), 3);
        instructions.insert_instruction(
            RETURNDATACOPY,
            Instruction::new(opcodes::returndatacopy),
            3,
        );
        instructions.insert_instruction(
            EXTCODECOPY,
            Instruction::new(opcodes::extcodecopy),
            gas::WARM_STORAGE_READ_COST as u16,
        );

        // Log opcodes
        instructions.insert_instruction(
            LOG0,
            Instruction::new(opcodes::log::<0, _>),
            gas::LOG as u16,
        );
        instructions.insert_instruction(
            LOG1,
            Instruction::new(opcodes::log::<1, _>),
            gas::LOG as u16,
        );
        instructions.insert_instruction(
            LOG2,
            Instruction::new(opcodes::log::<2, _>),
            gas::LOG as u16,
        );
        instructions.insert_instruction(
            LOG3,
            Instruction::new(opcodes::log::<3, _>),
            gas::LOG as u16,
        );
        instructions.insert_instruction(
            LOG4,
            Instruction::new(opcodes::log::<4, _>),
            gas::LOG as u16,
        );

        // Return opcodes
        instructions.insert_instruction(RETURN, Instruction::new(opcodes::ret), 0);
        instructions.insert_instruction(REVERT, Instruction::new(opcodes::revert), 0);
    }

    MonadInstructions::from_inner(spec, instructions)
}

impl<CTX: Host> MonadInstructionProvider for MonadInstructions<CTX> {
    fn set_spec(&mut self, spec: MonadHardfork) {
        if self.hardfork != spec {
            *self = monad_instructions(spec);
        }
    }
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
    #[cfg(feature = "memory_limit")]
    use crate::cfg::MONAD_MEMORY_LIMIT;
    use crate::{
        api::{
            builder::MonadBuilder,
            default_ctx::{monad_context_with_db, MonadContext},
        },
        precompiles::MonadPrecompiles,
        reserve_balance::{
            abi::RESERVE_BALANCE_ADDRESS, interface::IReserveBalance::dippedIntoReserveCall,
            tracker::ReserveBalanceInit,
        },
        staking::{interface::IMonadStaking::getEpochCall, storage::STAKING_ADDRESS},
        MonadCfgEnv, MonadJournalTr,
    };
    use alloc::{string::String, vec, vec::Vec};
    use alloy_sol_types::SolCall;
    #[cfg(feature = "memory_limit")]
    use revm::context_interface::result::OutOfGasError;
    use revm::{
        bytecode::opcode,
        context::TxEnv,
        context_interface::{
            result::{ExecutionResult, HaltReason},
            ContextError, ContextTr,
        },
        database::InMemoryDB,
        handler::{EvmTr, PrecompileProvider},
        inspector::InspectEvm,
        interpreter::{CallInputs, InterpreterResult},
        primitives::{hardfork::SpecId, Address, AddressSet, Bytes, TxKind, U256},
        state::{Account, AccountInfo, Bytecode},
        ExecuteEvm, Inspector,
    };
    use std::{cell::RefCell, rc::Rc};

    const DUPN_OPCODE: u8 = 0xE6;
    const SWAPN_OPCODE: u8 = 0xE7;
    const EXCHANGE_OPCODE: u8 = 0xE8;

    #[test]
    fn test_monad_gas_params_cold_storage_cost() {
        let params = monad_gas_params(MonadHardfork::MonadEight);
        assert_eq!(params.get(GasId::cold_storage_cost()), COLD_SLOAD_COST);
    }

    #[test]
    fn test_monad_gas_params_cold_storage_additional_cost() {
        let params = monad_gas_params(MonadHardfork::MonadEight);
        assert_eq!(
            params.get(GasId::cold_storage_additional_cost()),
            COLD_SLOAD_COST - WARM_STORAGE_READ_COST
        );
    }

    #[test]
    fn test_monad_gas_params_cold_account_additional_cost() {
        let params = monad_gas_params(MonadHardfork::MonadEight);
        assert_eq!(
            params.get(GasId::cold_account_additional_cost()),
            COLD_ACCOUNT_ACCESS_COST - WARM_STORAGE_READ_COST
        );
    }

    #[test]
    fn test_monad_gas_params_warm_storage_unchanged() {
        let params = monad_gas_params(MonadHardfork::MonadEight);
        assert_eq!(params.get(GasId::warm_storage_read_cost()), WARM_STORAGE_READ_COST);
    }

    #[test]
    fn test_monad_vs_ethereum_cold_costs() {
        let monad = monad_gas_params(MonadHardfork::MonadEight);
        let eth = GasParams::new_spec(SpecId::PRAGUE);

        // Monad cold storage: 8100 vs Ethereum: 2100
        assert_eq!(monad.get(GasId::cold_storage_cost()), 8100);
        assert_eq!(eth.get(GasId::cold_storage_cost()), 2100);

        // Monad cold account additional: 10000 vs Ethereum: 2500
        assert_eq!(monad.get(GasId::cold_account_additional_cost()), 10000);
        assert_eq!(eth.get(GasId::cold_account_additional_cost()), 2500);
    }

    fn run_contract(spec: MonadHardfork, code: Vec<u8>) -> ExecutionResult<HaltReason> {
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
        spec: MonadHardfork,
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
        spec: MonadHardfork,
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

    fn push2(code: &mut Vec<u8>, value: u16) {
        code.push(opcode::PUSH2);
        code.extend_from_slice(&value.to_be_bytes());
    }

    fn memory_expanding_call_contract(
        opcode: u8,
        target: Address,
        input_len: u16,
        output_len: u16,
    ) -> Vec<u8> {
        let mut code = Vec::new();
        push2(&mut code, output_len);
        push2(&mut code, 0x2000);
        push2(&mut code, input_len);
        push2(&mut code, 0x1000);

        match opcode {
            opcode::CALL | opcode::CALLCODE => {
                code.push(opcode::PUSH0); // value
                code.push(opcode::PUSH20);
                code.extend_from_slice(target.as_slice());
                code.push(opcode::GAS);
                code.push(opcode);
            }
            opcode::DELEGATECALL | opcode::STATICCALL => {
                code.push(opcode::PUSH20);
                code.extend_from_slice(target.as_slice());
                code.push(opcode::GAS);
                code.push(opcode);
            }
            _ => unreachable!("only CALL-like opcodes are supported"),
        }

        code.push(opcode::STOP);
        code
    }

    fn run_memory_expanding_call(
        spec: MonadHardfork,
        opcode: u8,
        input_len: u16,
        output_len: u16,
    ) -> u64 {
        let callee = Address::from([0x44; 20]);
        let code = memory_expanding_call_contract(opcode, callee, input_len, output_len);
        let result = run_contract_with_input_and_accounts(
            spec,
            Bytecode::new_raw(Bytes::from(code)),
            Bytes::new(),
            &[(callee, Bytecode::new_raw(Bytes::new()))],
        );

        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "memory-expanding CALL-like contract should succeed on {spec:?}"
        );
        result.tx_gas_used()
    }

    const fn standard_memory_cost(words: u64) -> u64 {
        3 * words + words * words / 512
    }

    #[derive(Clone, Copy, Debug)]
    struct SwitchSpecInspector {
        target: Address,
        spec: MonadHardfork,
    }

    impl Inspector<MonadContext<InMemoryDB>> for SwitchSpecInspector {
        fn call(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &mut CallInputs,
        ) -> Option<revm::interpreter::CallOutcome> {
            if inputs.target_address == self.target {
                let mut cfg = context.cfg.clone().into_inner();
                cfg.spec = self.spec;
                context.cfg = MonadCfgEnv::from(cfg);
            }
            None
        }
    }

    #[derive(Clone, Debug)]
    struct SwitchSpecsInspector {
        specs: Vec<(Address, MonadHardfork)>,
    }

    impl Inspector<MonadContext<InMemoryDB>> for SwitchSpecsInspector {
        fn call(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &mut CallInputs,
        ) -> Option<revm::interpreter::CallOutcome> {
            if let Some((_, spec)) =
                self.specs.iter().find(|(target, _)| *target == inputs.target_address)
            {
                let mut cfg = context.cfg.clone().into_inner();
                cfg.spec = *spec;
                context.cfg = MonadCfgEnv::from(cfg);
            }
            None
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct SwitchSpecAndFailReturnInspector {
        target: Address,
        spec: MonadHardfork,
    }

    impl Inspector<MonadContext<InMemoryDB>> for SwitchSpecAndFailReturnInspector {
        fn call(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &mut CallInputs,
        ) -> Option<revm::interpreter::CallOutcome> {
            if inputs.target_address == self.target {
                let mut cfg = context.cfg.clone().into_inner();
                cfg.spec = self.spec;
                context.cfg = MonadCfgEnv::from(cfg);
            }
            None
        }

        fn call_end(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &CallInputs,
            _outcome: &mut revm::interpreter::CallOutcome,
        ) {
            if inputs.target_address == self.target {
                *context.error() = Err(ContextError::Custom("intentional return failure".into()));
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct SwitchSpecWithReserveAccountInspector {
        target: Address,
        tracked: Address,
        spec: MonadHardfork,
    }

    impl Inspector<MonadContext<InMemoryDB>> for SwitchSpecWithReserveAccountInspector {
        fn call(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &mut CallInputs,
        ) -> Option<revm::interpreter::CallOutcome> {
            if inputs.target_address == self.target {
                let mut account = Account::from(AccountInfo {
                    balance: U256::from(12_000_000_000_000_000_000u128),
                    ..Default::default()
                });
                account.info.balance = U256::from(9_000_000_000_000_000_000u128);
                account.mark_created_locally();
                account.mark_selfdestructed_locally();
                context.journaled_state.state.insert(self.tracked, account.clone());
                context
                    .journaled_state
                    .reserve_balance_mut()
                    .on_debit(Some(&account), self.tracked);

                let mut cfg = context.cfg.clone().into_inner();
                cfg.spec = self.spec;
                context.cfg = MonadCfgEnv::from(cfg);
            }
            None
        }
    }

    #[derive(Clone, Debug)]
    struct TrackingPrecompiles {
        inner: MonadPrecompiles,
        selected_specs: Rc<RefCell<Vec<MonadHardfork>>>,
    }

    impl PrecompileProvider<MonadContext<InMemoryDB>> for TrackingPrecompiles {
        type Output = InterpreterResult;

        fn set_spec(&mut self, spec: MonadHardfork) -> bool {
            self.selected_specs.borrow_mut().push(spec);
            PrecompileProvider::<MonadContext<InMemoryDB>>::set_spec(&mut self.inner, spec)
        }

        fn run(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &CallInputs,
        ) -> Result<Option<Self::Output>, String> {
            PrecompileProvider::<MonadContext<InMemoryDB>>::run(&mut self.inner, context, inputs)
        }

        fn warm_addresses(&self) -> &AddressSet {
            PrecompileProvider::<MonadContext<InMemoryDB>>::warm_addresses(&self.inner)
        }

        fn contains(&self, address: &Address) -> bool {
            PrecompileProvider::<MonadContext<InMemoryDB>>::contains(&self.inner, address)
        }
    }

    #[derive(Clone, Debug)]
    struct ReserveTrackingPrecompiles {
        inner: MonadPrecompiles,
        violations: Rc<RefCell<Vec<bool>>>,
        observed_address: Address,
    }

    impl PrecompileProvider<MonadContext<InMemoryDB>> for ReserveTrackingPrecompiles {
        type Output = InterpreterResult;

        fn set_spec(&mut self, spec: MonadHardfork) -> bool {
            PrecompileProvider::<MonadContext<InMemoryDB>>::set_spec(&mut self.inner, spec)
        }

        fn run(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &CallInputs,
        ) -> Result<Option<Self::Output>, String> {
            if inputs.bytecode_address == self.observed_address {
                self.violations
                    .borrow_mut()
                    .push(context.journaled_state.reserve_balance().has_violation());
            }
            PrecompileProvider::<MonadContext<InMemoryDB>>::run(&mut self.inner, context, inputs)
        }

        fn warm_addresses(&self) -> &AddressSet {
            PrecompileProvider::<MonadContext<InMemoryDB>>::warm_addresses(&self.inner)
        }

        fn contains(&self, address: &Address) -> bool {
            PrecompileProvider::<MonadContext<InMemoryDB>>::contains(&self.inner, address)
        }
    }

    #[derive(Clone, Debug)]
    struct FailingPrecompiles {
        inner: TrackingPrecompiles,
        fail_address: Address,
        fail_next: bool,
    }

    impl PrecompileProvider<MonadContext<InMemoryDB>> for FailingPrecompiles {
        type Output = InterpreterResult;

        fn set_spec(&mut self, spec: MonadHardfork) -> bool {
            PrecompileProvider::<MonadContext<InMemoryDB>>::set_spec(&mut self.inner, spec)
        }

        fn run(
            &mut self,
            context: &mut MonadContext<InMemoryDB>,
            inputs: &CallInputs,
        ) -> Result<Option<Self::Output>, String> {
            if self.fail_next && inputs.bytecode_address == self.fail_address {
                self.fail_next = false;
                return Err("intentional precompile failure".into());
            }
            PrecompileProvider::<MonadContext<InMemoryDB>>::run(&mut self.inner, context, inputs)
        }

        fn warm_addresses(&self) -> &AddressSet {
            PrecompileProvider::<MonadContext<InMemoryDB>>::warm_addresses(&self.inner)
        }

        fn contains(&self, address: &Address) -> bool {
            PrecompileProvider::<MonadContext<InMemoryDB>>::contains(&self.inner, address)
        }
    }

    fn store_at(offset: u32) -> Vec<u8> {
        let mut code = vec![opcode::PUSH0, opcode::PUSH3];
        code.extend_from_slice(&offset.to_be_bytes()[1..]);
        code.extend_from_slice(&[opcode::MSTORE, opcode::STOP]);
        code
    }

    fn call_then_store_at(target: Address, offset: u16) -> Vec<u8> {
        let mut code = vec![
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH20,
        ];
        code.extend_from_slice(target.as_slice());
        code.extend_from_slice(&[
            opcode::GAS,
            opcode::CALL,
            opcode::POP,
            opcode::PUSH0,
            opcode::PUSH2,
        ]);
        code.extend_from_slice(&offset.to_be_bytes());
        code.extend_from_slice(&[opcode::MSTORE, opcode::STOP]);
        code
    }

    #[cfg(feature = "memory_limit")]
    fn call_and_return_success(target: Address) -> Vec<u8> {
        let mut code = vec![
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH20,
        ];
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

    fn run_frame_spec_transition(
        parent_spec: MonadHardfork,
        child_spec: MonadHardfork,
        child_offset: u32,
        parent_offset: u16,
    ) -> u64 {
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let child = Address::from([0x33; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(
                child,
                parent_offset,
            )))),
        );
        db.insert_account_info(
            child,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(store_at(child_offset)))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector = SwitchSpecInspector { target: child, spec: child_spec };
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = TrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(parent_spec),
            selected_specs: Rc::clone(&selected_specs),
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(tx).expect("transitioning contract call should execute");
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "transitioning contract call should succeed: {parent_spec:?} -> {child_spec:?}"
        );
        assert!(
            selected_specs
                .borrow()
                .windows(3)
                .any(|specs| specs == [parent_spec, child_spec, parent_spec]),
            "precompile provider should follow and restore frame specs"
        );
        result.tx_gas_used()
    }

    fn run_immediate_precompile_transition(
        parent_spec: MonadHardfork,
        child_spec: MonadHardfork,
        parent_offset: u16,
    ) -> u64 {
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let precompile = revm::precompile::u64_to_address(4);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(
                precompile,
                parent_offset,
            )))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector = SwitchSpecInspector { target: precompile, spec: child_spec };
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = TrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(parent_spec),
            selected_specs: Rc::clone(&selected_specs),
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(tx).expect("nested precompile call should execute");
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "nested precompile call should succeed: {parent_spec:?} -> {child_spec:?}"
        );
        assert!(
            selected_specs
                .borrow()
                .windows(3)
                .any(|specs| specs == [parent_spec, child_spec, parent_spec]),
            "precompile provider should restore the parent after an immediate result"
        );
        result.tx_gas_used()
    }

    fn run_reserve_policy_transition(
        parent_spec: MonadHardfork,
        child_spec: MonadHardfork,
    ) -> (Vec<bool>, bool) {
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let child = Address::from([0x33; 20]);
        let tracked = Address::from([0x44; 20]);
        let identity = revm::precompile::u64_to_address(4);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo {
                balance: U256::from(100_000_000_000_000_000_000u128),
                ..Default::default()
            },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(child, 0)))),
        );
        db.insert_account_info(
            child,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(identity, 0)))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector =
            SwitchSpecWithReserveAccountInspector { target: child, tracked, spec: child_spec };
        let violations = Rc::new(RefCell::new(Vec::new()));
        let precompiles = ReserveTrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(parent_spec),
            violations: Rc::clone(&violations),
            observed_address: identity,
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;
        let chain = evm.ctx().chain.clone();
        let sender_account = Account::from(AccountInfo {
            balance: U256::from(100_000_000_000_000_000_000u128),
            ..Default::default()
        });
        evm.ctx().journaled_state.reserve_balance_mut().init(ReserveBalanceInit {
            chain: &chain,
            spec: parent_spec,
            sender: caller,
            effective_gas_price: 0,
            gas_limit: 1_000_000,
            sender_is_delegated: false,
            sender_account: Some(&sender_account),
        });
        evm.ctx().journaled_state.set_preserve_reserve_balance_tracker(true);

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(tx).expect("reserve-policy transition should execute");
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "reserve-policy transition should succeed: {parent_spec:?} -> {child_spec:?}"
        );

        let final_violation = evm.ctx().journal().reserve_balance().has_violation();
        let observed = violations.take();
        (observed, final_violation)
    }

    fn run_nested_exact_spec_transition() -> Vec<MonadHardfork> {
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let child = Address::from([0x33; 20]);
        let grandchild = Address::from([0x44; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(child, 0)))),
        );
        db.insert_account_info(
            child,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(grandchild, 0)))),
        );
        db.insert_account_info(
            grandchild,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(vec![opcode::STOP]))),
        );

        let parent_spec = MonadHardfork::MonadNext;
        let child_spec = MonadHardfork::MonadNine;
        let grandchild_spec = MonadHardfork::MonadEight;
        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector = SwitchSpecsInspector {
            specs: vec![(child, child_spec), (grandchild, grandchild_spec)],
        };
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = TrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(parent_spec),
            selected_specs: Rc::clone(&selected_specs),
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(tx).expect("nested hardfork transitions should execute");
        assert!(matches!(result, ExecutionResult::Success { .. }));
        selected_specs.take()
    }

    #[cfg(feature = "memory_limit")]
    fn run_memory_limit_contract(offset: u32) -> ExecutionResult<HaltReason> {
        let caller = Address::from([0x11; 20]);
        let contract = Address::from([0x22; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            contract,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(store_at(offset)))),
        );

        let mut cfg = MonadCfgEnv::new_with_spec(MonadHardfork::MonadNine);
        cfg.0.memory_limit = 128 * 1024 * 1024;
        let ctx = monad_context_with_db(db).with_cfg(cfg);
        let mut evm = ctx.build_monad();
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(contract))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        evm.transact(tx).expect("memory limit contract should execute").result
    }

    #[cfg(feature = "memory_limit")]
    fn run_frame_memory_limit_transition(
        parent_spec: MonadHardfork,
        child_spec: MonadHardfork,
        child_offset: u32,
    ) -> bool {
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let child = Address::from([0x33; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_and_return_success(child)))),
        );
        db.insert_account_info(
            child,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(store_at(child_offset)))),
        );

        let mut cfg = MonadCfgEnv::new_with_spec(parent_spec);
        cfg.0.memory_limit = 128 * 1024 * 1024;
        cfg.0.tx_gas_limit_cap = Some(u64::MAX);
        let ctx = monad_context_with_db(db).with_cfg(cfg);
        let inspector = SwitchSpecInspector { target: child, spec: child_spec };
        let mut evm = ctx.build_monad_with_inspector(inspector);
        evm.ctx().block.basefee = 0;
        evm.ctx().block.gas_limit = 300_000_000;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(300_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(tx).expect("transitioning contract call should execute");
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "transitioning contract call should succeed: {parent_spec:?} -> {child_spec:?}"
        );
        U256::from_be_slice(
            result.output().expect("parent contract should return the child success flag").as_ref(),
        ) == U256::from(1)
    }

    fn memory_expansion_delta(spec: MonadHardfork) -> u64 {
        let base_words = 1;
        let expanded_words = (0x2000 + 0x20) / 32;
        if MonadHardfork::MonadNine.is_enabled_in(spec) {
            expanded_words / 2 - base_words / 2
        } else {
            standard_memory_cost(expanded_words) - standard_memory_cost(base_words)
        }
    }

    #[test]
    fn test_instruction_provider_preserves_exact_monad_spec() {
        let mut instructions =
            monad_instructions::<MonadContext<InMemoryDB>>(MonadHardfork::MonadNine);
        assert_eq!(instructions.hardfork(), MonadHardfork::MonadNine);

        instructions.set_spec(MonadHardfork::MonadNext);
        assert_eq!(instructions.hardfork(), MonadHardfork::MonadNext);
    }

    #[test]
    fn test_call_like_memory_expansion_cost_is_spec_dependent() {
        let expanded_words = (0x2000 + 0x20) / 32;
        let standard_cost = standard_memory_cost(expanded_words);
        let mip3_cost = crate::memory::monad_memory_cost(expanded_words as usize);

        for opcode in [opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL] {
            let monad_eight_base =
                run_memory_expanding_call(MonadHardfork::MonadEight, opcode, 0, 0);
            let monad_eight_expanded =
                run_memory_expanding_call(MonadHardfork::MonadEight, opcode, 0x20, 0x20);
            assert_eq!(
                monad_eight_expanded - monad_eight_base,
                standard_cost,
                "MonadEight should use standard revm memory expansion for opcode 0x{opcode:02x}"
            );

            let monad_nine_base = run_memory_expanding_call(MonadHardfork::MonadNine, opcode, 0, 0);
            let monad_nine_expanded =
                run_memory_expanding_call(MonadHardfork::MonadNine, opcode, 0x20, 0x20);
            assert_eq!(
                monad_nine_expanded - monad_nine_base,
                mip3_cost,
                "MonadNine should use MIP-3 memory expansion for opcode 0x{opcode:02x}"
            );
        }
    }

    #[test]
    fn test_instruction_provider_follows_frame_spec_transitions() {
        for (parent_spec, child_spec) in [
            (MonadHardfork::MonadEight, MonadHardfork::MonadNine),
            (MonadHardfork::MonadNine, MonadHardfork::MonadEight),
        ] {
            let base = run_frame_spec_transition(parent_spec, child_spec, 0, 0);
            let child_expanded = run_frame_spec_transition(parent_spec, child_spec, 0x2000, 0);
            assert_eq!(
                child_expanded - base,
                memory_expansion_delta(child_spec),
                "child frame should use {child_spec:?} memory pricing"
            );

            let parent_expanded = run_frame_spec_transition(parent_spec, child_spec, 0, 0x2000);
            assert_eq!(
                parent_expanded - base,
                memory_expansion_delta(parent_spec),
                "parent frame should restore {parent_spec:?} memory pricing"
            );
        }
    }

    #[test]
    fn test_immediate_precompile_restores_parent_frame_spec() {
        for (parent_spec, child_spec) in [
            (MonadHardfork::MonadEight, MonadHardfork::MonadNine),
            (MonadHardfork::MonadNine, MonadHardfork::MonadEight),
            (MonadHardfork::MonadNext, MonadHardfork::MonadNine),
            (MonadHardfork::MonadNine, MonadHardfork::MonadNext),
        ] {
            let base = run_immediate_precompile_transition(parent_spec, child_spec, 0);
            let parent_expanded =
                run_immediate_precompile_transition(parent_spec, child_spec, 0x2000);
            assert_eq!(
                parent_expanded - base,
                memory_expansion_delta(parent_spec),
                "parent frame should restore {parent_spec:?} pricing after an immediate precompile"
            );
        }
    }

    #[test]
    fn test_exact_specs_restore_across_nested_frames() {
        let selected_specs = run_nested_exact_spec_transition();
        assert!(
            selected_specs.windows(5).any(|specs| {
                specs
                    == [
                        MonadHardfork::MonadNext,
                        MonadHardfork::MonadNine,
                        MonadHardfork::MonadEight,
                        MonadHardfork::MonadNine,
                        MonadHardfork::MonadNext,
                    ]
            }),
            "frame restoration should preserve exact Monad hardfork identity: {selected_specs:?}"
        );
    }

    #[test]
    fn test_reserve_policy_follows_and_restores_frame_spec() {
        let (observed, final_violation) =
            run_reserve_policy_transition(MonadHardfork::MonadEight, MonadHardfork::MonadNine);
        assert_eq!(observed, [false], "MonadNine child should apply the exemption");
        assert!(final_violation, "MonadEight parent policy should be restored");

        let (observed, final_violation) =
            run_reserve_policy_transition(MonadHardfork::MonadNine, MonadHardfork::MonadEight);
        assert_eq!(observed, [true], "MonadEight child should enforce the reserve");
        assert!(!final_violation, "MonadNine parent policy should be restored");
    }

    #[test]
    fn test_frame_spec_is_reset_after_precompile_error() {
        let parent_spec = MonadHardfork::MonadEight;
        let child_spec = MonadHardfork::MonadNine;
        let caller = Address::from([0x11; 20]);
        let first = Address::from([0x22; 20]);
        let second = Address::from([0x33; 20]);
        let precompile = revm::precompile::u64_to_address(4);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            first,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(precompile, 0)))),
        );
        db.insert_account_info(
            second,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(store_at(0)))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector = SwitchSpecInspector { target: precompile, spec: child_spec };
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = FailingPrecompiles {
            inner: TrackingPrecompiles {
                inner: MonadPrecompiles::new_with_spec(parent_spec),
                selected_specs: Rc::clone(&selected_specs),
            },
            fail_address: precompile,
            fail_next: true,
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let first_tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(first))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        assert!(evm.inspect_one_tx(first_tx).is_err());
        assert_eq!(evm.0.instruction.hardfork(), parent_spec);
        assert_eq!(selected_specs.borrow().last(), Some(&parent_spec));

        let mut cfg = evm.ctx().cfg.clone().into_inner();
        cfg.spec = parent_spec;
        evm.ctx().cfg = MonadCfgEnv::from(cfg);
        let second_tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(second))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.inspect_one_tx(second_tx).expect("transaction after error should execute");
        assert!(matches!(result, ExecutionResult::Success { .. }));
        assert!(
            selected_specs
                .borrow()
                .windows(3)
                .any(|specs| specs == [parent_spec, child_spec, parent_spec]),
            "the next root frame should replace the failed child provider spec"
        );
    }

    #[test]
    fn test_parent_frame_spec_is_restored_after_return_error() {
        let parent_spec = MonadHardfork::MonadNext;
        let child_spec = MonadHardfork::MonadEight;
        let caller = Address::from([0x11; 20]);
        let parent = Address::from([0x22; 20]);
        let child = Address::from([0x33; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            parent,
            AccountInfo::default()
                .with_code(Bytecode::new_raw(Bytes::from(call_then_store_at(child, 0)))),
        );
        db.insert_account_info(
            child,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(vec![opcode::STOP]))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(parent_spec));
        let inspector = SwitchSpecAndFailReturnInspector { target: child, spec: child_spec };
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = TrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(parent_spec),
            selected_specs: Rc::clone(&selected_specs),
        };
        let mut evm = ctx.build_monad_with_inspector(inspector).with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(parent))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        assert!(evm.inspect_one_tx(tx).is_err());
        assert_eq!(evm.0.instruction.hardfork(), parent_spec);
        assert_eq!(selected_specs.borrow().last(), Some(&parent_spec));
    }

    #[test]
    fn test_root_frame_spec_remains_selected_after_return() {
        let spec = MonadHardfork::MonadNext;
        let caller = Address::from([0x11; 20]);
        let contract = Address::from([0x22; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            contract,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(vec![opcode::STOP]))),
        );

        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
        let selected_specs = Rc::new(RefCell::new(Vec::new()));
        let precompiles = TrackingPrecompiles {
            inner: MonadPrecompiles::new_with_spec(spec),
            selected_specs: Rc::clone(&selected_specs),
        };
        let mut evm = ctx.build_monad().with_precompiles(precompiles);
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(contract))
            .gas_limit(1_000_000)
            .gas_price(0)
            .build_fill();
        let result = evm.transact(tx).expect("root frame should execute").result;
        assert!(matches!(result, ExecutionResult::Success { .. }));
        assert_eq!(evm.0.instruction.hardfork(), spec);
        assert_eq!(selected_specs.borrow().last(), Some(&spec));
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_monad_nine_clamps_materialized_memory_limit_at_protocol_boundary() {
        let last_word_offset = MONAD_MEMORY_LIMIT as u32 - 32;
        let at_limit_offset = MONAD_MEMORY_LIMIT as u32;

        let below_limit = run_memory_limit_contract(last_word_offset);
        assert!(
            matches!(below_limit, ExecutionResult::Success { .. }),
            "the last word ending at 8 MiB should fit"
        );

        let above_limit = run_memory_limit_contract(at_limit_offset);
        assert!(
            matches!(
                above_limit,
                ExecutionResult::Halt {
                    reason: HaltReason::OutOfGas(OutOfGasError::MemoryLimit),
                    ..
                }
            ),
            "the first word ending above 8 MiB should exceed the memory limit"
        );
    }

    #[test]
    #[cfg(feature = "memory_limit")]
    fn test_memory_limit_follows_frame_spec_transitions() {
        let last_word_offset = MONAD_MEMORY_LIMIT as u32 - 32;
        let at_limit_offset = MONAD_MEMORY_LIMIT as u32;

        assert!(run_frame_memory_limit_transition(
            MonadHardfork::MonadEight,
            MonadHardfork::MonadNine,
            last_word_offset,
        ));
        assert!(!run_frame_memory_limit_transition(
            MonadHardfork::MonadEight,
            MonadHardfork::MonadNine,
            at_limit_offset,
        ));
        assert!(run_frame_memory_limit_transition(
            MonadHardfork::MonadNine,
            MonadHardfork::MonadEight,
            at_limit_offset,
        ));
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

        let monad_eight_result = run_contract(MonadHardfork::MonadEight, clz_contract.clone());
        assert!(
            matches!(
                monad_eight_result,
                ExecutionResult::Halt { reason: HaltReason::NotActivated, .. }
            ),
            "CLZ should be unavailable before MonadNine, got {monad_eight_result:?}"
        );

        let monad_nine_result = run_contract(MonadHardfork::MonadNine, clz_contract);
        let output = monad_nine_result.output().expect("CLZ should return data on MonadNine");
        assert_eq!(
            U256::from_be_slice(output.as_ref()),
            U256::from(255),
            "CLZ(1) should return 255 on MonadNine"
        );
    }

    #[test]
    fn test_extended_stack_opcode_bytes_are_unavailable_on_monad_nine_and_next() {
        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            for opcode in [DUPN_OPCODE, SWAPN_OPCODE, EXCHANGE_OPCODE] {
                let result = run_contract(spec, vec![opcode]);
                assert!(
                    matches!(
                        result,
                        ExecutionResult::Halt {
                            reason: HaltReason::OpcodeNotFound | HaltReason::NotActivated,
                            ..
                        }
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

        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
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

        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
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

        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
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

        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
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

        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_eip7702(STAKING_ADDRESS),
                input.clone(),
                &[],
            );
            assert!(
                matches!(result, ExecutionResult::Revert { ref output, .. } if output.is_empty()),
                "delegated top-level staking call should revert with empty output on {spec:?}"
            );
        }
    }

    #[test]
    fn test_internal_call_to_delegated_staking_precompile_reverts() {
        let delegated_target = Address::from([0x55; 20]);
        let caller_code =
            call_returns_success_flag_contract(delegated_target, getEpochCall::SELECTOR);

        for spec in [MonadHardfork::MonadEight, MonadHardfork::MonadNine, MonadHardfork::MonadNext]
        {
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
                "internal CALL into delegated staking precompile should fail on {spec:?}"
            );
        }
    }

    #[test]
    fn test_top_level_delegated_reserve_balance_precompile_call_reverts() {
        let input = Bytes::from(dippedIntoReserveCall::SELECTOR.to_vec());

        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
            let result = run_contract_with_input_and_accounts(
                spec,
                Bytecode::new_eip7702(RESERVE_BALANCE_ADDRESS),
                input.clone(),
                &[],
            );
            assert!(
                matches!(result, ExecutionResult::Revert { ref output, .. } if output.is_empty()),
                "delegated top-level reserve-balance call should revert with empty output on {spec:?}"
            );
        }
    }

    #[test]
    fn test_internal_call_to_delegated_reserve_balance_precompile_reverts() {
        let delegated_target = Address::from([0x66; 20]);
        let caller_code =
            call_returns_success_flag_contract(delegated_target, dippedIntoReserveCall::SELECTOR);

        for spec in [MonadHardfork::MonadNine, MonadHardfork::MonadNext] {
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
                "internal CALL into delegated reserve-balance precompile should fail on {spec:?}"
            );
        }
    }

    #[test]
    fn test_create_still_succeeds_for_regular_contracts() {
        let result = run_contract(
            MonadHardfork::MonadNine,
            vec![opcode::PUSH0, opcode::PUSH0, opcode::PUSH0, opcode::CREATE, opcode::STOP],
        );
        assert!(
            matches!(result, ExecutionResult::Success { .. }),
            "regular CREATE should still succeed, got {result:?}"
        );
    }
}
