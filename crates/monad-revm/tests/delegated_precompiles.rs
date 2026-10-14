//! EIP-7702 precompile execution and gas regressions.

use monad_revm::{
    monad_context_with_db, precompiles::MonadPrecompiles,
    reserve_balance::abi::RESERVE_BALANCE_ADDRESS, staking::STAKING_ADDRESS, MonadBuilder,
    MonadCfgEnv, MonadContext, MonadHaltReason, MonadHardfork,
};
use revm::{
    bytecode::opcode,
    context::{result::ExecutionResult, TxEnv},
    context_interface::{
        either::Either,
        transaction::{Authorization, RecoveredAuthority, RecoveredAuthorization},
    },
    database::InMemoryDB,
    handler::PrecompileProvider,
    inspector::InspectEvm,
    interpreter::{CallInputs, CallOutcome, Gas, InstructionResult, InterpreterResult},
    precompile::u64_to_address,
    primitives::{Address, AddressSet, Bytes, U256},
    state::{AccountInfo, Bytecode},
    ExecuteEvm, Inspector,
};
use rstest::rstest;

const SENDER: Address = Address::repeat_byte(0x11);
const AUTHORITY: Address = Address::repeat_byte(0x22);
const PROXY: Address = Address::repeat_byte(0x33);
const CALL_GAS: u64 = 100_000;
const CALLDATA: &[u8] = b"hello";

#[derive(Debug, Default)]
struct CallRecorder {
    started: Vec<CallInputs>,
    ended: Vec<(CallInputs, CallOutcome)>,
}

impl Inspector<MonadContext<InMemoryDB>> for CallRecorder {
    fn call(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        self.started.push(inputs.clone());
        None
    }

    fn call_end(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        self.ended.push((inputs.clone(), outcome.clone()));
    }
}

fn funded_db() -> InMemoryDB {
    let mut db = InMemoryDB::default();
    db.insert_account_info(
        SENDER,
        AccountInfo { balance: U256::from(10).pow(U256::from(27)), ..Default::default() },
    );
    db.insert_account_info(STAKING_ADDRESS, AccountInfo::default());
    db
}

fn forward_call(target: Address, call_opcode: u8, gas: u64) -> Bytecode {
    let mut code = vec![
        // Copy the transaction input into memory.
        opcode::CALLDATASIZE,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::CALLDATACOPY,
        // Request no output copy; return the actual returndata below.
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::CALLDATASIZE,
        opcode::PUSH0,
    ];
    if matches!(call_opcode, opcode::CALL | opcode::CALLCODE) {
        code.push(opcode::PUSH0);
    }
    code.push(opcode::PUSH20);
    code.extend_from_slice(target.as_slice());
    code.push(opcode::PUSH8);
    code.extend_from_slice(&gas.to_be_bytes());
    code.extend_from_slice(&[
        call_opcode,
        // Return the call status followed by its complete returndata.
        opcode::PUSH0,
        opcode::MSTORE,
        opcode::RETURNDATASIZE,
        opcode::PUSH0,
        opcode::PUSH1,
        32,
        opcode::RETURNDATACOPY,
        opcode::RETURNDATASIZE,
        opcode::PUSH1,
        32,
        opcode::ADD,
        opcode::PUSH0,
        opcode::RETURN,
    ]);
    Bytecode::new_raw(code.into())
}

fn execute(
    spec: MonadHardfork,
    db: InMemoryDB,
    target: Address,
    call_opcode: u8,
    call_gas: u64,
) -> (ExecutionResult<MonadHaltReason>, CallRecorder) {
    let (result, recorder, _) = execute_with_precompiles(
        spec,
        db,
        target,
        call_opcode,
        call_gas,
        MonadPrecompiles::new_with_spec(spec),
    );
    (result, recorder)
}

fn execute_with_precompiles<P>(
    spec: MonadHardfork,
    mut db: InMemoryDB,
    target: Address,
    call_opcode: u8,
    call_gas: u64,
    precompiles: P,
) -> (ExecutionResult<MonadHaltReason>, CallRecorder, P)
where
    P: PrecompileProvider<MonadContext<InMemoryDB>, Output = InterpreterResult>,
{
    let destination = if call_opcode == 0 {
        target
    } else {
        db.insert_account_info(
            PROXY,
            AccountInfo::default().with_code(forward_call(target, call_opcode, call_gas)),
        );
        PROXY
    };
    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm =
        context.build_monad_with_inspector(CallRecorder::default()).with_precompiles(precompiles);
    let result = evm
        .inspect_one_tx(
            TxEnv::builder()
                .caller(SENDER)
                .to(destination)
                .data(Bytes::from_static(CALLDATA))
                .gas_limit(1_000_000)
                .gas_price(0)
                .build_fill(),
        )
        .unwrap();
    (result, evm.0.inspector, evm.0.precompiles)
}

fn assert_output(
    result: &ExecutionResult<MonadHaltReason>,
    call_opcode: u8,
    success: bool,
    data: &[u8],
) {
    if call_opcode == 0 {
        assert_eq!(result.is_success(), success);
        assert_eq!(result.output().unwrap().as_ref(), data);
    } else {
        assert!(result.is_success());
        let mut expected = U256::from(success as u8).to_be_bytes::<32>().to_vec();
        expected.extend_from_slice(data);
        assert_eq!(result.output().unwrap().as_ref(), expected);
    }
}

fn assert_inspector_addresses(recorder: &CallRecorder, call_opcode: u8, target: Address) {
    let addresses =
        recorder.started.iter().map(|inputs| inputs.bytecode_address).collect::<Vec<_>>();
    assert_eq!(addresses, if call_opcode == 0 { vec![target] } else { vec![PROXY, target] });
}

#[rstest]
fn delegated_ethereum_precompiles_execute_empty_code(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
    #[values(0, CALL_GAS)] call_gas: u64,
) {
    for target in MonadPrecompiles::new_with_spec(spec).precompiles().addresses() {
        let mut db = funded_db();
        db.insert_account_info(
            AUTHORITY,
            AccountInfo::default().with_code(Bytecode::new_eip7702(*target)),
        );
        let (result, recorder) = execute(spec, db, AUTHORITY, call_opcode, call_gas);
        assert_inspector_addresses(&recorder, call_opcode, AUTHORITY);
        assert_output(&result, call_opcode, true, &[]);

        // Canonical delegated precompiles return empty success with all msg.gas remaining.
        let (inputs, outcome) = &recorder.ended[0];
        assert!(outcome.result.result.is_ok(), "target {target}");
        assert!(outcome.result.output.is_empty(), "target {target}");
        assert_eq!(outcome.result.gas.remaining(), inputs.gas_limit, "target {target}");
        assert_eq!(outcome.result.gas.total_gas_spent(), 0, "target {target}");

        let expected_gas = if call_opcode == 0 {
            // EIP-7623 calldata floor exceeds the intrinsic gas for this empty execution.
            21_200
        } else {
            // Intrinsic gas, forwarding bytecode, cold authority, and warm delegation target.
            31_331
                + if spec == MonadHardfork::MonadEight { 3 } else { 0 }
                + if matches!(call_opcode, opcode::CALL | opcode::CALLCODE) { 2 } else { 0 }
        };
        assert_eq!(result.tx_gas_used(), expected_gas, "target {target}");
    }
}

#[rstest]
fn direct_identity_precompile_still_executes(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
) {
    let (result, recorder) = execute(spec, funded_db(), u64_to_address(4), call_opcode, CALL_GAS);
    assert_inspector_addresses(&recorder, call_opcode, u64_to_address(4));
    assert_output(&result, call_opcode, true, CALLDATA);
    let (inputs, outcome) = &recorder.ended[0];
    assert_eq!(outcome.result.gas.total_gas_spent(), 18);
    assert_eq!(outcome.result.gas.remaining(), inputs.gas_limit - 18);
}

#[rstest]
fn delegated_contract_can_call_identity_directly(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
) {
    let mut db = funded_db();
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(PROXY)),
    );
    db.insert_account_info(
        PROXY,
        AccountInfo::default().with_code(forward_call(u64_to_address(4), call_opcode, CALL_GAS)),
    );
    let (result, recorder) = execute(spec, db, AUTHORITY, 0, CALL_GAS);
    assert_output(&result, call_opcode, true, CALLDATA);
    let (inputs, outcome) = &recorder.ended[0];
    assert_eq!(outcome.result.gas.total_gas_spent(), 18);
    assert_eq!(outcome.result.gas.remaining(), inputs.gas_limit - 18);
}

#[derive(Default)]
struct RootInputRecorder {
    started: Option<CallInputs>,
    ended: Option<(CallInputs, CallOutcome, U256)>,
}

impl Inspector<MonadContext<InMemoryDB>> for RootInputRecorder {
    fn call(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        assert!(self.started.replace(inputs.clone()).is_none());
        None
    }

    fn call_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        let balance = context.journaled_state.state[&AUTHORITY].info.balance;
        assert!(self.ended.replace((inputs.clone(), outcome.clone(), balance)).is_none());
    }
}

#[rstest]
fn delegated_root_callbacks_keep_completed_inputs_and_reserve_outcome(
    #[values(false, true)] precompile: bool,
    #[values(false, true)] protected: bool,
) {
    let mon = U256::from(10).pow(U256::from(18));
    let target = if precompile { u64_to_address(4) } else { PROXY };
    let mut db = funded_db();
    db.insert_account_info(
        SENDER,
        AccountInfo { balance: mon * U256::from(12), ..Default::default() },
    );
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(target)),
    );
    db.insert_account_info(
        PROXY,
        AccountInfo::default().with_code(Bytecode::new_raw(
            vec![opcode::PUSH1, 1, opcode::PUSH0, opcode::RETURN].into(),
        )),
    );
    let mut context = monad_context_with_db(db);
    context.block.basefee = 0;
    if protected {
        context.chain.parent_senders_and_authorities.insert(SENDER);
    }
    let mut evm = context.build_monad_with_inspector(RootInputRecorder::default());
    let result = evm
        .inspect_one_tx(
            TxEnv::builder()
                .caller(SENDER)
                .to(AUTHORITY)
                .value(mon * U256::from(3))
                .gas_limit(CALL_GAS)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();
    let recorder = &evm.0.inspector;
    assert_eq!(recorder.started.as_ref().unwrap().bytecode_address, AUTHORITY);
    let (inputs, outcome, balance) = recorder.ended.as_ref().unwrap();
    // Immediate results retain the input clone; executed frames expose their resolved input.
    assert_eq!(inputs.bytecode_address, if precompile { AUTHORITY } else { target });
    assert_eq!(inputs.target_address, AUTHORITY);
    assert_eq!(*balance, if protected { U256::ZERO } else { mon * U256::from(3) });
    if protected {
        assert!(matches!(
            result,
            ExecutionResult::Halt { reason: MonadHaltReason::ReserveBalanceViolation, .. }
        ));
        assert_eq!(outcome.result.result, InstructionResult::Revert);
    } else {
        assert!(result.is_success());
        assert!(outcome.result.result.is_ok());
    }
    assert_eq!(outcome.result.output.as_ref(), if precompile { &[][..] } else { &[0][..] });
}

#[rstest]
fn delegation_chain_to_precompile_is_not_followed(
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
) {
    let intermediate = Address::repeat_byte(0x44);
    let mut db = funded_db();
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(intermediate)),
    );
    db.insert_account_info(
        intermediate,
        AccountInfo::default().with_code(Bytecode::new_eip7702(u64_to_address(4))),
    );
    let (result, recorder) = execute(MonadHardfork::MonadTen, db, AUTHORITY, call_opcode, CALL_GAS);
    if call_opcode == 0 {
        assert!(result.is_halt());
    } else {
        assert_output(&result, call_opcode, false, &[]);
    }
    assert!(recorder.ended[0].1.result.result.is_halt());
}

#[derive(Debug)]
struct CustomPrecompiles {
    address: Address,
    addresses: AddressSet,
    removed: bool,
    executions: usize,
    called_addresses: Vec<Address>,
}

impl PrecompileProvider<MonadContext<InMemoryDB>> for CustomPrecompiles {
    type Output = InterpreterResult;

    fn set_spec(&mut self, _spec: MonadHardfork) -> bool {
        false
    }

    fn run(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        inputs: &CallInputs,
    ) -> Result<Option<Self::Output>, String> {
        self.called_addresses.push(inputs.bytecode_address);
        if inputs.bytecode_address != self.address || self.removed {
            return Ok(None);
        }
        self.executions += 1;
        let mut gas = Gas::new_with_regular_gas_and_reservoir(inputs.gas_limit, inputs.reservoir);
        assert!(gas.record_regular_cost(37));
        Ok(Some(InterpreterResult::new(
            InstructionResult::Return,
            Bytes::from_static(b"custom"),
            gas,
        )))
    }

    fn warm_addresses(&self) -> &AddressSet {
        &self.addresses
    }
}

#[rstest]
fn custom_providers_observe_protocol_delegation_rules(
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
    #[values(false, true)] delegated: bool,
    #[values(false, true)] protocol_address: bool,
    #[values(false, true)] removed: bool,
) {
    let address = if protocol_address { u64_to_address(4) } else { Address::repeat_byte(0x44) };
    let mut db = funded_db();
    // Stored code must be ignored at a delegated protocol address, even if its provider was removed.
    db.insert_account_info(
        address,
        AccountInfo::default().with_code(Bytecode::new_raw(
            vec![opcode::PUSH0, opcode::PUSH0, opcode::REVERT].into(),
        )),
    );
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(address)),
    );
    let provider = CustomPrecompiles {
        address,
        addresses: if removed { AddressSet::default() } else { [address].into_iter().collect() },
        removed,
        executions: 0,
        called_addresses: Vec::new(),
    };
    let target = if delegated { AUTHORITY } else { address };
    let (result, recorder, provider) = execute_with_precompiles(
        MonadHardfork::MonadTen,
        db,
        target,
        call_opcode,
        CALL_GAS,
        provider,
    );
    assert_inspector_addresses(&recorder, call_opcode, target);

    let suppressed = protocol_address && delegated;
    let mut provider_addresses = if call_opcode == 0 { Vec::new() } else { vec![PROXY] };
    if !suppressed {
        provider_addresses.push(address);
    }
    assert_eq!(provider.called_addresses, provider_addresses);
    let executed = !suppressed && !removed;
    assert_output(
        &result,
        call_opcode,
        suppressed || executed,
        if executed { b"custom" } else { b"" },
    );
    assert_eq!(provider.executions, usize::from(executed));
    assert_eq!(
        recorder.ended[0].1.result.gas.total_gas_spent(),
        if suppressed {
            0
        } else if removed {
            4
        } else {
            37
        }
    );
}

#[rstest]
fn authorization_installed_by_transaction_suppresses_precompile(
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
) {
    let identity = u64_to_address(4);
    let mut db = funded_db();
    db.insert_account_info(
        PROXY,
        AccountInfo::default().with_code(forward_call(AUTHORITY, call_opcode, CALL_GAS)),
    );
    let mut context = monad_context_with_db(db);
    context.block.basefee = 0;
    let mut evm = context.build_monad_with_inspector(CallRecorder::default());
    let result = evm
        .inspect_tx(
            TxEnv::builder()
                .caller(SENDER)
                .to(if call_opcode == 0 { AUTHORITY } else { PROXY })
                .data(Bytes::from_static(CALLDATA))
                .gas_limit(1_000_000)
                .gas_price(0)
                .gas_priority_fee(Some(0))
                .authorization_list(vec![Either::Right(RecoveredAuthorization::new_unchecked(
                    Authorization { chain_id: U256::ZERO, address: identity, nonce: 0 },
                    RecoveredAuthority::Valid(AUTHORITY),
                ))])
                .build_fill(),
        )
        .unwrap();
    assert_output(&result.result, call_opcode, true, &[]);
    assert_eq!(
        result.state[&AUTHORITY].info.code.as_ref().unwrap().eip7702_address(),
        Some(identity)
    );
    assert_eq!(evm.0.inspector.ended[0].1.result.gas.total_gas_spent(), 0);
}

#[test]
fn delegation_does_not_affect_subsequent_transactions() {
    let identity = u64_to_address(4);
    let mut db = funded_db();
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(identity)),
    );
    let mut context = monad_context_with_db(db);
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    for (target, expected) in [(AUTHORITY, b"".as_slice()), (identity, CALLDATA), (AUTHORITY, b"")]
    {
        let result = evm
            .transact(
                TxEnv::builder()
                    .caller(SENDER)
                    .to(target)
                    .data(Bytes::from_static(CALLDATA))
                    .gas_limit(1_000_000)
                    .gas_price(0)
                    .build_fill(),
            )
            .unwrap();
        assert_output(&result.result, 0, true, expected);
    }
}

#[rstest]
fn top_level_delegated_identity_preserves_value_without_execution_gas(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let identity = u64_to_address(4);
    let mut db = funded_db();
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(identity)),
    );
    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm = context.build_monad_with_inspector(CallRecorder::default());
    let result = evm
        .inspect_tx(
            TxEnv::builder()
                .caller(SENDER)
                .to(AUTHORITY)
                .value(U256::from(1))
                .gas_limit(21_000)
                .gas_price(0)
                .build_fill(),
        )
        .unwrap();
    assert_output(&result.result, 0, true, &[]);
    assert_eq!(result.result.tx_gas_used(), 21_000);
    assert_eq!(result.state[&AUTHORITY].info.balance, U256::from(1));
    assert_eq!(result.state[&identity].info.balance, U256::ZERO);
    let (inputs, outcome) = &evm.0.inspector.ended[0];
    assert_eq!(inputs.gas_limit, 0);
    assert_eq!(outcome.result.gas.remaining(), 0);
    assert_eq!(outcome.result.gas.total_gas_spent(), 0);
}

#[rstest]
fn delegated_native_precompiles_keep_their_restrictions(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(0, opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)]
    call_opcode: u8,
    #[values(STAKING_ADDRESS, RESERVE_BALANCE_ADDRESS)] target: Address,
) {
    let mut db = funded_db();
    db.insert_account_info(
        AUTHORITY,
        AccountInfo::default().with_code(Bytecode::new_eip7702(target)),
    );
    let (result, recorder) = execute(spec, db, AUTHORITY, call_opcode, CALL_GAS);
    let active = target == STAKING_ADDRESS || spec != MonadHardfork::MonadEight;
    assert_output(&result, call_opcode, !active, &[]);
    let (inputs, outcome) = &recorder.ended[0];
    assert_eq!(outcome.result.gas.remaining(), if active { 0 } else { inputs.gas_limit });
}
