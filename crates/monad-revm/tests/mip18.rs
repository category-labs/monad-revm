//! MIP-18 call stack introspection regressions.

use monad_revm::{
    extension::{CALLERN_SELECTOR, CALLSTACKDEPTH_SELECTOR, EXTENSION},
    monad_context_with_db, MonadBuilder, MonadCfgEnv, MonadContext, MonadHaltReason, MonadHardfork,
};
use revm::{
    bytecode::opcode,
    context::{result::ExecutionResult, TxEnv},
    context_interface::result::{HaltReason, OutOfGasError},
    database::InMemoryDB,
    inspector::{InspectEvm, NoOpInspector},
    interpreter::{CallInputs, CallOutcome},
    precompile::u64_to_address,
    primitives::{Address, TxKind, U256},
    state::{AccountInfo, Bytecode},
    ExecuteEvm, Inspector,
};
use rstest::rstest;

const SENDER: Address = Address::repeat_byte(0x11);
const ROOT: Address = Address::repeat_byte(0x22);
const CHILD: Address = Address::repeat_byte(0x33);
const LEAF: Address = Address::repeat_byte(0x44);
const SIBLING: Address = Address::repeat_byte(0x55);
const PROBE_WORDS: usize = 10;

fn database(accounts: impl IntoIterator<Item = (Address, Vec<u8>)>) -> InMemoryDB {
    let mut db = InMemoryDB::default();
    db.insert_account_info(
        SENDER,
        AccountInfo { balance: U256::from(10).pow(U256::from(27)), ..Default::default() },
    );
    for (address, code) in accounts {
        db.insert_account_info(
            address,
            AccountInfo::default().with_code(Bytecode::new_raw(code.into())),
        );
    }
    db
}

fn transaction(gas_limit: u64) -> TxEnv {
    TxEnv::builder().caller(SENDER).to(ROOT).gas_limit(gas_limit).gas_price(0).build_fill()
}

fn execute(
    db: InMemoryDB,
    spec: MonadHardfork,
    tx: TxEnv,
    inspect: bool,
) -> ExecutionResult<MonadHaltReason> {
    let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
    if inspect {
        evm.inspect_one_tx(tx).unwrap()
    } else {
        evm.transact_one(tx).unwrap()
    }
}

fn run(code: Vec<u8>, gas_limit: u64, inspect: bool) -> ExecutionResult<MonadHaltReason> {
    execute(database([(ROOT, code)]), MonadHardfork::MonadNext, transaction(gas_limit), inspect)
}

fn push_u16(code: &mut Vec<u8>, value: usize) {
    code.push(opcode::PUSH2);
    code.extend_from_slice(&u16::try_from(value).unwrap().to_be_bytes());
}

fn store_word(code: &mut Vec<u8>, index: usize) {
    push_u16(code, index * 32);
    code.push(opcode::MSTORE);
}

fn return_memory(code: &mut Vec<u8>, size: usize) {
    push_u16(code, size);
    code.extend_from_slice(&[opcode::PUSH0, opcode::RETURN]);
}

fn return_top(mut code: Vec<u8>) -> Vec<u8> {
    store_word(&mut code, 0);
    return_memory(&mut code, 32);
    code
}

fn probe() -> Vec<u8> {
    let mut code = vec![EXTENSION, CALLSTACKDEPTH_SELECTOR];
    store_word(&mut code, 0);
    for (index, n) in
        [U256::ZERO, U256::from(1), U256::from(2), U256::from(3), U256::from(1) << 64, U256::MAX]
            .into_iter()
            .enumerate()
    {
        code.push(opcode::PUSH32);
        code.extend_from_slice(&n.to_be_bytes::<32>());
        code.extend_from_slice(&[EXTENSION, CALLERN_SELECTOR]);
        store_word(&mut code, index + 1);
    }
    code.extend_from_slice(&[EXTENSION, CALLSTACKDEPTH_SELECTOR, EXTENSION, CALLERN_SELECTOR]);
    store_word(&mut code, 7);
    code.push(opcode::CALLER);
    store_word(&mut code, 8);
    code.push(opcode::ORIGIN);
    store_word(&mut code, 9);
    return_memory(&mut code, PROBE_WORDS * 32);
    code
}

fn assert_probe(result: &ExecutionResult<MonadHaltReason>, senders: &[Address]) {
    assert!(result.is_success(), "{result:?}");
    let actual = result.output().unwrap().chunks(32).map(U256::from_be_slice).collect::<Vec<_>>();
    let mut expected = vec![U256::from(senders.len() - 1)];
    for n in 0..4 {
        expected.push(
            senders.iter().rev().nth(n).map_or(U256::ZERO, |sender| sender.into_word().into()),
        );
    }
    expected.extend([
        U256::ZERO,
        U256::ZERO,
        senders[0].into_word().into(),
        senders.last().unwrap().into_word().into(),
        senders[0].into_word().into(),
    ]);
    assert_eq!(actual, expected);
}

fn assert_halt(result: ExecutionResult<MonadHaltReason>, expected: HaltReason) {
    match result {
        ExecutionResult::Halt { reason, .. } => {
            assert_eq!(reason, MonadHaltReason::Ethereum(expected))
        }
        result => panic!("expected {expected:?}, got {result:?}"),
    }
}

fn call(target: Address, call_opcode: u8) -> Vec<u8> {
    let mut code = vec![opcode::PUSH0; 4];
    if matches!(call_opcode, opcode::CALL | opcode::CALLCODE) {
        code.push(opcode::PUSH0);
    }
    code.push(opcode::PUSH20);
    code.extend_from_slice(target.as_slice());
    code.extend_from_slice(&[opcode::GAS, call_opcode]);
    code
}

fn return_data(code: &mut Vec<u8>) {
    code.extend_from_slice(&[
        opcode::RETURNDATASIZE,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::RETURNDATACOPY,
        opcode::RETURNDATASIZE,
        opcode::PUSH0,
        opcode::RETURN,
    ]);
}

fn forward(target: Address, call_opcode: u8) -> Vec<u8> {
    let mut code = call(target, call_opcode);
    code.push(opcode::POP);
    return_data(&mut code);
    code
}

#[rstest]
fn root_call_and_create(#[values(false, true)] inspect: bool, #[values(false, true)] create: bool) {
    let mut tx = transaction(1_000_000);
    if create {
        tx.kind = TxKind::Create;
        tx.data = probe().into();
    }
    let result = execute(database([(ROOT, probe())]), MonadHardfork::MonadNext, tx, inspect);
    assert_probe(&result, &[SENDER]);
}

#[rstest]
fn nested_callers(
    #[values(false, true)] inspect: bool,
    #[values(opcode::CALL, opcode::CALLCODE, opcode::DELEGATECALL, opcode::STATICCALL)] call_opcode: u8,
) {
    let db = database([
        (ROOT, forward(CHILD, call_opcode)),
        (CHILD, forward(LEAF, opcode::CALL)),
        (LEAF, probe()),
    ]);
    let result = execute(db, MonadHardfork::MonadNext, transaction(1_000_000), inspect);
    let child_sender = if call_opcode == opcode::DELEGATECALL { SENDER } else { ROOT };
    let leaf_sender =
        if matches!(call_opcode, opcode::DELEGATECALL | opcode::CALLCODE) { ROOT } else { CHILD };
    assert_probe(&result, &[SENDER, child_sender, leaf_sender]);
}

#[rstest]
fn delegated_account_keeps_message_depth(#[values(false, true)] inspect: bool) {
    let mut db = database([(ROOT, forward(CHILD, opcode::CALL)), (LEAF, probe())]);
    db.insert_account_info(CHILD, AccountInfo::default().with_code(Bytecode::new_eip7702(LEAF)));
    let result = execute(db, MonadHardfork::MonadNext, transaction(1_000_000), inspect);
    assert_probe(&result, &[SENDER, ROOT]);
}

#[rstest]
fn nested_creation(
    #[values(false, true)] inspect: bool,
    #[values(opcode::CREATE, opcode::CREATE2)] create_opcode: u8,
) {
    let initcode = probe();
    let mut code = Vec::new();
    push_u16(&mut code, initcode.len());
    push_u16(&mut code, 0);
    code.extend_from_slice(&[opcode::PUSH0, opcode::CODECOPY]);
    if create_opcode == opcode::CREATE2 {
        code.push(opcode::PUSH0);
    }
    push_u16(&mut code, initcode.len());
    code.extend_from_slice(&[
        opcode::PUSH0,
        opcode::PUSH0,
        create_opcode,
        opcode::PUSH0,
        opcode::MSTORE,
    ]);
    // Read back the probe output deposited as the new contract's runtime code.
    push_u16(&mut code, PROBE_WORDS * 32);
    code.extend_from_slice(&[
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::MLOAD,
        opcode::EXTCODECOPY,
    ]);
    return_memory(&mut code, PROBE_WORDS * 32);
    let offset = u16::try_from(code.len()).unwrap();
    code[4..6].copy_from_slice(&offset.to_be_bytes());
    code.extend(initcode);
    assert_probe(&run(code, 1_000_000, inspect), &[SENDER, ROOT]);
}

#[rstest]
fn returned_frames_do_not_leak_callers(
    #[values(false, true)] inspect: bool,
    #[values(opcode::STOP, opcode::REVERT, opcode::INVALID)] exit_opcode: u8,
) {
    let mut child = call(LEAF, opcode::CALL);
    child.extend_from_slice(&[opcode::POP, opcode::PUSH0, opcode::PUSH0, exit_opcode]);
    let mut root = call(CHILD, opcode::DELEGATECALL);
    root.push(opcode::POP);
    root.extend(forward(SIBLING, opcode::CALL));
    let db = database([
        (ROOT, root),
        (CHILD, child),
        (SIBLING, forward(LEAF, opcode::CALL)),
        (LEAF, probe()),
    ]);
    assert_probe(
        &execute(db, MonadHardfork::MonadNext, transaction(10_000_000), inspect),
        &[SENDER, ROOT, SIBLING],
    );
}

#[rstest]
fn parent_depth_survives_immediate_and_completed_calls(
    #[values(false, true)] inspect: bool,
    #[values(0, 1, 2, 3, 4)] case: u8,
) {
    let (target, child) = match case {
        0 => (u64_to_address(4), Vec::new()),
        1 => (CHILD, Vec::new()),
        2 => (CHILD, probe()),
        3 => (CHILD, vec![opcode::PUSH0, opcode::PUSH0, opcode::REVERT]),
        _ => (CHILD, vec![opcode::INVALID]),
    };
    let mut root = call(target, opcode::CALL);
    root.push(opcode::POP);
    root.extend(probe());
    let result = execute(
        database([(ROOT, root), (CHILD, child)]),
        MonadHardfork::MonadNext,
        transaction(10_000_000),
        inspect,
    );
    assert_probe(&result, &[SENDER]);
}

#[rstest]
fn trailing_prefix_after_assigned_selector(#[values(false, true)] inspect: bool) {
    assert_halt(
        run(vec![EXTENSION, CALLSTACKDEPTH_SELECTOR, EXTENSION], 100_000, inspect),
        HaltReason::InvalidFEOpcode,
    );
}

#[rstest]
fn extension_gas(#[values(false, true)] inspect: bool) {
    for (code, gas) in [
        (vec![EXTENSION, CALLSTACKDEPTH_SELECTOR], 2),
        (vec![opcode::PUSH0, EXTENSION, CALLERN_SELECTOR], 4),
    ] {
        let result = run(code.clone(), 21_000 + gas, inspect);
        assert!(result.is_success(), "{result:?}");
        assert_eq!(result.tx_gas_used(), 21_000 + gas);
        assert_halt(
            run(code, 21_000 + gas - 1, inspect),
            HaltReason::OutOfGas(OutOfGasError::Basic),
        );
    }
    let mut code = vec![opcode::PUSH32];
    code.extend_from_slice(&[0xff; 32]);
    code.extend_from_slice(&[EXTENSION, CALLERN_SELECTOR]);
    let result = run(code, 21_005, inspect);
    assert!(result.is_success(), "{result:?}");
    assert_eq!(result.tx_gas_used(), 21_005);
}

#[rstest]
fn extension_stack_limits(#[values(false, true)] inspect: bool) {
    assert_halt(
        run(vec![EXTENSION, CALLERN_SELECTOR], 100_000, inspect),
        HaltReason::StackUnderflow,
    );
    let mut code = vec![opcode::PUSH0; 1024];
    code.extend_from_slice(&[EXTENSION, CALLSTACKDEPTH_SELECTOR]);
    assert_halt(run(code, 100_000, inspect), HaltReason::StackOverflow);
    let mut code = vec![opcode::PUSH0; 1024];
    code.extend_from_slice(&[EXTENSION, CALLERN_SELECTOR]);
    assert!(run(code, 100_000, inspect).is_success());
    let mut code = vec![opcode::PUSH0; 1023];
    code.extend_from_slice(&[EXTENSION, CALLSTACKDEPTH_SELECTOR]);
    assert!(run(code, 100_000, inspect).is_success());
}

#[rstest]
fn assigned_selectors_advance_program_counter(#[values(false, true)] inspect: bool) {
    for (code, expected_pc) in [
        (vec![EXTENSION, CALLSTACKDEPTH_SELECTOR, opcode::POP, opcode::PC], 3),
        (vec![opcode::PUSH0, EXTENSION, CALLERN_SELECTOR, opcode::POP, opcode::PC], 4),
    ] {
        let result = run(return_top(code), 100_000, inspect);
        assert!(result.is_success(), "{result:?}");
        assert_eq!(U256::from_be_slice(result.output().unwrap()), U256::from(expected_pc));
    }
}

#[derive(Debug)]
struct SwitchSpec(MonadHardfork);

impl Inspector<MonadContext<InMemoryDB>> for SwitchSpec {
    fn call(
        &mut self,
        ctx: &mut MonadContext<InMemoryDB>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        if inputs.target_address == CHILD {
            let mut cfg = ctx.cfg.clone().into_inner();
            cfg.spec = self.0;
            ctx.cfg = MonadCfgEnv::from(cfg);
        }
        None
    }
}

#[test]
fn child_can_enable_extensions_with_legacy_ancestors() {
    let db = database([(ROOT, forward(CHILD, opcode::CALL)), (CHILD, probe())]);
    let ctx =
        monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(MonadHardfork::MonadTen));
    let mut evm = ctx.build_monad_with_inspector(SwitchSpec(MonadHardfork::MonadNext));
    assert_probe(&evm.inspect_one_tx(transaction(1_000_000)).unwrap(), &[SENDER, ROOT]);
}

#[test]
fn parent_extensions_are_restored_after_legacy_child() {
    let mut root = call(CHILD, opcode::CALL);
    root.push(opcode::POP);
    root.extend(probe());
    let db = database([(ROOT, root), (CHILD, probe())]);
    let ctx =
        monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(MonadHardfork::MonadNext));
    let mut evm = ctx.build_monad_with_inspector(SwitchSpec(MonadHardfork::MonadTen));
    assert_probe(&evm.inspect_one_tx(transaction(1_000_000)).unwrap(), &[SENDER]);
}

#[rstest]
fn reused_evm_overwrites_callers(#[values(false, true)] inspect: bool) {
    let db = database([(ROOT, forward(CHILD, opcode::CALL)), (CHILD, probe())]);
    let ctx =
        monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(MonadHardfork::MonadNext));
    let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
    for (nonce, to, senders) in
        [(0, ROOT, vec![SENDER, ROOT]), (1, CHILD, vec![SENDER]), (0, CHILD, vec![SIBLING])]
    {
        let mut tx = transaction(1_000_000);
        tx.nonce = nonce;
        tx.caller = senders[0];
        tx.kind = TxKind::Call(to);
        let result = if inspect { evm.inspect_one_tx(tx) } else { evm.transact_one(tx) }.unwrap();
        assert_probe(&result, &senders);
    }
}
