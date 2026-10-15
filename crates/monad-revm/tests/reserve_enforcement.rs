//! Reserve-balance enforcement at protocol transaction boundaries.

use monad_revm::{
    monad_context_with_db, reserve_balance::tracker::ReserveBalanceInit, MonadBuilder, MonadCfgEnv,
    MonadContext, MonadEvm, MonadHaltReason, MonadHardfork, MonadJournalTr,
};
use revm::{
    bytecode::opcode,
    context::{ContextError, TxEnv},
    context_interface::{
        either::Either,
        journaled_state::account::JournaledAccountTr,
        result::{EVMError, ExecutionResult, HaltReason, Output, ResultAndState},
        transaction::{Authorization, RecoveredAuthority, RecoveredAuthorization},
        JournalTr,
    },
    database::InMemoryDB,
    handler::FrameResult,
    inspector::NoOpInspector,
    interpreter::{
        CallInputs, CallOutcome, CreateInputs, CreateOutcome, FrameInput, Gas, InstructionResult,
        Interpreter, InterpreterResult,
    },
    primitives::{Address, Bytes, TxKind, U256},
    state::{AccountInfo, Bytecode, EvmState},
    ExecuteCommitEvm, ExecuteEvm, InspectCommitEvm, InspectEvm, Inspector,
};
use rstest::rstest;

const SENDER: Address = Address::repeat_byte(0x11);
const TARGET: Address = Address::repeat_byte(0x22);
const RECIPIENT: Address = Address::repeat_byte(0x33);
const BENEFICIARY: Address = Address::repeat_byte(0x44);
const AUTHORITY: Address = Address::repeat_byte(0x55);
const DELEGATE: Address = Address::repeat_byte(0x66);
const GAS_LIMIT: u64 = 1_000_000;

type TestEvm = MonadEvm<MonadContext<InMemoryDB>, NoOpInspector>;

fn mon(amount: u64) -> U256 {
    U256::from(amount) * U256::from(10).pow(U256::from(18))
}

fn evm(spec: MonadHardfork, protected: bool) -> TestEvm {
    let mut db = InMemoryDB::default();
    db.insert_account_info(SENDER, AccountInfo { balance: mon(12), ..Default::default() });
    let mut ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    ctx.block.basefee = 0;
    ctx.block.beneficiary = BENEFICIARY;
    if protected {
        ctx.chain.parent_senders_and_authorities.insert(SENDER);
    }
    ctx.build_monad_with_inspector(NoOpInspector)
}

fn execute(evm: &mut TestEvm, tx: TxEnv, inspect: bool) -> ResultAndState<MonadHaltReason> {
    if inspect {
        evm.inspect_tx(tx).unwrap()
    } else {
        evm.transact(tx).unwrap()
    }
}

fn tx(create: bool, code: Vec<u8>, value: U256, evm: &mut TestEvm) -> TxEnv {
    if !create {
        evm.0.ctx.journaled_state.database.insert_account_info(
            TARGET,
            AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(code.clone()))),
        );
    }
    TxEnv::builder()
        .caller(SENDER)
        .kind(if create { TxKind::Create } else { TxKind::Call(TARGET) })
        .data(if create { Bytes::from(code) } else { Bytes::new() })
        .value(value)
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill()
}

fn assert_violation(out: &ResultAndState<MonadHaltReason>, sender_nonce: u64, gas_limit: u64) {
    assert!(
        matches!(
            out.result,
            ExecutionResult::Halt { reason: MonadHaltReason::ReserveBalanceViolation, .. }
        ),
        "unexpected result: {:?}",
        out.result
    );
    assert_eq!(out.state[&SENDER].info.balance, mon(12) - U256::from(gas_limit));
    assert_eq!(out.state[&SENDER].info.nonce, sender_nonce);
    assert_eq!(out.state[&BENEFICIARY].info.balance, U256::from(gas_limit));
    assert!(out.result.logs().is_empty());
    assert_eq!(out.result.gas().inner_refunded(), 0);
}

fn assert_unchanged_empty_account(out: &ResultAndState<MonadHaltReason>, address: Address) {
    if let Some(account) = out.state.get(&address) {
        assert_eq!(account.info.balance, U256::ZERO);
        assert_eq!(account.info.nonce, 0);
        assert!(account.info.code.as_ref().is_none_or(Bytecode::is_empty));
        assert!(account.storage.values().all(|slot| slot.present_value() == U256::ZERO));
    }
}

fn push_value(code: &mut Vec<u8>, value: U256) {
    code.push(opcode::PUSH32);
    code.extend_from_slice(&value.to_be_bytes::<32>());
}

fn call(code: &mut Vec<u8>, recipient: Address, value: U256) {
    code.extend_from_slice(&[opcode::PUSH0; 4]);
    push_value(code, value);
    code.push(opcode::PUSH20);
    code.extend_from_slice(recipient.as_slice());
    code.extend_from_slice(&[opcode::GAS, opcode::CALL, opcode::POP]);
}

fn storage_and_log() -> Vec<u8> {
    vec![
        opcode::PUSH1,
        42,
        opcode::PUSH0,
        opcode::SSTORE,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::LOG0,
    ]
}

fn return_code(code: &mut Vec<u8>) {
    // Return one zero byte, which deploys STOP for a creation transaction.
    code.extend_from_slice(&[opcode::PUSH1, 1, opcode::PUSH0, opcode::RETURN]);
}

#[rstest]
fn reserve_violation_reverts_execution(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] create: bool,
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(spec, true);
    let recipient = if create { SENDER.create(0) } else { TARGET };
    let mut tx = tx(create, Vec::new(), mon(3), &mut evm);
    tx.gas_limit = 100_000;
    let out = execute(&mut evm, tx, inspect);
    assert_violation(&out, 1, 100_000);
    assert_unchanged_empty_account(&out, recipient);
    assert_eq!(out.result.gas().total_gas_spent(), if create { 100_000 } else { 21_000 });
}

#[rstest]
fn reserve_violation_rolls_back_storage_logs_and_nested_transfers(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] create: bool,
    #[values(false, true)] inspect: bool,
    #[values(opcode::RETURN, opcode::REVERT, opcode::INVALID)] terminal: u8,
) {
    let mut code = storage_and_log();
    call(&mut code, RECIPIENT, mon(1));
    code.extend_from_slice(&[opcode::PUSH1, 1, opcode::PUSH0, terminal]);
    let mut protected = evm(spec, true);
    let tx = tx(create, code.clone(), mon(3), &mut protected);
    let mut unprotected = evm(spec, false);
    let unrestricted_tx = self::tx(create, code, mon(3), &mut unprotected);
    let reference = execute(&mut unprotected, unrestricted_tx, inspect);
    let out = execute(&mut protected, tx, inspect);
    assert_violation(&out, 1, GAS_LIMIT);
    assert_unchanged_empty_account(&out, RECIPIENT);
    if create {
        assert_unchanged_empty_account(&out, SENDER.create(0));
        assert_eq!(out.result.gas().total_gas_spent(), GAS_LIMIT);
    } else {
        let target = &out.state[&TARGET];
        assert_eq!(target.info.balance, U256::ZERO);
        assert_eq!(target.storage[&U256::ZERO].present_value(), U256::ZERO);
        assert_eq!(out.result.gas().total_gas_spent(), reference.result.gas().total_gas_spent());
    }
}

#[rstest]
fn temporary_reserve_violation_can_be_repaid_before_completion(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] create: bool,
    #[values(false, true)] inspect: bool,
) {
    let mut code = storage_and_log();
    call(&mut code, SENDER, mon(3));
    return_code(&mut code);
    let mut evm = evm(spec, true);
    let tx = tx(create, code, mon(3), &mut evm);
    let out = execute(&mut evm, tx, inspect);
    assert!(out.result.is_success(), "unexpected result: {:?}", out.result);
    assert_eq!(out.state[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
    assert_eq!(out.state[&SENDER].info.nonce, 1);
    assert_eq!(out.result.logs().len(), 1);
    let target = if create { SENDER.create(0) } else { TARGET };
    assert_eq!(out.state[&target].storage[&U256::ZERO].present_value(), U256::from(42));
}

#[rstest]
fn nested_revert_restores_reserve_before_outer_completion(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
    #[values(false, true)] revert_child: bool,
) {
    let mut evm = evm(spec, true);
    let mut child = Vec::new();
    call(&mut child, RECIPIENT, mon(3));
    child.extend_from_slice(&[
        opcode::PUSH0,
        opcode::PUSH0,
        if revert_child { opcode::REVERT } else { opcode::RETURN },
    ]);
    evm.0.ctx.journaled_state.database.insert_account_info(
        AUTHORITY,
        AccountInfo { balance: mon(12), ..Default::default() }
            .with_code(Bytecode::new_eip7702(DELEGATE)),
    );
    evm.0.ctx.journaled_state.database.insert_account_info(
        DELEGATE,
        AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(child))),
    );
    let mut parent = storage_and_log();
    call(&mut parent, AUTHORITY, U256::ZERO);
    return_code(&mut parent);
    let tx = tx(false, parent, U256::ZERO, &mut evm);
    let out = execute(&mut evm, tx, inspect);
    assert_eq!(out.state[&AUTHORITY].info.balance, mon(12));
    assert_unchanged_empty_account(&out, RECIPIENT);
    if revert_child {
        assert!(out.result.is_success(), "unexpected result: {:?}", out.result);
        assert_eq!(out.result.logs().len(), 1);
        assert_eq!(out.state[&TARGET].storage[&U256::ZERO].present_value(), U256::from(42));
    } else {
        assert_violation(&out, 1, GAS_LIMIT);
    }
}

/// Builds a recovered authorization using the authority's account nonce.
fn authorization(
    authority: Address,
    account_nonce: u64,
) -> Either<revm::context_interface::transaction::SignedAuthorization, RecoveredAuthorization> {
    Either::Right(RecoveredAuthorization::new_unchecked(
        Authorization { chain_id: U256::from(1), address: DELEGATE, nonce: account_nonce },
        RecoveredAuthority::Valid(authority),
    ))
}

#[rstest]
fn authorizations_survive_reserve_rejection(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(spec, false);
    let mut tx = tx(false, storage_and_log(), mon(3), &mut evm);
    tx.tx_type = 4;
    tx.authorization_list = vec![authorization(SENDER, 1), authorization(AUTHORITY, 0)];
    let out = execute(&mut evm, tx, inspect);
    assert_violation(&out, 2, GAS_LIMIT);
    assert_eq!(out.state[&AUTHORITY].info.nonce, 1);
    for address in [SENDER, AUTHORITY] {
        assert_eq!(
            out.state[&address].info.code.as_ref().and_then(Bytecode::eip7702_address),
            Some(DELEGATE)
        );
    }
    assert_eq!(out.state[&TARGET].info.balance, U256::ZERO);
    assert_eq!(out.state[&TARGET].storage[&U256::ZERO].present_value(), U256::ZERO);
}

#[rstest]
fn immediate_precompile_result_is_checked(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(spec, true);
    let identity = Address::with_last_byte(4);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(identity)
        .value(mon(3))
        .data(Bytes::from_static(b"hello"))
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();
    let out = execute(&mut evm, tx, inspect);
    assert_violation(&out, 1, GAS_LIMIT);
    assert_unchanged_empty_account(&out, identity);
}

#[rstest]
fn preserved_synthetic_calls_defer_reserve_enforcement(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(spec, true);
    // The first synthetic call leaves the transfer in place; a later call repays it.
    let mut code = vec![
        opcode::CALLVALUE,
        opcode::ISZERO,
        opcode::PUSH1,
        6,
        opcode::JUMPI,
        opcode::STOP,
        opcode::JUMPDEST,
    ];
    call(&mut code, SENDER, mon(3));
    evm.0.ctx.journaled_state.database.insert_account_info(
        RECIPIENT,
        AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(code))),
    );
    let sender = evm.0.ctx.journaled_state.load_account(SENDER).unwrap().data.clone();
    evm.0.ctx.journaled_state.reserve_balance_mut().init(ReserveBalanceInit {
        chain: &evm.0.ctx.chain,
        spec,
        sender: SENDER,
        effective_gas_price: 0,
        gas_limit: GAS_LIMIT,
        sender_is_delegated: false,
        sender_account: Some(&sender),
    });
    evm.0.ctx.journaled_state.set_preserve_reserve_balance_tracker(true);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(RECIPIENT)
        .value(mon(3))
        .gas_limit(GAS_LIMIT)
        .gas_price(0)
        .build_fill();
    let first =
        if inspect { evm.inspect_one_tx(tx).unwrap() } else { evm.transact_one(tx).unwrap() };
    assert!(first.is_success());
    assert!(evm.0.ctx.journaled_state.reserve_balance().has_violation());
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(RECIPIENT)
        .nonce(1)
        .gas_limit(GAS_LIMIT)
        .gas_price(0)
        .build_fill();
    let second =
        if inspect { evm.inspect_one_tx(tx).unwrap() } else { evm.transact_one(tx).unwrap() };
    assert!(second.is_success());
    assert!(!evm.0.ctx.journaled_state.reserve_balance().has_violation());
    assert_eq!(evm.finalize()[&SENDER].info.balance, mon(12));
}

#[rstest]
fn execution_checkpoint_preserves_top_level_precompile_error_context(
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(Address::with_last_byte(9))
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();
    let out = execute(&mut evm, tx, inspect);
    assert!(
        matches!(
            out.result,
            ExecutionResult::Halt {
                reason: MonadHaltReason::Ethereum(HaltReason::PrecompileErrorWithContext(_)),
                ..
            }
        ),
        "unexpected result: {:?}",
        out.result
    );
}

#[rstest]
fn committed_reserve_rejection_keeps_fees_and_allows_the_next_transaction(
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let first = tx(false, storage_and_log(), mon(3), &mut evm);
    let rejected = if inspect {
        evm.inspect_tx_commit(first).unwrap()
    } else {
        evm.transact_commit(first).unwrap()
    };
    assert!(matches!(
        rejected,
        ExecutionResult::Halt { reason: MonadHaltReason::ReserveBalanceViolation, .. }
    ));
    assert!(rejected.logs().is_empty());
    let db = &evm.0.ctx.journaled_state.database;
    assert_eq!(db.cache.accounts[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
    assert_eq!(db.cache.accounts[&SENDER].info.nonce, 1);
    assert_eq!(db.cache.accounts[&BENEFICIARY].info.balance, U256::from(GAS_LIMIT));
    assert_eq!(db.cache.accounts[&TARGET].info.balance, U256::ZERO);
    assert_eq!(
        db.cache.accounts[&TARGET].storage.get(&U256::ZERO).copied().unwrap_or_default(),
        U256::ZERO
    );
    assert_eq!(evm.0.ctx.journaled_state.depth(), 0);
    assert!(!evm.0.ctx.journaled_state.reserve_balance().tracking_enabled());

    let next = TxEnv::builder()
        .caller(SENDER)
        .to(RECIPIENT)
        .nonce(1)
        .value(mon(1))
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();
    let accepted = if inspect {
        evm.inspect_tx_commit(next).unwrap()
    } else {
        evm.transact_commit(next).unwrap()
    };
    assert!(accepted.is_success());
    let db = &evm.0.ctx.journaled_state.database;
    assert_eq!(db.cache.accounts[&SENDER].info.balance, mon(11) - U256::from(2 * GAS_LIMIT));
    assert_eq!(db.cache.accounts[&SENDER].info.nonce, 2);
    assert_eq!(db.cache.accounts[&BENEFICIARY].info.balance, U256::from(2 * GAS_LIMIT));
    assert_eq!(db.cache.accounts[&RECIPIENT].info.balance, mon(1));
}

#[rstest]
fn creation_reserve_check_observes_deployed_code(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] deploy_code: bool,
) {
    let mut evm = evm(spec, true);
    let created = SENDER.create(0);
    evm.0
        .ctx
        .journaled_state
        .database
        .insert_account_info(created, AccountInfo { balance: mon(12), ..Default::default() });
    let mut code = Vec::new();
    call(&mut code, RECIPIENT, mon(3));
    if deploy_code {
        return_code(&mut code);
    }
    let tx = tx(true, code, U256::ZERO, &mut evm);
    let out = execute(&mut evm, tx, false);
    if deploy_code {
        assert!(out.result.is_success());
        assert_eq!(out.state[&created].info.balance, mon(9));
        assert_eq!(out.state[&RECIPIENT].info.balance, mon(3));
        assert_eq!(
            out.state[&created].info.code.as_ref().unwrap().original_bytes(),
            Bytes::from_static(&[opcode::STOP])
        );
    } else {
        assert_violation(&out, 1, GAS_LIMIT);
        assert_eq!(out.state[&created].info.balance, mon(12));
        assert_eq!(out.state[&created].info.nonce, 0);
        assert_unchanged_empty_account(&out, RECIPIENT);
    }
}

#[test]
fn frame_initialization_failure_precedes_reserve_enforcement() {
    for create in [false, true] {
        let mut evm = evm(MonadHardfork::MonadTen, true);
        evm.0.ctx.chain.max_reserve_balance = U256::from(1);
        evm.0
            .ctx
            .journaled_state
            .database
            .insert_account_info(SENDER.create(0), AccountInfo { nonce: 1, ..Default::default() });
        let tx = tx(create, Vec::new(), if create { U256::ZERO } else { mon(13) }, &mut evm);
        let out = execute(&mut evm, tx, false);
        let expected = if create { HaltReason::CreateCollision } else { HaltReason::OutOfFunds };
        assert!(
            matches!(out.result, ExecutionResult::Halt { reason: MonadHaltReason::Ethereum(reason), .. } if reason == expected)
        );
    }
}

#[rstest]
fn fees_above_reserve_reject_nonemptying_transactions(
    #[values(MonadHardfork::MonadEight, MonadHardfork::MonadNine)] spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
    #[values(false, true)] protected: bool,
    #[values(10_000_000_000_000, 10_200_000_000_000)] gas_price: u128,
) {
    let mut evm = evm(spec, protected);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(RECIPIENT)
        .gas_limit(GAS_LIMIT)
        .gas_price(gas_price)
        .build_fill();
    let out = execute(&mut evm, tx, inspect);
    let fee = U256::from(GAS_LIMIT) * U256::from(gas_price);

    // The canonical client rejects fees above reserve even when the sender remains solvent.
    if protected && fee > mon(10) {
        assert!(matches!(
            out.result,
            ExecutionResult::Halt { reason: MonadHaltReason::ReserveBalanceViolation, .. }
        ));
    } else {
        assert!(out.result.is_success(), "unexpected result: {:?}", out.result);
    }
    assert_eq!(out.state[&SENDER].info.balance, mon(12) - fee);
    assert_eq!(out.state[&SENDER].info.nonce, 1);
    assert_eq!(out.state[&BENEFICIARY].info.balance, fee);
    assert!(out.result.logs().is_empty());
    assert_eq!(out.result.gas().inner_refunded(), 0);
}

#[rstest]
fn creation_nonce_overflow_precedes_reserve_enforcement(#[values(false, true)] inspect: bool) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    evm.0.ctx.cfg.disable_nonce_check = true;
    evm.0.ctx.chain.max_reserve_balance = U256::from(1);
    evm.0.ctx.journaled_state.database.insert_account_info(
        SENDER,
        AccountInfo { balance: mon(12), nonce: u64::MAX, ..Default::default() },
    );
    let tx = tx(true, storage_and_log(), U256::ZERO, &mut evm);
    let out = execute(&mut evm, tx, inspect);
    assert!(matches!(out.result, ExecutionResult::Success { output: Output::Create(_, None), .. }));
    assert_eq!(out.state[&SENDER].info.nonce, u64::MAX);
    assert_eq!(out.state[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
    assert_eq!(out.state[&BENEFICIARY].info.balance, U256::from(GAS_LIMIT));
    assert!(out.result.logs().is_empty());
    assert_unchanged_empty_account(&out, SENDER.create(u64::MAX));
}

#[derive(Default)]
struct FrameOutcomeInspector {
    events: Vec<(&'static str, usize)>,
    roots: Vec<RootOutcome>,
}

struct RootOutcome {
    result: InstructionResult,
    logs: usize,
    gas_remaining: u64,
    state: EvmState,
}

impl FrameOutcomeInspector {
    fn record_end(
        &mut self,
        event: &'static str,
        context: &MonadContext<InMemoryDB>,
        result: &InterpreterResult,
    ) {
        let depth = context.journaled_state.depth();
        self.events.push((event, depth));
        if depth == 0 {
            self.roots.push(RootOutcome {
                result: result.result,
                logs: context.journaled_state.logs().len(),
                gas_remaining: result.gas.remaining(),
                state: context.journaled_state.state.clone(),
            });
        }
    }
}

impl Inspector<MonadContext<InMemoryDB>> for FrameOutcomeInspector {
    fn frame_start(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _input: &mut FrameInput,
    ) -> Option<FrameResult> {
        self.events.push(("frame_start", context.journaled_state.depth()));
        None
    }

    fn call(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        self.events.push(("call", context.journaled_state.depth()));
        None
    }

    fn create(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &mut CreateInputs,
    ) -> Option<CreateOutcome> {
        self.events.push(("create", context.journaled_state.depth()));
        None
    }

    fn initialize_interp(
        &mut self,
        _interp: &mut Interpreter,
        context: &mut MonadContext<InMemoryDB>,
    ) {
        self.events.push(("initialize", context.journaled_state.depth()));
    }

    fn log(&mut self, context: &mut MonadContext<InMemoryDB>, _log: revm::primitives::Log) {
        self.events.push(("log", context.journaled_state.depth()));
    }

    fn call_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        self.record_end("call_end", context, &outcome.result);
    }

    fn create_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        self.record_end("create_end", context, &outcome.result);
    }

    fn frame_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _input: &FrameInput,
        result: &mut FrameResult,
    ) {
        self.record_end("frame_end", context, result.interpreter_result());
    }
}

#[rstest]
fn root_end_callbacks_observe_reserve_rejection_after_execution_rollback(
    #[values(false, true)] create: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let child = vec![opcode::PUSH0, opcode::PUSH0, opcode::LOG0];
    evm.0.ctx.journaled_state.database.insert_account_info(
        RECIPIENT,
        AccountInfo::default().with_code(Bytecode::new_raw(Bytes::from(child))),
    );
    let mut code = storage_and_log();
    call(&mut code, RECIPIENT, mon(1));
    return_code(&mut code);
    let tx = tx(create, code, mon(3), &mut evm);
    let mut evm = evm.with_inspector(FrameOutcomeInspector::default());
    let out = evm.inspect_tx(tx).unwrap();
    let inspector = &evm.0.inspector;
    assert_eq!(
        inspector.events,
        vec![
            ("frame_start", 0),
            (if create { "create" } else { "call" }, 0),
            ("initialize", 1),
            ("log", 1),
            ("frame_start", 1),
            ("call", 1),
            ("initialize", 2),
            ("log", 2),
            ("call_end", 1),
            ("frame_end", 1),
            (if create { "create_end" } else { "call_end" }, 0),
            ("frame_end", 0),
        ]
    );
    assert_eq!(inspector.roots.len(), 2);
    for root in &inspector.roots {
        assert_eq!(root.result, InstructionResult::Revert);
        assert_eq!(root.logs, 0);
        assert_eq!(root.gas_remaining, GAS_LIMIT - out.result.gas().total_gas_spent());
        assert_eq!(root.state[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
        assert_eq!(root.state[&SENDER].info.nonce, 1);
        assert_eq!(root.state[&RECIPIENT].info.balance, U256::ZERO);
        let target = if create { SENDER.create(0) } else { TARGET };
        let target = &root.state[&target];
        assert_eq!(target.info.balance, U256::ZERO);
        assert!(target.storage.values().all(|slot| slot.present_value() == U256::ZERO));
        if create {
            assert_eq!(target.info.nonce, 0);
            assert!(target.info.code.as_ref().is_none_or(Bytecode::is_empty));
        }
    }
    assert_violation(&out, 1, GAS_LIMIT);
}

#[test]
fn immediate_call_callbacks_observe_reserve_rejection() {
    for recipient in [TARGET, Address::with_last_byte(4)] {
        let mut evm =
            evm(MonadHardfork::MonadTen, true).with_inspector(FrameOutcomeInspector::default());
        let tx = TxEnv::builder()
            .caller(SENDER)
            .to(recipient)
            .value(mon(3))
            .gas_limit(GAS_LIMIT)
            .gas_price(1)
            .build_fill();
        let out = evm.inspect_tx(tx).unwrap();
        let inspector = &evm.0.inspector;
        assert_eq!(
            inspector.events,
            [("frame_start", 0), ("call", 0), ("call_end", 0), ("frame_end", 0)]
        );
        for root in &inspector.roots {
            assert_eq!(root.result, InstructionResult::Revert);
            assert_eq!(root.state[&recipient].info.balance, U256::ZERO);
        }
        assert_violation(&out, 1, GAS_LIMIT);
    }
}

#[test]
fn root_end_callbacks_retain_authorizations() {
    let mut evm = evm(MonadHardfork::MonadTen, false);
    let mut tx = tx(false, storage_and_log(), mon(3), &mut evm);
    tx.tx_type = 4;
    tx.authorization_list = vec![authorization(SENDER, 1), authorization(AUTHORITY, 0)];
    let mut evm = evm.with_inspector(FrameOutcomeInspector::default());
    let out = evm.inspect_tx(tx).unwrap();
    for root in &evm.0.inspector.roots {
        assert_eq!(root.result, InstructionResult::Revert);
        assert_eq!(root.state[&SENDER].info.nonce, 2);
        assert_eq!(root.state[&AUTHORITY].info.nonce, 1);
        for address in [SENDER, AUTHORITY] {
            assert_eq!(
                root.state[&address].info.code.as_ref().and_then(Bytecode::eip7702_address),
                Some(DELEGATE)
            );
        }
    }
    assert_violation(&out, 2, GAS_LIMIT);
}

struct OutcomeEditor {
    generic: bool,
    success: bool,
}

impl OutcomeEditor {
    fn edit(&self, result: &mut InterpreterResult) {
        assert_eq!(result.result, InstructionResult::Revert);
        if self.success {
            result.result = InstructionResult::Return;
        }
        result.output = Bytes::from_static(b"inspector output");
    }
}

impl Inspector<MonadContext<InMemoryDB>> for OutcomeEditor {
    fn call_end(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        _inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        if !self.generic {
            self.edit(&mut outcome.result);
        }
    }

    fn create_end(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        _inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        if !self.generic {
            self.edit(&mut outcome.result);
        }
    }

    fn frame_end(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        _input: &FrameInput,
        result: &mut FrameResult,
    ) {
        if self.generic {
            self.edit(result.interpreter_result_mut());
        }
    }
}

#[rstest]
fn inspector_can_override_adjudicated_root_outcomes(
    #[values(false, true)] create: bool,
    #[values(false, true)] generic: bool,
    #[values(false, true)] success: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let tx = tx(create, storage_and_log(), mon(3), &mut evm);
    let mut evm =
        evm.with_inspector((OutcomeEditor { generic, success }, FrameOutcomeInspector::default()));
    let out = evm.inspect_tx(tx).unwrap();
    assert_eq!(out.result.is_success(), success);
    assert_eq!(matches!(out.result, ExecutionResult::Revert { .. }), !success);
    assert_eq!(out.result.output(), Some(&Bytes::from_static(b"inspector output")));
    assert_eq!(out.state[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
    assert_eq!(out.state[&SENDER].info.nonce, 1);
    assert!(out.result.logs().is_empty());
    let target = if create { SENDER.create(0) } else { TARGET };
    assert_eq!(out.state[&target].info.balance, U256::ZERO);
    assert!(out.state[&target].storage.values().all(|slot| slot.present_value() == U256::ZERO));
    assert_eq!(evm.0.inspector.1.roots.len(), 2);
    assert_eq!(
        evm.0.inspector.1.roots.last().unwrap().result,
        if success { InstructionResult::Return } else { InstructionResult::Revert }
    );
}

struct ShortCircuitInspector {
    generic: bool,
    advance_nonce: bool,
}

impl ShortCircuitInspector {
    fn result(
        &self,
        context: &mut MonadContext<InMemoryDB>,
        gas_limit: u64,
        create: bool,
    ) -> FrameResult {
        if create && self.advance_nonce {
            context.journaled_state.load_account_mut(SENDER).unwrap().bump_nonce();
        }
        let recipient = if create { SENDER.create(0) } else { TARGET };
        assert!(context.journaled_state.transfer(SENDER, recipient, mon(3)).unwrap().is_none());
        let result = InterpreterResult {
            result: InstructionResult::Return,
            output: Bytes::new(),
            gas: Gas::new(gas_limit),
        };
        if create {
            FrameResult::Create(CreateOutcome::new(result, None))
        } else {
            FrameResult::Call(CallOutcome::new(result, 0..0))
        }
    }
}

impl Inspector<MonadContext<InMemoryDB>> for ShortCircuitInspector {
    fn frame_start(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        input: &mut FrameInput,
    ) -> Option<FrameResult> {
        if !self.generic {
            return None;
        }
        Some(match input {
            FrameInput::Call(inputs) => self.result(context, inputs.gas_limit, false),
            FrameInput::Create(inputs) => self.result(context, inputs.gas_limit(), true),
            FrameInput::Empty => unreachable!(),
        })
    }

    fn call(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        let FrameResult::Call(outcome) = self.result(context, inputs.gas_limit, false) else {
            unreachable!()
        };
        Some(outcome)
    }

    fn create(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        inputs: &mut CreateInputs,
    ) -> Option<CreateOutcome> {
        let FrameResult::Create(outcome) = self.result(context, inputs.gas_limit(), true) else {
            unreachable!()
        };
        Some(outcome)
    }
}

#[rstest]
fn short_circuit_callbacks_observe_rejection_without_inventing_create_nonce(
    #[values(false, true)] create: bool,
    #[values(false, true)] generic: bool,
    #[values(false, true)] advance_nonce: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let tx = tx(create, storage_and_log(), mon(3), &mut evm);
    let inspector =
        (ShortCircuitInspector { generic, advance_nonce }, FrameOutcomeInspector::default());
    let mut evm = evm.with_inspector(inspector);
    let out = evm.inspect_tx(tx).unwrap();
    let nonce = if create { u64::from(advance_nonce) } else { 1 };
    assert_violation(&out, nonce, GAS_LIMIT);
    let inspector = &evm.0.inspector.1;
    let mut expected = vec![("frame_start", 0)];
    if !generic {
        expected.push((if create { "create" } else { "call" }, 0));
    }
    expected.extend([(if create { "create_end" } else { "call_end" }, 0), ("frame_end", 0)]);
    assert_eq!(inspector.events, expected);
    assert_eq!(inspector.roots.len(), 2);
    for root in &inspector.roots {
        assert_eq!(root.result, InstructionResult::Revert);
        assert_eq!(root.state[&SENDER].info.nonce, nonce);
        assert_eq!(root.state[&SENDER].info.balance, mon(12) - U256::from(GAS_LIMIT));
        let recipient = if create { SENDER.create(0) } else { TARGET };
        assert_eq!(root.state[&recipient].info.balance, U256::ZERO);
    }
}

struct PendingErrorInspector;

impl Inspector<MonadContext<InMemoryDB>> for PendingErrorInspector {
    fn frame_start(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        input: &mut FrameInput,
    ) -> Option<FrameResult> {
        let (create, gas_limit) = match input {
            FrameInput::Call(inputs) => (false, inputs.gas_limit),
            FrameInput::Create(inputs) => (true, inputs.gas_limit()),
            FrameInput::Empty => unreachable!(),
        };
        let mut result = ShortCircuitInspector { generic: true, advance_nonce: create }
            .result(context, gas_limit, create);
        assert!(context.journaled_state.reserve_balance().has_violation());
        result.interpreter_result_mut().result = InstructionResult::FatalExternalError;
        context.error = Err(ContextError::Custom("host failure".into()));
        Some(result)
    }
}

struct FatalOutcomeEditor {
    success: bool,
}

impl FatalOutcomeEditor {
    fn edit(&self, context: &MonadContext<InMemoryDB>, result: &mut InterpreterResult) {
        assert!(context.error.is_err());
        assert_eq!(result.result, InstructionResult::FatalExternalError);
        if self.success {
            result.result = InstructionResult::Return;
        }
    }
}

impl Inspector<MonadContext<InMemoryDB>> for FatalOutcomeEditor {
    fn call_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        self.edit(context, &mut outcome.result);
    }

    fn create_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        self.edit(context, &mut outcome.result);
    }

    fn frame_end(
        &mut self,
        context: &mut MonadContext<InMemoryDB>,
        _input: &FrameInput,
        _result: &mut FrameResult,
    ) {
        assert!(context.error.is_err());
    }
}

#[rstest]
fn pending_host_error_preserves_fatal_callbacks_and_aborts_commit(
    #[values(false, true)] create: bool,
    #[values(false, true)] success: bool,
) {
    let mut evm = evm(MonadHardfork::MonadTen, true);
    let tx = tx(create, storage_and_log(), mon(3), &mut evm);
    let inspector =
        (PendingErrorInspector, (FrameOutcomeInspector::default(), FatalOutcomeEditor { success }));
    let mut evm = evm.with_inspector(inspector);
    assert!(
        matches!(evm.inspect_tx_commit(tx), Err(EVMError::Custom(message)) if message == "host failure")
    );
    let roots = &evm.0.inspector.1 .0.roots;
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].result, InstructionResult::FatalExternalError);
    assert_eq!(
        roots[1].result,
        if success { InstructionResult::Return } else { InstructionResult::FatalExternalError }
    );
    assert_eq!(evm.0.ctx.journaled_state.database.cache.accounts[&SENDER].info.balance, mon(12));
    assert_eq!(evm.0.ctx.journaled_state.database.cache.accounts[&SENDER].info.nonce, 0);
    assert!(evm.0.ctx.journaled_state.state.is_empty());
    assert_eq!(evm.0.ctx.journaled_state.depth(), 0);
    assert!(!evm.0.ctx.journaled_state.reserve_balance().tracking_enabled());

    let mut evm = evm.with_inspector(NoOpInspector);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(RECIPIENT)
        .value(mon(1))
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();
    assert!(evm.inspect_tx_commit(tx).unwrap().is_success());
}
