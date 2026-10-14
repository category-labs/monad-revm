//! Fatal database error regressions for staking execution.

use alloy_sol_types::SolCall;
use monad_revm::{
    monad_context_with_db,
    staking::{
        constants::{MON, SYSTEM_ADDRESS, UNIT_BIAS},
        interface::IMonadStaking::{delegateCall, getEpochCall, syscallSnapshotCall, withdrawCall},
        run_staking_precompile, run_staking_with_reader,
        storage::{
            accumulator_key, consensus_view_key, global_slots, snapshot_view_key, validator_key,
            validator_offsets, valset_slots, withdrawal_key, withdrawal_offsets,
        },
        write::{run_staking_write, StakingStorage},
        StorageReader, STAKING_ADDRESS,
    },
    MonadBuilder, MonadCfgEnv, MonadContext, MonadHaltReason, MonadHardfork, MonadJournalTr,
};
use revm::{
    bytecode::opcode,
    context::{
        result::{EVMError, ExecutionResult},
        TxEnv,
    },
    context_interface::{ContextError, ContextTr, JournalTr},
    database::InMemoryDB,
    database_interface::DBErrorMarker,
    handler::{system_call::SystemCallEvm, EvmTr, FrameResult},
    inspector::{InspectCommitEvm, InspectEvm, NoOpInspector},
    interpreter::{
        CallInput, CallInputs, CallOutcome, CallScheme, CallValue, FrameInput, InstructionResult,
    },
    precompile::PrecompileHalt,
    primitives::{Address, Log, B256, U256},
    state::{AccountInfo, Bytecode, EvmState},
    Database, DatabaseCommit, ExecuteCommitEvm, ExecuteEvm, Inspector,
};
use rstest::rstest;
use std::{collections::BTreeMap, fmt, sync::Arc};

const SENDER: Address = Address::repeat_byte(0x11);
const PROXY: Address = Address::repeat_byte(0x22);
const BENEFICIARY: Address = Address::repeat_byte(0x33);
const GAS_LIMIT: u64 = 2_000_000;

#[derive(Debug)]
struct DbFailure(Arc<()>);

impl fmt::Display for DbFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("staking state unavailable")
    }
}

impl std::error::Error for DbFailure {}
impl DBErrorMarker for DbFailure {}

#[derive(Debug, Default)]
struct FailingDb {
    inner: InMemoryDB,
    storage_failure: Option<(U256, DbFailure)>,
    account_failure: Option<(Address, DbFailure)>,
    commits: usize,
}

impl Database for FailingDb {
    type Error = DbFailure;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        if self.account_failure.as_ref().is_some_and(|(target, _)| *target == address) {
            return Err(self.account_failure.take().unwrap().1);
        }
        Ok(self.inner.basic(address).unwrap())
    }

    fn code_by_hash(&mut self, hash: B256) -> Result<Bytecode, Self::Error> {
        Ok(self.inner.code_by_hash(hash).unwrap())
    }

    fn storage(&mut self, address: Address, key: U256) -> Result<U256, Self::Error> {
        if address == STAKING_ADDRESS
            && self.storage_failure.as_ref().is_some_and(|(target, _)| *target == key)
        {
            return Err(self.storage_failure.take().unwrap().1);
        }
        Ok(self.inner.storage(address, key).unwrap())
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        Ok(self.inner.block_hash(number).unwrap())
    }
}

impl DatabaseCommit for FailingDb {
    fn commit(&mut self, changes: EvmState) {
        self.commits += 1;
        self.inner.commit(changes);
    }
}

fn left_aligned_u64(value: u64) -> U256 {
    U256::from(value) << 192
}

fn slots(db: &FailingDb) -> BTreeMap<U256, U256> {
    db.inner.cache.accounts[&STAKING_ADDRESS]
        .storage
        .iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|(key, value)| (*key, *value))
        .collect()
}

fn catch_staking_failure() -> Bytecode {
    Bytecode::new_raw(
        vec![
            // Forward calldata to staking, then persist a write if execution resumes.
            opcode::CALLDATASIZE,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::CALLDATACOPY,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::CALLDATASIZE,
            opcode::PUSH0,
            opcode::PUSH0,
            opcode::PUSH2,
            0x10,
            0x00,
            opcode::PUSH3,
            0x0f,
            0x42,
            0x40,
            opcode::CALL,
            opcode::PUSH0,
            opcode::MSTORE,
            opcode::PUSH1,
            1,
            opcode::PUSH0,
            opcode::SSTORE,
            opcode::PUSH1,
            32,
            opcode::PUSH0,
            opcode::RETURN,
        ]
        .into(),
    )
}

#[derive(Clone, Copy, Debug)]
enum FailureSite {
    Getter,
    JournalRead,
    JournalWrite,
}

#[rstest]
#[case::getter(FailureSite::Getter)]
#[case::journal_read_after_write(FailureSite::JournalRead)]
#[case::journal_write_after_write(FailureSite::JournalWrite)]
fn database_failure_aborts_transaction(
    #[case] site: FailureSite,
    #[values(false, true)] nested: bool,
    #[values(false, true)] commit: bool,
    #[values(false, true)] inspect: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let proxy = if matches!(site, FailureSite::Getter) { PROXY } else { SYSTEM_ADDRESS };
    let sender =
        if nested || matches!(site, FailureSite::Getter) { SENDER } else { SYSTEM_ADDRESS };
    let mut db = FailingDb::default();
    db.inner.insert_account_info(sender, AccountInfo { balance: MON, ..Default::default() });
    db.inner.insert_account_info(
        STAKING_ADDRESS,
        AccountInfo { balance: MON, nonce: 1, ..Default::default() },
    );
    if nested {
        db.inner
            .insert_account_info(proxy, AccountInfo::default().with_code(catch_staking_failure()));
    }
    for (key, value) in [
        (global_slots::EPOCH, left_aligned_u64(10)),
        (valset_slots::CONSENSUS, left_aligned_u64(1)),
        (valset_slots::CONSENSUS + U256::from(1), left_aligned_u64(1)),
        (consensus_view_key(1, 0), MON),
        (consensus_view_key(1, 1), U256::from(123)),
    ] {
        db.inner.insert_account_storage(STAKING_ADDRESS, key, value).unwrap();
    }
    let initial_storage = slots(&db);
    let marker = Arc::new(());
    let key = match site {
        FailureSite::Getter => global_slots::EPOCH,
        // Snapshot sets IN_BOUNDARY before either of these accesses.
        FailureSite::JournalRead => valset_slots::CONSENSUS,
        // The snapshot array and stake have also been written before this cold SSTORE.
        FailureSite::JournalWrite => snapshot_view_key(1, 1),
    };
    db.storage_failure = Some((key, DbFailure(marker.clone())));
    // A post-execution failure must not replace the staking database error.
    db.account_failure = Some((BENEFICIARY, DbFailure(Arc::new(()))));
    let data = if matches!(site, FailureSite::Getter) {
        getEpochCall {}.abi_encode()
    } else {
        syscallSnapshotCall {}.abi_encode()
    };
    let tx = TxEnv::builder()
        .caller(sender)
        .to(if nested { proxy } else { STAKING_ADDRESS })
        .data(data.into())
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();
    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    context.block.beneficiary = BENEFICIARY;
    let mut evm = context.build_monad_with_inspector(NoOpInspector);
    let result = match (inspect, commit) {
        (false, false) => evm.transact(tx.clone()).map(|result| result.result),
        (false, true) => evm.transact_commit(tx.clone()),
        (true, false) => evm.inspect_tx(tx.clone()).map(|result| result.result),
        (true, true) => evm.inspect_tx_commit(tx.clone()),
    };
    let error = result.expect_err("database failures must abort execution");
    let EVMError::Database(DbFailure(original)) = error else {
        panic!("expected the original database error, got {error:?}");
    };
    assert!(Arc::ptr_eq(&original, &marker));
    assert!(evm.ctx().error().is_ok());
    assert_eq!(evm.ctx().db().commits, 0);
    assert_eq!(slots(evm.ctx().db()), initial_storage);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&sender].info.nonce, 0);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&sender].info.balance, MON);
    assert!(evm.finalize().values().all(|account| !account.is_touched()));

    assert!(evm.ctx().db_mut().account_failure.take().is_some());

    // The one-shot failure is gone. Reusing the EVM must charge exactly once.
    let result = evm.transact_commit(tx).unwrap();
    assert!(result.is_success());
    assert_eq!(evm.ctx().db().commits, 1);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&sender].info.nonce, 1);
    assert_eq!(
        evm.ctx().db().inner.cache.accounts[&sender].info.balance,
        MON - U256::from(GAS_LIMIT)
    );
}

#[derive(Default)]
struct FatalErrorInspector {
    outcomes: Vec<(InstructionResult, bool)>,
    frame_outcomes: Vec<(InstructionResult, bool)>,
    edit_outcome: bool,
}

impl Inspector<MonadContext<FailingDb>> for FatalErrorInspector {
    fn call_end(
        &mut self,
        context: &mut MonadContext<FailingDb>,
        _inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        let pending_error = context.error().is_err();
        self.outcomes.push((outcome.result.result, pending_error));
        if pending_error && self.edit_outcome {
            outcome.result.result = InstructionResult::Return;
        }
    }

    fn frame_end(
        &mut self,
        context: &mut MonadContext<FailingDb>,
        _inputs: &FrameInput,
        outcome: &mut FrameResult,
    ) {
        self.frame_outcomes.push((outcome.instruction_result(), context.error().is_err()));
    }
}

#[rstest]
fn database_failure_takes_precedence_over_reserve_violation(
    #[values(false, true)] inspect: bool,
    #[values(false, true)] nested: bool,
    #[values(false, true)] edit_outcome: bool,
) {
    let initial_balance = MON * U256::from(12);
    let marker = Arc::new(());
    let failure_key =
        if nested { global_slots::EPOCH } else { validator_key(1, validator_offsets::STAKE) };
    let target = if nested { PROXY } else { STAKING_ADDRESS };
    let mut db = FailingDb {
        storage_failure: Some((failure_key, DbFailure(marker.clone()))),
        ..Default::default()
    };
    db.inner.insert_account_info(
        SENDER,
        AccountInfo { balance: initial_balance, ..Default::default() },
    );
    db.inner.insert_account_info(PROXY, AccountInfo::default().with_code(catch_staking_failure()));
    let mut context = monad_context_with_db(db);
    context.block.basefee = 0;
    context.block.beneficiary = BENEFICIARY;
    context.chain.parent_senders_and_authorities.insert(SENDER);
    let mut evm = context
        .build_monad_with_inspector(FatalErrorInspector { edit_outcome, ..Default::default() });
    let data = if nested {
        getEpochCall {}.abi_encode()
    } else {
        delegateCall { validatorId: 1 }.abi_encode()
    };
    let mut tx = TxEnv::builder()
        .caller(SENDER)
        .to(target)
        .value(MON * U256::from(3))
        .data(data.into())
        .gas_limit(GAS_LIMIT)
        .gas_price(1)
        .build_fill();

    // The initial transfer crosses the reserve before the staking read fails.
    let outcome =
        if inspect { evm.inspect_tx_commit(tx.clone()) } else { evm.transact_commit(tx.clone()) };
    let Err(EVMError::Database(DbFailure(original))) = outcome else {
        panic!("database failure must take precedence, got {outcome:?}");
    };
    assert!(Arc::ptr_eq(&original, &marker));
    if inspect {
        assert_eq!(evm.0.inspector.outcomes, [(InstructionResult::FatalExternalError, true)]);
        let expected = if edit_outcome {
            InstructionResult::Return
        } else {
            InstructionResult::FatalExternalError
        };
        assert_eq!(evm.0.inspector.frame_outcomes, [(expected, true)]);
    } else {
        assert!(evm.0.inspector.outcomes.is_empty());
        assert!(evm.0.inspector.frame_outcomes.is_empty());
    }
    assert_eq!(evm.ctx().db().commits, 0);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&SENDER].info.balance, initial_balance);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&SENDER].info.nonce, 0);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&target].info.balance, U256::ZERO);
    assert!(evm.ctx().journal().evm_state().values().all(|account| !account.is_touched()));
    assert!(evm.ctx().error().is_ok());
    assert_eq!(evm.ctx().journal().depth(), 0);
    assert!(!evm.ctx().journal().reserve_balance().tracking_enabled());
    assert!(!evm.ctx().journal().reserve_balance().has_violation());

    // Retrying after the database recovers now rejects only the execution effects.
    let result = if inspect {
        evm.inspect_tx_commit(tx.clone()).unwrap()
    } else {
        evm.transact_commit(tx.clone()).unwrap()
    };
    assert!(matches!(
        result,
        ExecutionResult::Halt { reason: MonadHaltReason::ReserveBalanceViolation, .. }
    ));
    let db = evm.ctx_ref().db();
    assert_eq!(db.commits, 1);
    assert_eq!(
        db.inner.cache.accounts[&SENDER].info.balance,
        initial_balance - U256::from(GAS_LIMIT)
    );
    assert_eq!(db.inner.cache.accounts[&SENDER].info.nonce, 1);
    assert_eq!(db.inner.cache.accounts[&BENEFICIARY].info.balance, U256::from(GAS_LIMIT));
    assert_eq!(db.inner.cache.accounts[&target].info.balance, U256::ZERO);
    assert!(db.inner.cache.accounts[&target].storage.values().all(U256::is_zero));

    // Neither failure may leak into the next valid transaction on the same EVM.
    tx.value = U256::ZERO;
    tx.nonce = 1;
    tx.data = getEpochCall {}.abi_encode().into();
    let result =
        if inspect { evm.inspect_tx_commit(tx).unwrap() } else { evm.transact_commit(tx).unwrap() };
    assert!(result.is_success());
    assert_eq!(evm.ctx().db().commits, 2);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&SENDER].info.nonce, 2);
    assert_eq!(
        evm.ctx().db().inner.cache.accounts[&SENDER].info.balance,
        initial_balance - U256::from(2 * GAS_LIMIT)
    );
    if nested {
        assert_eq!(evm.ctx().db().inner.cache.accounts[&PROXY].storage[&U256::ZERO], U256::from(1));
    }
}

#[rstest]
fn staking_contract_error_remains_catchable(#[values(false, true)] nested: bool) {
    let mut db = FailingDb::default();
    db.inner.insert_account_info(SENDER, AccountInfo { balance: MON, ..Default::default() });
    db.inner.insert_account_info(PROXY, AccountInfo::default().with_code(catch_staking_failure()));
    let mut context = monad_context_with_db(db);
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(SENDER)
                .to(if nested { PROXY } else { STAKING_ADDRESS })
                .data(delegateCall { validatorId: 1 }.abi_encode().into())
                .gas_limit(GAS_LIMIT)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();
    if nested {
        assert!(result.is_success());
        assert_eq!(result.output().unwrap().as_ref(), &[0; 32]);
        assert_eq!(evm.ctx().db().inner.cache.accounts[&PROXY].storage[&U256::ZERO], U256::from(1));
    } else {
        assert!(matches!(result, ExecutionResult::Revert { .. }));
        assert_eq!(result.output().unwrap().as_ref(), b"unknown validator");
    }
    assert_eq!(evm.ctx().db().commits, 1);
    assert_eq!(evm.ctx().db().inner.cache.accounts[&SENDER].info.nonce, 1);
}

#[rstest]
fn transfer_errors_remain_distinct_after_storage_mutation(
    #[values(false, true)] database_failure: bool,
) {
    let mut db = FailingDb::default();
    db.inner.insert_account_info(
        STAKING_ADDRESS,
        AccountInfo {
            balance: if database_failure { MON } else { U256::ZERO },
            nonce: 1,
            ..Default::default()
        },
    );
    for (key, value) in [
        (global_slots::EPOCH, left_aligned_u64(10)),
        (withdrawal_key(1, &SENDER, 0, withdrawal_offsets::AMOUNT), U256::from(2)),
        (withdrawal_key(1, &SENDER, 0, withdrawal_offsets::EPOCH), left_aligned_u64(1)),
        (accumulator_key(1, 1, 0), UNIT_BIAS / U256::from(2)),
        (accumulator_key(1, 1, 1), U256::from(1)),
        (validator_key(1, validator_offsets::UNCLAIMED_REWARDS), U256::from(1)),
    ] {
        db.inner.insert_account_storage(STAKING_ADDRESS, key, value).unwrap();
    }
    let marker = Arc::new(());
    if database_failure {
        db.account_failure = Some((SENDER, DbFailure(marker.clone())));
    }
    let mut context = monad_context_with_db(db);
    context.journal_mut().load_account(STAKING_ADDRESS).unwrap();
    let inputs = CallInputs {
        input: CallInput::Bytes(withdrawCall { validatorId: 1, withdrawId: 0 }.abi_encode().into()),
        return_memory_offset: 0..0,
        gas_limit: GAS_LIMIT,
        reservoir: 0,
        bytecode_address: STAKING_ADDRESS,
        known_bytecode: (B256::ZERO, Bytecode::default()),
        target_address: STAKING_ADDRESS,
        caller: SENDER,
        value: CallValue::Transfer(U256::ZERO),
        scheme: CallScheme::Call,
        is_static: false,
        charged_new_account_state_gas: false,
    };
    // The caller is intentionally not preloaded to exercise the adapter's transfer account read.
    let result = run_staking_precompile(&mut context, &inputs).unwrap().unwrap();
    assert_eq!(
        context.journal_mut().sload(STAKING_ADDRESS, accumulator_key(1, 1, 1)).unwrap().data,
        U256::ZERO
    );
    if database_failure {
        assert_eq!(result.result, revm::interpreter::InstructionResult::FatalExternalError);
        let Err(ContextError::Db(DbFailure(original))) =
            core::mem::replace(context.error(), Ok(()))
        else {
            panic!("transfer database error was not retained");
        };
        assert!(Arc::ptr_eq(&original, &marker));
    } else {
        assert_eq!(result.result, revm::interpreter::InstructionResult::Revert);
        assert_eq!(result.output.as_ref(), b"Transfer failed: OutOfFunds");
        assert!(context.error().is_ok());
    }
    assert_eq!(context.db().commits, 0);
}

#[test]
fn system_call_database_error_discards_partial_snapshot() {
    let marker = Arc::new(());
    let mut db = FailingDb {
        storage_failure: Some((valset_slots::CONSENSUS, DbFailure(marker.clone()))),
        ..Default::default()
    };
    db.inner.insert_account_info(STAKING_ADDRESS, AccountInfo { nonce: 1, ..Default::default() });
    let mut evm = monad_context_with_db(db).build_monad();
    let data = syscallSnapshotCall {}.abi_encode().into();
    let error = evm.system_call_one_with_caller(SYSTEM_ADDRESS, STAKING_ADDRESS, data).unwrap_err();
    let EVMError::Database(DbFailure(original)) = error else {
        panic!("expected the original database error, got {error:?}");
    };
    assert!(Arc::ptr_eq(&original, &marker));
    assert!(evm.ctx().error().is_ok());
    assert_eq!(
        evm.ctx().journal_mut().sload(STAKING_ADDRESS, global_slots::IN_BOUNDARY).unwrap().data,
        U256::ZERO
    );
    assert_eq!(evm.ctx().db().commits, 0);
}

#[derive(Default)]
struct PendingErrorStorage(Option<String>);

impl StorageReader for PendingErrorStorage {
    fn sload(&mut self, _key: U256) -> Result<U256, PrecompileHalt> {
        Ok(U256::ZERO)
    }

    fn take_fatal_error(&mut self) -> Option<String> {
        self.0.take()
    }
}

impl StakingStorage for PendingErrorStorage {
    fn sstore(&mut self, _key: U256, _value: U256) -> Result<(), PrecompileHalt> {
        Ok(())
    }

    fn transfer(
        &mut self,
        _from: Address,
        _to: Address,
        _amount: U256,
    ) -> Result<(), PrecompileHalt> {
        Ok(())
    }

    fn emit_log(&mut self, _log: Log) -> Result<(), PrecompileHalt> {
        Ok(())
    }
}

#[rstest]
#[case::short(vec![])]
#[case::unknown(vec![0xde, 0xad, 0xbe, 0xef])]
#[case::getter(getEpochCall {}.abi_encode())]
#[case::writer(delegateCall { validatorId: 1 }.abi_encode())]
#[case::malformed(delegateCall::SELECTOR.to_vec())]
fn fatal_adapter_error_is_drained_on_every_exit(
    #[case] input: Vec<u8>,
    #[values(false, true)] write: bool,
    #[values(0, GAS_LIMIT)] gas_limit: u64,
    #[values(U256::ZERO, U256::from(1))] value: U256,
) {
    let mut storage = PendingErrorStorage(Some("original storage failure".into()));
    let mut run = || {
        if write {
            run_staking_write(&input, gas_limit, &mut storage, &SENDER, value)
        } else {
            run_staking_with_reader(&input, gas_limit, &mut storage, value)
        }
    };
    assert_eq!(run().unwrap_err(), "original storage failure");
    assert!(run().is_ok(), "the previous fatal error must not leak into the next call");
}
