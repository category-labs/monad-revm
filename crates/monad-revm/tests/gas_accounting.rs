//! Charged transaction gas and execution consumption remain distinct.

use monad_revm::{
    monad_context_with_db, MonadBuilder, MonadCfgEnv, MonadContext, MonadEvm, MonadHardfork,
};
use revm::{
    bytecode::opcode,
    context::TxEnv,
    context_interface::result::ExecutionResult,
    database::InMemoryDB,
    handler::system_call::{SystemCallEvm, SYSTEM_CALL_GAS_LIMIT},
    inspector::InspectSystemCallEvm,
    interpreter::{CallInputs, CallOutcome},
    primitives::{Address, Bytes, U256},
    state::{AccountInfo, Bytecode},
    DatabaseCommit, ExecuteEvm, InspectEvm, Inspector,
};
use rstest::rstest;

const SENDER: Address = Address::repeat_byte(0x11);
const TARGET: Address = Address::repeat_byte(0x22);
const BENEFICIARY: Address = Address::repeat_byte(0x33);
const INITIAL_BALANCE: u64 = 1_000_000;

#[derive(Default)]
struct RemainingGasInspector {
    remaining: Option<u64>,
}

impl Inspector<MonadContext<InMemoryDB>> for RemainingGasInspector {
    fn call_end(
        &mut self,
        _context: &mut MonadContext<InMemoryDB>,
        _inputs: &CallInputs,
        outcome: &mut CallOutcome,
    ) {
        self.remaining = Some(outcome.result.gas.remaining());
    }
}

fn evm(spec: MonadHardfork) -> MonadEvm<MonadContext<InMemoryDB>, RemainingGasInspector> {
    let mut db = InMemoryDB::default();
    db.insert_account_info(
        SENDER,
        AccountInfo { balance: U256::from(INITIAL_BALANCE), ..Default::default() },
    );
    let mut ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    ctx.block.basefee = 0;
    ctx.block.beneficiary = BENEFICIARY;
    ctx.build_monad_with_inspector(RemainingGasInspector::default())
}

#[rstest]
fn transactions_report_charged_and_consumed_gas(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
) {
    let mut evm = evm(spec);
    let mut cumulative_gas_used = 0;
    for (nonce, (code, gas_limit, consumed, cumulative)) in [
        (vec![], 100_000, 21_000, 100_000),
        (vec![opcode::PUSH0, opcode::PUSH0, opcode::REVERT], 120_000, 21_004, 220_000),
        (vec![opcode::INVALID], 140_000, 140_000, 360_000),
    ]
    .into_iter()
    .enumerate()
    {
        evm.0.ctx.journaled_state.database.insert_account_info(
            TARGET,
            AccountInfo::default().with_code(Bytecode::new_raw(code.into())),
        );
        let tx = TxEnv::builder()
            .caller(SENDER)
            .to(TARGET)
            .nonce(nonce as u64)
            .gas_limit(gas_limit)
            .gas_price(1)
            .build_fill();
        let out = if inspect { evm.inspect_tx(tx).unwrap() } else { evm.transact(tx).unwrap() };
        match (&out.result, nonce) {
            (ExecutionResult::Success { .. }, 0)
            | (ExecutionResult::Revert { .. }, 1)
            | (ExecutionResult::Halt { .. }, 2) => {}
            _ => panic!("unexpected result: {:?}", out.result),
        }
        assert_eq!(out.result.tx_gas_used(), gas_limit);
        assert_eq!(out.result.gas().block_regular_gas_used(), gas_limit);
        assert_eq!(out.result.gas().block_state_gas_used(), 0);
        assert_eq!(out.result.gas().total_gas_spent(), consumed);
        assert_eq!(out.result.gas().floor_gas(), gas_limit);
        assert_eq!(out.result.gas().inner_refunded(), 0);
        assert_eq!(out.result.gas().final_refunded(), 0);
        cumulative_gas_used += out.result.tx_gas_used();
        assert_eq!(cumulative_gas_used, cumulative);
        assert_eq!(out.state[&SENDER].info.balance, U256::from(INITIAL_BALANCE - cumulative));
        assert_eq!(out.state[&SENDER].info.nonce, nonce as u64 + 1);
        assert_eq!(out.state[&BENEFICIARY].info.balance, U256::from(cumulative));
        if inspect && !out.result.is_halt() {
            assert_eq!(evm.0.inspector.remaining, Some(gas_limit - consumed));
        }
        evm.0.ctx.journaled_state.database.commit(out.state);
    }
}

#[rstest]
fn calldata_floor_does_not_replace_execution_consumption(#[values(false, true)] inspect: bool) {
    let mut evm = evm(MonadHardfork::MonadTen);
    let tx = TxEnv::builder()
        .caller(SENDER)
        .to(TARGET)
        .data(Bytes::from_static(b"hello"))
        .gas_limit(100_000)
        .gas_price(0)
        .build_fill();
    let out = if inspect { evm.inspect_tx(tx).unwrap() } else { evm.transact(tx).unwrap() };
    assert!(out.result.is_success());
    assert_eq!(out.result.tx_gas_used(), 100_000);
    assert_eq!(out.result.gas().total_gas_spent(), 21_080);
    assert_eq!(out.state[&SENDER].info.balance, U256::from(INITIAL_BALANCE));
    if inspect {
        assert_eq!(evm.0.inspector.remaining, Some(100_000 - 21_080));
    }
}

#[rstest]
fn system_calls_retain_execution_only_gas(
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
    #[values(false, true)] inspect: bool,
) {
    for (code, consumed) in [
        (vec![], 0),
        (vec![opcode::PUSH0, opcode::PUSH0, opcode::REVERT], 4),
        (vec![opcode::INVALID], SYSTEM_CALL_GAS_LIMIT),
    ] {
        let mut evm = evm(spec);
        evm.0.ctx.journaled_state.database.insert_account_info(
            TARGET,
            AccountInfo::default().with_code(Bytecode::new_raw(code.into())),
        );
        let out = if inspect {
            evm.inspect_system_call_with_caller(SENDER, TARGET, Bytes::new()).unwrap()
        } else {
            evm.system_call_with_caller(SENDER, TARGET, Bytes::new()).unwrap()
        };
        assert_eq!(out.result.tx_gas_used(), consumed);
        assert_eq!(out.result.gas().total_gas_spent(), consumed);
        assert_eq!(out.result.gas().floor_gas(), 0);
        assert_eq!(out.state[&SENDER].info.balance, U256::from(INITIAL_BALANCE));
        assert_eq!(out.state[&SENDER].info.nonce, 0);
        assert!(out.state.get(&BENEFICIARY).is_none_or(|account| account.info.balance.is_zero()));
    }
}

#[test]
fn replay_reports_the_transaction_charge() {
    let mut evm = evm(MonadHardfork::MonadTen);
    evm.0.ctx.tx =
        TxEnv::builder().caller(SENDER).to(TARGET).gas_limit(100_000).gas_price(1).build_fill();
    let out = evm.replay().unwrap();
    assert_eq!(out.result.tx_gas_used(), 100_000);
    assert_eq!(out.result.gas().total_gas_spent(), 21_000);
    assert_eq!(out.state[&SENDER].info.balance, U256::from(900_000));
}
