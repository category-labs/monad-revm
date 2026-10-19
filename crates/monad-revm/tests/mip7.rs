//! MIP-7 extension decoding and legacy bytecode analysis regressions.

use monad_revm::{
    extension::EXTENSION, monad_context_with_db, MonadBuilder, MonadCfgEnv, MonadHaltReason,
    MonadHardfork,
};
use revm::{
    bytecode::opcode,
    context::{result::ExecutionResult, TxEnv},
    context_interface::result::HaltReason,
    database::InMemoryDB,
    inspector::{InspectEvm, NoOpInspector},
    primitives::{Address, U256},
    state::{AccountInfo, Bytecode},
    ExecuteEvm,
};
use rstest::rstest;

const GAS_LIMIT: u64 = 100_000;

fn execute(spec: MonadHardfork, code: Vec<u8>, inspect: bool) -> ExecutionResult<MonadHaltReason> {
    let sender = Address::repeat_byte(0x11);
    let contract = Address::repeat_byte(0x22);
    let mut db = InMemoryDB::default();
    db.insert_account_info(
        sender,
        AccountInfo { balance: U256::from(1_000_000), ..Default::default() },
    );
    db.insert_account_info(
        contract,
        AccountInfo::default().with_code(Bytecode::new_raw(code.into())),
    );
    let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
    let tx =
        TxEnv::builder().caller(sender).to(contract).gas_limit(GAS_LIMIT).gas_price(0).build_fill();
    if inspect { evm.inspect_one_tx(tx) } else { evm.transact_one(tx) }.unwrap()
}

fn assert_halt(result: ExecutionResult<MonadHaltReason>, expected: HaltReason) {
    assert_eq!(result.tx_gas_used(), GAS_LIMIT);
    match result {
        ExecutionResult::Halt { reason, .. } => {
            assert_eq!(reason, MonadHaltReason::Ethereum(expected));
        }
        result => panic!("expected {expected:?}, got {result:?}"),
    }
}

fn invalid_extension_reason(spec: MonadHardfork) -> HaltReason {
    if spec == MonadHardfork::MonadNext {
        HaltReason::InvalidFEOpcode
    } else {
        HaltReason::OpcodeNotFound
    }
}

#[rstest]
fn unassigned_selectors(
    #[values(false, true)] inspect: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    for selector in 0..=u8::MAX {
        assert_halt(
            execute(spec, vec![EXTENSION, selector], inspect),
            invalid_extension_reason(spec),
        );
    }
}

#[rstest]
fn missing_selector(
    #[values(false, true)] inspect: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    for code in [vec![EXTENSION], vec![opcode::PUSH0, EXTENSION]] {
        assert_halt(execute(spec, code, inspect), invalid_extension_reason(spec));
    }
}

#[rstest]
fn legacy_jumpdest_analysis(
    #[values(false, true)] inspect: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let result = execute(
        spec,
        vec![opcode::PUSH1, 4, opcode::JUMP, EXTENSION, opcode::JUMPDEST, opcode::STOP],
        inspect,
    );
    assert!(result.is_success(), "{result:?}");
    assert_halt(
        execute(
            spec,
            vec![
                opcode::PUSH1,
                5,
                opcode::JUMP,
                EXTENSION,
                opcode::PUSH1,
                opcode::JUMPDEST,
                opcode::STOP,
            ],
            inspect,
        ),
        HaltReason::InvalidJump,
    );
    let result = execute(
        spec,
        vec![
            opcode::PUSH3,
            EXTENSION,
            0,
            opcode::JUMPDEST,
            opcode::PUSH0,
            opcode::MSTORE,
            opcode::PUSH1,
            32,
            opcode::PUSH0,
            opcode::RETURN,
        ],
        inspect,
    );
    assert!(result.is_success(), "{result:?}");
    assert_eq!(U256::from_be_slice(result.output().unwrap()), U256::from(0xAE005B));
}
