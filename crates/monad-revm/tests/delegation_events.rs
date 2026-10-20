//! Transaction-level ordering and rollback regressions for staking delegation events.

use alloy_sol_types::{SolCall, SolEvent};
use monad_revm::{
    monad_context_with_db,
    staking::{
        constants::{ACTIVE_VALIDATOR_STAKE, DUST_THRESHOLD, MIN_AUTH_ADDRESS_STAKE, MON},
        interface::IMonadStaking::{
            addValidatorCall, compoundCall, delegateCall, ClaimRewards, Delegate, ValidatorCreated,
            ValidatorStatusChanged,
        },
        storage::{
            delegator_key, delegator_offsets, global_slots, validator_key, validator_offsets,
            STAKING_ADDRESS,
        },
        types::validator_flags,
    },
    MonadBuilder, MonadCfgEnv, MonadHardfork,
};
use revm::{
    bytecode::opcode,
    context::{result::ExecutionResult, TxEnv},
    context_interface::ContextTr,
    database::InMemoryDB,
    handler::EvmTr,
    primitives::{Address, Log, U256},
    state::{AccountInfo, Bytecode},
    ExecuteCommitEvm,
};
use rstest::rstest;
use std::collections::BTreeMap;

const SENDER: Address = Address::repeat_byte(0x11);
const PROXY: Address = Address::repeat_byte(0x22);
const VALIDATOR_ID: u64 = 1;
const EPOCH: u64 = 10;
const GAS_LIMIT: u64 = 1_000_000;

fn database(in_boundary: bool) -> InMemoryDB {
    let mut db = InMemoryDB::default();
    db.insert_account_info(
        SENDER,
        AccountInfo { balance: ACTIVE_VALIDATOR_STAKE + MON, ..Default::default() },
    );
    db.insert_account_info(
        STAKING_ADDRESS,
        AccountInfo { balance: MON, nonce: 1, ..Default::default() },
    );
    set_storage(&mut db, global_slots::EPOCH, U256::from(EPOCH) << 192);
    set_storage(&mut db, global_slots::IN_BOUNDARY, U256::from(in_boundary));
    db
}

fn set_storage(db: &mut InMemoryDB, key: U256, value: U256) {
    db.insert_account_storage(STAKING_ADDRESS, key, value).unwrap();
}

fn set_validator(db: &mut InMemoryDB, auth_address: Address, stake: U256, flags: u64) {
    let mut address_flags = [0; 32];
    address_flags[..20].copy_from_slice(auth_address.as_slice());
    address_flags[20..28].copy_from_slice(&flags.to_be_bytes());
    set_storage(
        db,
        validator_key(VALIDATOR_ID, validator_offsets::ADDRESS_FLAGS),
        U256::from_be_bytes(address_flags),
    );
    set_storage(db, validator_key(VALIDATOR_ID, validator_offsets::STAKE), stake);
}

fn staking_storage(db: &InMemoryDB) -> BTreeMap<U256, U256> {
    db.cache.accounts[&STAKING_ADDRESS]
        .storage
        .iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|(key, value)| (*key, *value))
        .collect()
}

fn event_log(event: impl SolEvent) -> Log {
    Log { address: STAKING_ADDRESS, data: event.encode_log_data() }
}

#[rstest]
#[case::unchanged(validator_flags::OK)]
#[case::stake_threshold(validator_flags::STAKE_TOO_LOW)]
#[case::auth_reactivation(validator_flags::WITHDRAWN)]
#[case::both_flags(validator_flags::STAKE_TOO_LOW | validator_flags::WITHDRAWN)]
fn delegate_and_compound_event_order(
    #[case] flags: u64,
    #[values(false, true)] compound: bool,
    #[values(false, true)] in_boundary: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let amount = DUST_THRESHOLD;
    let validator_stake = if flags & validator_flags::STAKE_TOO_LOW != 0 {
        ACTIVE_VALIDATOR_STAKE - amount
    } else {
        ACTIVE_VALIDATOR_STAKE
    };
    let delegator_stake = if flags & validator_flags::WITHDRAWN != 0 {
        MIN_AUTH_ADDRESS_STAKE - amount
    } else {
        MIN_AUTH_ADDRESS_STAKE
    };
    let mut db = database(in_boundary);
    set_validator(&mut db, SENDER, validator_stake, flags);
    set_storage(
        &mut db,
        delegator_key(VALIDATOR_ID, &SENDER, delegator_offsets::STAKE),
        delegator_stake,
    );
    if compound {
        set_storage(
            &mut db,
            delegator_key(VALIDATOR_ID, &SENDER, delegator_offsets::REWARDS),
            amount,
        );
    }
    let data = if compound {
        compoundCall { validatorId: VALIDATOR_ID }.abi_encode()
    } else {
        delegateCall { validatorId: VALIDATOR_ID }.abi_encode()
    };
    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(SENDER)
                .to(STAKING_ADDRESS)
                .data(data.into())
                .value(if compound { U256::ZERO } else { amount })
                .gas_limit(GAS_LIMIT)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();

    assert!(result.is_success());
    assert_eq!(result.output().unwrap().as_ref(), delegateCall::abi_encode_returns(&true));
    let mut expected = Vec::new();
    if compound {
        expected.push(event_log(ClaimRewards {
            validatorId: VALIDATOR_ID,
            delegator: SENDER,
            amount,
            epoch: EPOCH,
        }));
    }
    expected.push(event_log(Delegate {
        validatorId: VALIDATOR_ID,
        delegator: SENDER,
        amount,
        activationEpoch: EPOCH + if in_boundary { 2 } else { 1 },
    }));
    if flags != validator_flags::OK {
        expected.push(event_log(ValidatorStatusChanged {
            validatorId: VALIDATOR_ID,
            flags: validator_flags::OK,
        }));
    }
    assert_eq!(result.logs(), expected);
}

#[rstest]
fn registration_event_order(
    #[values(MIN_AUTH_ADDRESS_STAKE, ACTIVE_VALIDATOR_STAKE)] amount: U256,
    #[values(false, true)] in_boundary: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let auth_address = Address::repeat_byte(0xaa);
    let commission = MON / U256::from(10);
    let mut payload = Vec::new();
    payload.extend_from_slice(&[2; 33]);
    payload.extend_from_slice(&[3; 48]);
    payload.extend_from_slice(auth_address.as_slice());
    payload.extend_from_slice(&amount.to_be_bytes::<32>());
    payload.extend_from_slice(&commission.to_be_bytes::<32>());
    let data = addValidatorCall {
        payload: payload.into(),
        signedSecpMessage: vec![0; 64].into(),
        signedBlsMessage: vec![0; 96].into(),
    }
    .abi_encode();

    let mut context =
        monad_context_with_db(database(in_boundary)).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(SENDER)
                .to(STAKING_ADDRESS)
                .data(data.into())
                .value(amount)
                .gas_limit(GAS_LIMIT)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();

    assert!(result.is_success());
    assert_eq!(
        result.output().unwrap().as_ref(),
        addValidatorCall::abi_encode_returns(&VALIDATOR_ID)
    );
    let mut expected = vec![
        event_log(ValidatorCreated {
            validatorId: VALIDATOR_ID,
            authAddress: auth_address,
            commission,
        }),
        event_log(Delegate {
            validatorId: VALIDATOR_ID,
            delegator: auth_address,
            amount,
            activationEpoch: EPOCH + if in_boundary { 2 } else { 1 },
        }),
    ];
    if amount >= ACTIVE_VALIDATOR_STAKE {
        expected.push(event_log(ValidatorStatusChanged {
            validatorId: VALIDATOR_ID,
            flags: validator_flags::OK,
        }));
    }
    assert_eq!(result.logs(), expected);
}

fn catch_staking_revert(value: U256) -> Bytecode {
    let mut code = vec![
        // Copy calldata and forward it to staking with 500,000 gas.
        opcode::CALLDATASIZE,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::CALLDATACOPY,
        opcode::PUSH0,
        opcode::PUSH0,
        opcode::CALLDATASIZE,
        opcode::PUSH0,
        opcode::PUSH32,
    ];
    code.extend_from_slice(&value.to_be_bytes::<32>());
    code.extend_from_slice(&[
        opcode::PUSH2,
        0x10,
        0x00,
        opcode::PUSH3,
        0x07,
        0xa1,
        0x20,
        opcode::CALL,
        // Save the child status and persist a parent write after the child returns.
        opcode::PUSH0,
        opcode::MSTORE,
        opcode::PUSH1,
        1,
        opcode::PUSH1,
        1,
        opcode::SSTORE,
        // Return the child status followed by its complete returndata.
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

#[rstest]
fn later_failure_reverts_delegation_events(
    #[values(false, true)] compound: bool,
    #[values(false, true)] nested: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let delegator = if nested { PROXY } else { SENDER };
    let value = if compound { U256::ZERO } else { DUST_THRESHOLD };
    let mut db = database(false);
    db.insert_account_info(
        PROXY,
        AccountInfo { balance: MON, ..Default::default() }.with_code(catch_staking_revert(value)),
    );
    set_validator(&mut db, delegator, U256::MAX, validator_flags::OK);
    if compound {
        set_storage(
            &mut db,
            delegator_key(VALIDATOR_ID, &delegator, delegator_offsets::REWARDS),
            DUST_THRESHOLD,
        );
    }
    let initial_storage = staking_storage(&db);
    let data = if compound {
        compoundCall { validatorId: VALIDATOR_ID }.abi_encode()
    } else {
        delegateCall { validatorId: VALIDATOR_ID }.abi_encode()
    };
    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(SENDER)
                .to(if nested { PROXY } else { STAKING_ADDRESS })
                .data(data.into())
                .value(if nested { U256::ZERO } else { value })
                .gas_limit(GAS_LIMIT)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();

    let db = evm.ctx_ref().db();
    let output = result.output().unwrap();
    if nested {
        assert!(result.is_success(), "the parent must catch the child's revert");
        assert_eq!(&output[..32], &[0; 32]);
        assert_eq!(&output[32..], b"internal error");
        assert_eq!(db.cache.accounts[&PROXY].storage.get(&U256::from(1)), Some(&U256::from(1)));
    } else {
        assert!(matches!(result, ExecutionResult::Revert { .. }));
        assert_eq!(output.as_ref(), b"internal error");
    }
    assert!(result.logs().is_empty());
    assert_eq!(staking_storage(db), initial_storage);
    assert_eq!(db.cache.accounts[&STAKING_ADDRESS].info.balance, MON);
    assert_eq!(db.cache.accounts[&PROXY].info.balance, MON);
    assert_eq!(
        db.cache.accounts[&SENDER].info.balance,
        ACTIVE_VALIDATOR_STAKE + MON - U256::from(GAS_LIMIT)
    );
    assert_eq!(db.cache.accounts[&SENDER].info.nonce, 1);
}
