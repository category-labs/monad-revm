//! Transaction-level regressions for staking compound calls.

use alloy_sol_types::SolCall;
use monad_revm::{
    monad_context_with_db,
    staking::{
        constants::{DUST_THRESHOLD, MON, UNIT_BIAS},
        interface::IMonadStaking::compoundCall,
        storage::{
            accumulator_key, delegator_key, delegator_offsets, global_slots, validator_key,
            validator_offsets, STAKING_ADDRESS,
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
    primitives::{Address, U256},
    state::{AccountInfo, Bytecode},
    ExecuteCommitEvm,
};
use rstest::rstest;
use std::collections::BTreeMap;

fn left_aligned_u64(value: u64) -> U256 {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&value.to_be_bytes());
    U256::from_be_bytes(bytes)
}

fn staking_storage(db: &InMemoryDB) -> BTreeMap<U256, U256> {
    db.cache.accounts[&STAKING_ADDRESS]
        .storage
        .iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|(key, value)| (*key, *value))
        .collect()
}

fn catch_compound_revert() -> Bytecode {
    Bytecode::new_raw(
        vec![
            // Copy calldata and forward it to staking with 500,000 gas.
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
            0x07,
            0xa1,
            0x20,
            opcode::CALL,
            // Save the call status and persist a parent write after the child returns.
            opcode::PUSH0,
            opcode::MSTORE,
            opcode::PUSH1,
            1,
            opcode::PUSH1,
            1,
            opcode::SSTORE,
            // Return the call status followed by the child's complete returndata.
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
        ]
        .into(),
    )
}

#[rstest]
#[case(MonadHardfork::MonadEight)]
#[case(MonadHardfork::MonadNine)]
#[case(MonadHardfork::MonadTen)]
#[case(MonadHardfork::MonadNext)]
fn compound_dust_reverts_reward_settlement(
    #[case] spec: MonadHardfork,
    #[values(false, true)] nested: bool,
) {
    let sender = Address::repeat_byte(0x11);
    let proxy = Address::repeat_byte(0x22);
    let beneficiary = Address::repeat_byte(0x33);
    let delegator = if nested { proxy } else { sender };
    let validator_id = 1;
    let epoch = 10;
    let gas_limit = 1_000_000;
    let accumulator = UNIT_BIAS / DUST_THRESHOLD;

    let mut db = InMemoryDB::default();
    db.insert_account_info(sender, AccountInfo { balance: MON, ..Default::default() });
    db.insert_account_info(proxy, AccountInfo::default().with_code(catch_compound_revert()));
    db.insert_account_info(
        STAKING_ADDRESS,
        AccountInfo { balance: MON, nonce: 1, ..Default::default() },
    );
    let mut address_flags = [0; 32];
    address_flags[..20].copy_from_slice(Address::repeat_byte(0xaa).as_slice());
    address_flags[20..28].copy_from_slice(&validator_flags::STAKE_TOO_LOW.to_be_bytes());

    // Settlement earns one wei, activates pending stake, and clears the last accumulator
    // reference before the dust check rejects the compound. Every mutation must roll back.
    for (key, value) in [
        (global_slots::EPOCH, left_aligned_u64(epoch)),
        (
            validator_key(validator_id, validator_offsets::ADDRESS_FLAGS),
            U256::from_be_bytes(address_flags),
        ),
        (validator_key(validator_id, validator_offsets::STAKE), DUST_THRESHOLD * U256::from(2)),
        (validator_key(validator_id, validator_offsets::UNCLAIMED_REWARDS), U256::from(1)),
        (validator_key(validator_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN), accumulator),
        (delegator_key(validator_id, &delegator, delegator_offsets::STAKE), DUST_THRESHOLD),
        (delegator_key(validator_id, &delegator, delegator_offsets::DELTA_STAKE), DUST_THRESHOLD),
        (
            delegator_key(validator_id, &delegator, delegator_offsets::EPOCHS),
            left_aligned_u64(epoch),
        ),
        (accumulator_key(epoch, validator_id, 0), accumulator),
        (accumulator_key(epoch, validator_id, 1), U256::from(1)),
    ] {
        db.insert_account_storage(STAKING_ADDRESS, key, value).unwrap();
    }
    let initial_storage = staking_storage(&db);

    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 2;
    context.block.beneficiary = beneficiary;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(sender)
                .to(if nested { proxy } else { STAKING_ADDRESS })
                .data(compoundCall { validatorId: validator_id }.abi_encode().into())
                .gas_limit(gas_limit)
                .gas_price(7)
                .build_fill(),
        )
        .unwrap();

    let context = evm.ctx();
    let db = context.db();
    let output = result.output().unwrap();
    if nested {
        assert!(result.is_success(), "the parent must catch the child's revert");
        assert_eq!(U256::from_be_slice(&output[..32]), U256::ZERO);
        assert_eq!(&output[32..], b"delegation is too small");
        assert_eq!(db.cache.accounts[&proxy].storage.get(&U256::from(1)), Some(&U256::from(1)));
    } else {
        assert!(matches!(result, ExecutionResult::Revert { .. }));
        assert_eq!(output.as_ref(), b"delegation is too small");
    }
    assert!(result.logs().is_empty());
    assert_eq!(staking_storage(db), initial_storage);
    assert_eq!(db.cache.accounts[&STAKING_ADDRESS].info.balance, MON);
    assert_eq!(db.cache.accounts[&sender].info.nonce, 1);
    assert_eq!(
        db.cache.accounts[&sender].info.balance,
        MON - U256::from(gas_limit) * U256::from(7)
    );
    assert_eq!(db.cache.accounts[&beneficiary].info.balance, U256::from(gas_limit) * U256::from(5));
}
