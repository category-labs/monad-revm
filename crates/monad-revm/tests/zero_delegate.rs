//! Zero-value delegation regressions for the staking precompile.

use alloy_sol_types::SolCall;
use monad_revm::{
    monad_context_with_db,
    staking::{
        abi::gas,
        constants::{DUST_THRESHOLD, MON},
        interface::IMonadStaking::delegateCall,
        storage::{validator_key, validator_offsets},
        write::{run_staking_write, StakingStorage},
        StorageReader, STAKING_ADDRESS,
    },
    MonadBuilder, MonadCfgEnv, MonadHardfork,
};
use revm::{
    context::{result::ExecutionResult, TxEnv},
    context_interface::ContextTr,
    database::InMemoryDB,
    handler::EvmTr,
    interpreter::InstructionResult,
    precompile::PrecompileHalt,
    primitives::{Address, Log, U256},
    state::AccountInfo,
    ExecuteCommitEvm,
};
use rstest::rstest;
use std::collections::BTreeMap;

const SENDER: Address = Address::repeat_byte(0x11);
const GAS_LIMIT: u64 = 1_000_000;

#[derive(Default)]
struct UnavailableStorage {
    reads: usize,
}

impl StorageReader for UnavailableStorage {
    fn sload(&mut self, _key: U256) -> Result<U256, PrecompileHalt> {
        self.reads += 1;
        Err(PrecompileHalt::Other("storage unavailable".into()))
    }
}

impl StakingStorage for UnavailableStorage {
    fn sstore(&mut self, _key: U256, _value: U256) -> Result<(), PrecompileHalt> {
        panic!("zero-value delegation must not write storage");
    }

    fn transfer(
        &mut self,
        _from: Address,
        _to: Address,
        _amount: U256,
    ) -> Result<(), PrecompileHalt> {
        panic!("zero-value delegation must not transfer funds");
    }

    fn emit_log(&mut self, _log: Log) -> Result<(), PrecompileHalt> {
        panic!("zero-value delegation must not emit logs");
    }
}

#[rstest]
fn zero_delegate_does_not_read_storage(#[values(0, 1, u64::MAX)] validator_id: u64) {
    let mut storage = UnavailableStorage::default();
    let result = run_staking_write(
        &delegateCall { validatorId: validator_id }.abi_encode(),
        gas::DELEGATE,
        &mut storage,
        &SENDER,
        U256::ZERO,
    )
    .unwrap();

    assert_eq!(result.result, InstructionResult::Return);
    assert_eq!(result.output.as_ref(), delegateCall::abi_encode_returns(&true));
    assert_eq!(result.gas.remaining(), 0);
    assert_eq!(storage.reads, 0);
}

#[rstest]
fn zero_delegate_still_validates_input(#[values(4, 35)] input_len: usize) {
    let mut storage = UnavailableStorage::default();
    let input = delegateCall { validatorId: 1 }.abi_encode();
    let result =
        run_staking_write(&input[..input_len], gas::DELEGATE, &mut storage, &SENDER, U256::ZERO)
            .unwrap();

    assert_eq!(result.result, InstructionResult::Revert);
    assert!(!result.output.is_empty());
    assert_eq!(storage.reads, 0);
}

#[test]
fn zero_delegate_still_requires_gas() {
    let mut storage = UnavailableStorage::default();
    let result = run_staking_write(
        &delegateCall { validatorId: 1 }.abi_encode(),
        gas::DELEGATE - 1,
        &mut storage,
        &SENDER,
        U256::ZERO,
    )
    .unwrap();

    assert_eq!(result.result, InstructionResult::PrecompileOOG);
    assert_eq!(storage.reads, 0);
}

fn staking_storage(db: &InMemoryDB) -> BTreeMap<U256, U256> {
    db.cache.accounts[&STAKING_ADDRESS]
        .storage
        .iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|(key, value)| (*key, *value))
        .collect()
}

#[rstest]
#[case::zero_id(0, U256::ZERO, true)]
#[case::existing(1, U256::ZERO, true)]
#[case::unknown(u64::MAX, U256::ZERO, true)]
#[case::nonzero_zero_id(0, DUST_THRESHOLD, false)]
#[case::nonzero_unknown(u64::MAX, DUST_THRESHOLD, false)]
#[case::dust_unknown(u64::MAX, U256::from(1), false)]
fn delegate_noop_and_unknown_validator(
    #[case] validator_id: u64,
    #[case] value: U256,
    #[case] success: bool,
    #[values(
        MonadHardfork::MonadEight,
        MonadHardfork::MonadNine,
        MonadHardfork::MonadTen,
        MonadHardfork::MonadNext
    )]
    spec: MonadHardfork,
) {
    let mut db = InMemoryDB::default();
    db.insert_account_info(SENDER, AccountInfo { balance: MON, ..Default::default() });
    db.insert_account_info(
        STAKING_ADDRESS,
        AccountInfo { balance: MON, nonce: 1, ..Default::default() },
    );
    let mut address_flags = [0; 32];
    address_flags[..20].copy_from_slice(SENDER.as_slice());
    db.insert_account_storage(
        STAKING_ADDRESS,
        validator_key(1, validator_offsets::ADDRESS_FLAGS),
        U256::from_be_bytes(address_flags),
    )
    .unwrap();
    let initial_storage = staking_storage(&db);

    let mut context = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
    context.block.basefee = 0;
    let mut evm = context.build_monad();
    let result = evm
        .transact_commit(
            TxEnv::builder()
                .caller(SENDER)
                .to(STAKING_ADDRESS)
                .data(delegateCall { validatorId: validator_id }.abi_encode().into())
                .value(value)
                .gas_limit(GAS_LIMIT)
                .gas_price(1)
                .build_fill(),
        )
        .unwrap();

    if success {
        assert!(result.is_success());
        assert_eq!(result.output().unwrap().as_ref(), delegateCall::abi_encode_returns(&true));
    } else {
        assert!(matches!(result, ExecutionResult::Revert { .. }));
        assert_eq!(result.output().unwrap().as_ref(), b"unknown validator");
    }
    assert!(result.logs().is_empty());
    let db = evm.ctx_ref().db();
    assert_eq!(staking_storage(db), initial_storage);
    assert_eq!(db.cache.accounts[&STAKING_ADDRESS].info.balance, MON);
    assert_eq!(db.cache.accounts[&SENDER].info.balance, MON - U256::from(GAS_LIMIT));
    assert_eq!(db.cache.accounts[&SENDER].info.nonce, 1);
}
