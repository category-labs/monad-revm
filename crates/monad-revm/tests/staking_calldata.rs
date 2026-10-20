//! Staking calldata consumption across native and custom storage dispatch.

use alloy_sol_types::SolCall;
use monad_revm::{
    monad_context_with_db,
    staking::{
        abi::gas,
        constants::{DUST_THRESHOLD, SYSTEM_ADDRESS},
        interface::IMonadStaking::*,
        run_staking_precompile, run_staking_with_reader,
        write::{run_staking_write, StakingStorage},
        StorageReader, STAKING_ADDRESS,
    },
};
use revm::{
    context_interface::{ContextTr, JournalTr},
    database::InMemoryDB,
    interpreter::{
        CallInput, CallInputs, CallScheme, CallValue, InstructionResult, InterpreterResult,
    },
    precompile::PrecompileHalt,
    primitives::{Address, Bytes, Log, B256, U256},
    state::Bytecode,
};
use rstest::rstest;
use std::collections::BTreeMap;

const GAS_LIMIT: u64 = 1_000_000;
const DELEGATOR: Address = Address::repeat_byte(0x11);

#[derive(Clone, Copy, Debug)]
enum Route {
    Native,
    Adapter,
}

#[derive(Default)]
struct Storage {
    slots: BTreeMap<U256, U256>,
    accesses: usize,
}

impl StorageReader for Storage {
    fn sload(&mut self, key: U256) -> Result<U256, PrecompileHalt> {
        self.accesses += 1;
        Ok(self.slots.get(&key).copied().unwrap_or_default())
    }
}

impl StakingStorage for Storage {
    fn sstore(&mut self, key: U256, value: U256) -> Result<(), PrecompileHalt> {
        self.accesses += 1;
        self.slots.insert(key, value);
        Ok(())
    }

    fn transfer(
        &mut self,
        _from: Address,
        _to: Address,
        _amount: U256,
    ) -> Result<(), PrecompileHalt> {
        self.accesses += 1;
        Ok(())
    }

    fn emit_log(&mut self, _log: Log) -> Result<(), PrecompileHalt> {
        self.accesses += 1;
        Ok(())
    }
}

fn run(
    route: Route,
    reader: bool,
    input: &[u8],
    gas_limit: u64,
    caller: Address,
    value: U256,
) -> (InterpreterResult, usize) {
    match route {
        Route::Native => {
            let mut context = monad_context_with_db(InMemoryDB::default());
            context.journal_mut().load_account(STAKING_ADDRESS).unwrap();
            let inputs = CallInputs {
                input: CallInput::Bytes(Bytes::copy_from_slice(input)),
                return_memory_offset: 0..0,
                gas_limit,
                reservoir: 0,
                bytecode_address: STAKING_ADDRESS,
                known_bytecode: (B256::ZERO, Bytecode::default()),
                target_address: STAKING_ADDRESS,
                caller,
                value: CallValue::Transfer(value),
                scheme: CallScheme::Call,
                is_static: false,
                charged_new_account_state_gas: false,
            };
            let result = run_staking_precompile(&mut context, &inputs).unwrap().unwrap();
            let accessed_slots = context.journal_mut().evm_state()[&STAKING_ADDRESS].storage.len();
            (result, accessed_slots)
        }
        Route::Adapter => {
            let mut storage = Storage::default();
            let result = if reader {
                run_staking_with_reader(input, gas_limit, &mut storage, value)
            } else {
                run_staking_write(input, gas_limit, &mut storage, &caller, value)
            }
            .unwrap();
            (result, storage.accesses)
        }
    }
}

fn assert_revert(result: &InterpreterResult, message: &str) {
    assert_eq!(result.result, InstructionResult::Revert);
    assert_eq!(result.output.as_ref(), message.as_bytes());
    assert_eq!(result.gas.remaining(), 0);
}

#[rstest]
#[case::validator(getValidatorCall { validatorId: 1 }.abi_encode(), gas::GET_VALIDATOR, true, &[8], None)]
#[case::delegator(getDelegatorCall { validatorId: 1, delegator: DELEGATOR }.abi_encode(), gas::GET_DELEGATOR, true, &[8, 20], None)]
#[case::withdrawal(getWithdrawalRequestCall { validatorId: 1, delegator: DELEGATOR, withdrawId: 7 }.abi_encode(), gas::GET_WITHDRAWAL_REQUEST, true, &[8, 20, 1], None)]
#[case::consensus(getConsensusValidatorSetCall { startIndex: 0 }.abi_encode(), gas::GET_CONSENSUS_VALIDATOR_SET, true, &[4], None)]
#[case::snapshot(getSnapshotValidatorSetCall { startIndex: 0 }.abi_encode(), gas::GET_SNAPSHOT_VALIDATOR_SET, true, &[4], None)]
#[case::execution(getExecutionValidatorSetCall { startIndex: 0 }.abi_encode(), gas::GET_EXECUTION_VALIDATOR_SET, true, &[4], None)]
#[case::delegations(getDelegationsCall { delegator: DELEGATOR, startValId: 1 }.abi_encode(), gas::GET_DELEGATIONS, true, &[20, 8], None)]
#[case::delegators(getDelegatorsCall { validatorId: 1, startDelegator: DELEGATOR }.abi_encode(), gas::GET_DELEGATORS, true, &[8, 20], None)]
#[case::delegate(delegateCall { validatorId: 1 }.abi_encode(), gas::DELEGATE, false, &[8], Some("unknown validator"))]
#[case::undelegate(undelegateCall { validatorId: 1, amount: U256::ZERO, withdrawId: 7 }.abi_encode(), gas::UNDELEGATE, false, &[8, 32, 1], None)]
#[case::withdraw(withdrawCall { validatorId: 1, withdrawId: 7 }.abi_encode(), gas::WITHDRAW, false, &[8, 1], Some("unknown withdrawal id"))]
#[case::compound(compoundCall { validatorId: 1 }.abi_encode(), gas::COMPOUND, false, &[8], None)]
#[case::claim(claimRewardsCall { validatorId: 1 }.abi_encode(), gas::CLAIM_REWARDS, false, &[8], None)]
#[case::commission(changeCommissionCall { validatorId: 1, commission: U256::ZERO }.abi_encode(), gas::CHANGE_COMMISSION, false, &[8, 32], Some("unknown validator"))]
#[case::reward(externalRewardCall { validatorId: 1 }.abi_encode(), gas::EXTERNAL_REWARD, false, &[8], Some("unknown validator"))]
#[case::mutating_delegator(getDelegatorCall { validatorId: 1, delegator: DELEGATOR }.abi_encode(), gas::GET_DELEGATOR, false, &[8, 20], None)]
#[case::epoch_change(syscallOnEpochChangeCall { epoch: 1 }.abi_encode(), gas::SYSCALL_ON_EPOCH_CHANGE, false, &[8], None)]
fn fixed_arguments_consume_exact_words(
    #[case] input: Vec<u8>,
    #[case] cost: u64,
    #[case] reader: bool,
    #[case] widths: &[usize],
    #[case] error: Option<&str>,
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    let value =
        if input.starts_with(&delegateCall::SELECTOR) { DUST_THRESHOLD } else { U256::ZERO };
    let (exact, _) = run(route, reader, &input, GAS_LIMIT, SYSTEM_ADDRESS, value);
    if let Some(error) = error {
        assert_revert(&exact, error);
    } else {
        assert_eq!(exact.result, InstructionResult::Return);
        assert_eq!(exact.gas.total_gas_spent(), cost);
    }

    let mut dirty = input.clone();
    for (word, width) in widths.iter().enumerate() {
        dirty[4 + word * 32..4 + word * 32 + 32 - width].fill(0xff);
    }
    let (dirty_result, _) = run(route, reader, &dirty, GAS_LIMIT, SYSTEM_ADDRESS, value);
    assert_eq!(dirty_result.result, exact.result);
    assert_eq!(dirty_result.output, exact.output);
    assert_eq!(dirty_result.gas.total_gas_spent(), exact.gas.total_gas_spent());

    for length in 4..input.len() {
        let (result, accesses) =
            run(route, reader, &input[..length], GAS_LIMIT, SYSTEM_ADDRESS, value);
        assert_revert(&result, "input too short");
        assert_eq!(accesses, 0, "truncated calldata must fail before storage access");
    }
    for suffix in [&[0xff][..], &[0; 32][..], &[0xff; 33][..]] {
        let mut extra = input.clone();
        extra.extend_from_slice(suffix);
        let (result, accesses) = run(route, reader, &extra, GAS_LIMIT, SYSTEM_ADDRESS, value);
        assert_revert(&result, "invalid input");
        assert_eq!(accesses, 0, "trailing calldata must fail before storage access");

        let (out_of_gas, accesses) = run(route, reader, &extra, cost - 1, SYSTEM_ADDRESS, value);
        assert_eq!(out_of_gas.result, InstructionResult::PrecompileOOG);
        assert_eq!(accesses, 0);
    }
}

#[rstest]
#[case::epoch(getEpochCall {}.abi_encode(), gas::GET_EPOCH)]
#[case::proposer(getProposerValIdCall {}.abi_encode(), gas::GET_PROPOSER_VAL_ID)]
fn parameterless_getters_ignore_argument_bytes(
    #[case] input: Vec<u8>,
    #[case] cost: u64,
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    let (exact, _) = run(route, true, &input, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
    assert_eq!(exact.result, InstructionResult::Return);
    for suffix in [&[0xff][..], &[0xff; 32][..], &[0xff; 65][..]] {
        let mut extra = input.clone();
        extra.extend_from_slice(suffix);
        let (result, _) = run(route, true, &extra, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        assert_eq!(result.result, InstructionResult::Return);
        assert_eq!(result.output, exact.output);
        assert_eq!(result.gas.total_gas_spent(), cost);

        let (nonpayable, accesses) =
            run(route, true, &extra, GAS_LIMIT, SYSTEM_ADDRESS, U256::from(1));
        assert_revert(&nonpayable, "value non-zero");
        assert_eq!(accesses, 0);
    }
}

#[rstest]
fn snapshot_rejects_argument_bytes(#[values(Route::Native, Route::Adapter)] route: Route) {
    let input = syscallSnapshotCall {}.abi_encode();
    let (result, _) = run(route, false, &input, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
    assert_eq!(result.result, InstructionResult::Return);
    for suffix in [&[0xff][..], &[0; 32][..]] {
        let mut extra = input.clone();
        extra.extend_from_slice(suffix);
        let (result, accesses) = run(route, false, &extra, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        assert_revert(&result, "invalid input");
        assert_eq!(accesses, 0);
    }
}

fn registration_input() -> Vec<u8> {
    addValidatorCall {
        payload: vec![0; 165].into(),
        signedSecpMessage: vec![0; 64].into(),
        signedBlsMessage: vec![0; 96].into(),
    }
    .abi_encode()
}

#[rstest]
fn registration_ignores_offsets_and_padding(#[values(Route::Native, Route::Adapter)] route: Route) {
    let input = registration_input();
    assert_eq!(input.len(), 4 + 544);
    // A decoded zero-stake registration reaches the stake check before any state access.
    let (exact, accesses) = run(route, false, &input, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
    assert_revert(&exact, "insufficient stake");
    assert_eq!(accesses, 0);

    for offsets in [[0; 96], [0xff; 96], [1; 96]] {
        let mut modified = input.clone();
        modified[4..100].copy_from_slice(&offsets);
        modified[297..324].fill(0xff);
        let (result, accesses) =
            run(route, false, &modified, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        assert_revert(&result, "insufficient stake");
        assert_eq!(accesses, 0);
    }
}

#[rstest]
fn registration_requires_complete_padded_tails(
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    let input = registration_input();
    for length in 4..input.len() {
        let (result, accesses) =
            run(route, false, &input[..length], GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        assert_revert(&result, "input too short");
        assert_eq!(accesses, 0);
    }
    for suffix in [&[0xff][..], &[0; 32][..]] {
        let mut extra = input.clone();
        extra.extend_from_slice(suffix);
        let (result, accesses) = run(route, false, &extra, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        assert_revert(&result, "invalid input");
        assert_eq!(accesses, 0);
    }
}

#[rstest]
#[case::payload(100, 165)]
#[case::secp_signature(324, 64)]
#[case::bls_signature(420, 96)]
fn registration_requires_exact_tail_lengths(
    #[case] offset: usize,
    #[case] length: u64,
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    for invalid in [U256::ZERO, U256::from(length - 1), U256::from(length + 1), U256::MAX] {
        let mut input = registration_input();
        input[offset..offset + 32].copy_from_slice(&invalid.to_be_bytes::<32>());
        for end in [offset + 32, input.len()] {
            let (result, accesses) =
                run(route, false, &input[..end], GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
            assert_revert(&result, "length mismatch");
            assert_eq!(accesses, 0);
        }
    }
}

#[rstest]
fn syscall_reward_allows_only_the_complete_synthetic_extension(
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    let input = syscallRewardCall { blockAuthor: DELEGATOR }.abi_encode();
    for length in 4..=101 {
        let mut modified = input.clone();
        modified.resize(length, 0xff);
        let (result, accesses) =
            run(route, false, &modified, GAS_LIMIT, SYSTEM_ADDRESS, U256::ZERO);
        let message = match length {
            4..36 => "input too short",
            36 | 68 => "not in validator set",
            _ => "invalid input",
        };
        assert_revert(&result, message);
        if length == 36 || length == 68 {
            assert_ne!(accesses, 0);
        } else {
            assert_eq!(accesses, 0);
        }
    }
}

#[rstest]
#[case::getter(getValidatorCall::SELECTOR.to_vec(), true, gas::GET_VALIDATOR)]
#[case::write(changeCommissionCall::SELECTOR.to_vec(), false, gas::CHANGE_COMMISSION)]
#[case::snapshot(syscallSnapshotCall {}.abi_encode().into_iter().chain([0xff]).collect(), false, gas::SYSCALL_SNAPSHOT)]
#[case::epoch_change(syscallOnEpochChangeCall::SELECTOR.to_vec(), false, gas::SYSCALL_ON_EPOCH_CHANGE)]
fn payability_precedes_decoding(
    #[case] input: Vec<u8>,
    #[case] reader: bool,
    #[case] cost: u64,
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    for gas_limit in [cost - 1, GAS_LIMIT] {
        let (result, accesses) =
            run(route, reader, &input, gas_limit, SYSTEM_ADDRESS, U256::from(1));
        assert_revert(&result, "value non-zero");
        assert_eq!(accesses, 0);
    }
}

#[rstest]
#[case::reward(syscallRewardCall::SELECTOR.to_vec(), gas::SYSCALL_REWARD)]
#[case::snapshot(syscallSnapshotCall {}.abi_encode().into_iter().chain([0xff]).collect(), gas::SYSCALL_SNAPSHOT)]
#[case::epoch_change(syscallOnEpochChangeCall::SELECTOR.to_vec(), gas::SYSCALL_ON_EPOCH_CHANGE)]
fn syscall_gas_and_authorization_precede_decoding(
    #[case] input: Vec<u8>,
    #[case] cost: u64,
    #[values(Route::Native, Route::Adapter)] route: Route,
) {
    let (result, accesses) = run(route, false, &input, cost - 1, DELEGATOR, U256::ZERO);
    assert_eq!(result.result, InstructionResult::PrecompileOOG);
    assert_eq!(accesses, 0);
    let (result, accesses) = run(route, false, &input, GAS_LIMIT, DELEGATOR, U256::ZERO);
    assert_revert(&result, "Unauthorized: not system address");
    assert_eq!(accesses, 0);
}
