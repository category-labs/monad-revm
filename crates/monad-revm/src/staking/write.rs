//! State-mutating functions for the staking precompile.
//!
//! Implements all 8 user-callable write functions and 3 syscalls.
//! Uses the [`StakingStorage`] trait for storage access, enabling both
//! direct REVM (`ContextTr`) and Foundry (`EvmInternals`) integration.

use super::{
    abi::gas,
    constants::{
        ACTIVE_VALIDATOR_STAKE, ACTIVE_VALSET_SIZE, DUST_THRESHOLD, MAX_COMMISSION,
        MAX_EXTERNAL_REWARD, MIN_AUTH_ADDRESS_STAKE, MIN_EXTERNAL_REWARD, MON, SYSTEM_ADDRESS,
        UNIT_BIAS, WITHDRAWAL_DELAY,
    },
    interface::IMonadStaking::*,
    storage::{
        accumulator_key, bitset_bucket_key, consensus_view_key, delegator_key, delegator_offsets,
        global_slots, snapshot_view_key, val_id_secp_key, validator_key, validator_offsets,
        valset_slots, withdrawal_key, withdrawal_offsets, STAKING_ADDRESS,
    },
    types::{validator_flags, Delegator, ListNode, RefCountedAccumulator, Validator},
    StorageReader,
};
use alloy_sol_types::{SolCall, SolEvent};
use revm::{
    interpreter::{Gas, InstructionResult, InterpreterResult},
    precompile::PrecompileError,
    primitives::{Address, Bytes, Log, LogData, B256, U256},
};

// ═══════════════════════════════════════════════════════════════════════════════
// StakingStorage Trait
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended storage trait for state-mutating staking operations.
///
/// Extends [`StorageReader`] with write capabilities needed by delegate,
/// undelegate, and other state-changing functions.
pub trait StakingStorage: StorageReader {
    /// Write a U256 value to storage at the given key.
    fn sstore(&mut self, key: U256, value: U256) -> Result<(), PrecompileError>;

    /// Transfer balance from one address to another.
    fn transfer(&mut self, from: Address, to: Address, amount: U256)
        -> Result<(), PrecompileError>;

    /// Emit a log entry.
    fn emit_log(&mut self, log: Log) -> Result<(), PrecompileError>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Storage Write Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn write_storage_u256<S: StakingStorage>(
    s: &mut S,
    key: U256,
    value: U256,
) -> Result<(), PrecompileError> {
    s.sstore(key, value)
}

/// Write a u64 value to storage (left-aligned, big-endian).
fn write_storage_u64<S: StakingStorage>(
    s: &mut S,
    key: U256,
    value: u64,
) -> Result<(), PrecompileError> {
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&value.to_be_bytes());
    s.sstore(key, U256::from_be_bytes(bytes))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Storage Read Helpers (using StorageReader trait)
// ═══════════════════════════════════════════════════════════════════════════════

fn read_u256<S: StorageReader>(s: &mut S, key: U256) -> Result<U256, PrecompileError> {
    s.sload(key)
}

fn read_u64<S: StorageReader>(s: &mut S, key: U256) -> Result<u64, PrecompileError> {
    let value = s.sload(key)?;
    let bytes = value.to_be_bytes::<32>();
    Ok(u64::from_be_bytes(bytes[0..8].try_into().unwrap()))
}

fn read_epoch<S: StorageReader>(s: &mut S) -> Result<u64, PrecompileError> {
    read_u64(s, global_slots::EPOCH)
}

fn read_in_boundary<S: StorageReader>(s: &mut S) -> Result<bool, PrecompileError> {
    let raw = read_u256(s, global_slots::IN_BOUNDARY)?;
    Ok(raw != U256::ZERO)
}

fn read_validator<S: StorageReader>(s: &mut S, val_id: u64) -> Result<Validator, PrecompileError> {
    let stake = read_u256(s, validator_key(val_id, validator_offsets::STAKE))?;
    let accumulated_reward_per_token =
        read_u256(s, validator_key(val_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN))?;
    let commission = read_u256(s, validator_key(val_id, validator_offsets::COMMISSION))?;

    let keys_slot_0 =
        read_u256(s, validator_key(val_id, validator_offsets::KEYS))?.to_be_bytes::<32>();
    let keys_slot_1 =
        read_u256(s, validator_key(val_id, validator_offsets::KEYS + 1))?.to_be_bytes::<32>();
    let keys_slot_2 =
        read_u256(s, validator_key(val_id, validator_offsets::KEYS + 2))?.to_be_bytes::<32>();

    let mut keys_concat = [0u8; 96];
    keys_concat[0..32].copy_from_slice(&keys_slot_0);
    keys_concat[32..64].copy_from_slice(&keys_slot_1);
    keys_concat[64..96].copy_from_slice(&keys_slot_2);

    let mut secp_pubkey = [0u8; 33];
    let mut bls_pubkey = [0u8; 48];
    secp_pubkey.copy_from_slice(&keys_concat[0..33]);
    bls_pubkey.copy_from_slice(&keys_concat[33..81]);

    let address_flags_raw =
        read_u256(s, validator_key(val_id, validator_offsets::ADDRESS_FLAGS))?.to_be_bytes::<32>();
    let auth_address = Address::from_slice(&address_flags_raw[0..20]);
    let flags = u64::from_be_bytes(address_flags_raw[20..28].try_into().unwrap());

    let unclaimed_rewards =
        read_u256(s, validator_key(val_id, validator_offsets::UNCLAIMED_REWARDS))?;

    Ok(Validator {
        stake,
        accumulated_reward_per_token,
        commission,
        secp_pubkey,
        bls_pubkey,
        auth_address,
        flags,
        unclaimed_rewards,
    })
}

fn read_delegator<S: StorageReader>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
) -> Result<Delegator, PrecompileError> {
    let stake = read_u256(s, delegator_key(val_id, addr, delegator_offsets::STAKE))?;
    let accumulated_reward_per_token =
        read_u256(s, delegator_key(val_id, addr, delegator_offsets::ACCUMULATED_REWARD_PER_TOKEN))?;
    let rewards = read_u256(s, delegator_key(val_id, addr, delegator_offsets::REWARDS))?;
    let delta_stake = read_u256(s, delegator_key(val_id, addr, delegator_offsets::DELTA_STAKE))?;
    let next_delta_stake =
        read_u256(s, delegator_key(val_id, addr, delegator_offsets::NEXT_DELTA_STAKE))?;
    let epochs_raw =
        read_u256(s, delegator_key(val_id, addr, delegator_offsets::EPOCHS))?.to_be_bytes::<32>();
    let delta_epoch = u64::from_be_bytes(epochs_raw[0..8].try_into().unwrap());
    let next_delta_epoch = u64::from_be_bytes(epochs_raw[8..16].try_into().unwrap());

    Ok(Delegator {
        stake,
        accumulated_reward_per_token,
        rewards,
        delta_stake,
        next_delta_stake,
        delta_epoch,
        next_delta_epoch,
    })
}

fn read_list_node<S: StorageReader>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
) -> Result<ListNode, PrecompileError> {
    let slot6 = read_u256(s, delegator_key(val_id, addr, delegator_offsets::LIST_NODE))?
        .to_be_bytes::<32>();
    let slot7 = read_u256(s, delegator_key(val_id, addr, delegator_offsets::LIST_NODE + 1))?
        .to_be_bytes::<32>();
    Ok(ListNode::from_slots(slot6, slot7))
}

fn read_accumulator<S: StorageReader>(
    s: &mut S,
    epoch: u64,
    val_id: u64,
) -> Result<RefCountedAccumulator, PrecompileError> {
    let value = read_u256(s, accumulator_key(epoch, val_id, 0))?;
    // Refcount is stored as u256 (right-aligned standard integer encoding)
    let refcount_u256 = read_u256(s, accumulator_key(epoch, val_id, 1))?;
    let refcount = refcount_u256.as_limbs()[0];
    Ok(RefCountedAccumulator { value, refcount })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Struct Write Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn write_validator_stake<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    stake: U256,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, validator_key(val_id, validator_offsets::STAKE), stake)
}

fn write_validator_acc<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    acc: U256,
) -> Result<(), PrecompileError> {
    write_storage_u256(
        s,
        validator_key(val_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN),
        acc,
    )
}

fn write_validator_commission<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    commission: U256,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, validator_key(val_id, validator_offsets::COMMISSION), commission)
}

fn write_validator_unclaimed_rewards<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    rewards: U256,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, validator_key(val_id, validator_offsets::UNCLAIMED_REWARDS), rewards)
}

fn write_validator_flags<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    val: &Validator,
    new_flags: u64,
) -> Result<(), PrecompileError> {
    // Slot 6 packs auth_address (20 bytes) + flags (8 bytes) + padding (4 bytes)
    let mut slot6 = [0u8; 32];
    slot6[0..20].copy_from_slice(val.auth_address.as_slice());
    slot6[20..28].copy_from_slice(&new_flags.to_be_bytes());
    write_storage_u256(
        s,
        validator_key(val_id, validator_offsets::ADDRESS_FLAGS),
        U256::from_be_bytes(slot6),
    )
}

fn write_validator_full<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    val: &Validator,
) -> Result<(), PrecompileError> {
    write_validator_stake(s, val_id, val.stake)?;
    write_validator_acc(s, val_id, val.accumulated_reward_per_token)?;
    write_validator_commission(s, val_id, val.commission)?;

    // Keys: 3 slots (secp33 + bls48 = 81 bytes)
    let mut keys_concat = [0u8; 96];
    keys_concat[0..33].copy_from_slice(&val.secp_pubkey);
    keys_concat[33..81].copy_from_slice(&val.bls_pubkey);
    let slot0: [u8; 32] = keys_concat[0..32].try_into().unwrap();
    let slot1: [u8; 32] = keys_concat[32..64].try_into().unwrap();
    let slot2: [u8; 32] = keys_concat[64..96].try_into().unwrap();
    write_storage_u256(
        s,
        validator_key(val_id, validator_offsets::KEYS),
        U256::from_be_bytes(slot0),
    )?;
    write_storage_u256(
        s,
        validator_key(val_id, validator_offsets::KEYS + 1),
        U256::from_be_bytes(slot1),
    )?;
    write_storage_u256(
        s,
        validator_key(val_id, validator_offsets::KEYS + 2),
        U256::from_be_bytes(slot2),
    )?;

    // Address + flags packed
    write_validator_flags(s, val_id, val, val.flags)?;
    write_validator_unclaimed_rewards(s, val_id, val.unclaimed_rewards)?;
    Ok(())
}

fn write_delegator<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
    del: &Delegator,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, delegator_key(val_id, addr, delegator_offsets::STAKE), del.stake)?;
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::ACCUMULATED_REWARD_PER_TOKEN),
        del.accumulated_reward_per_token,
    )?;
    write_storage_u256(s, delegator_key(val_id, addr, delegator_offsets::REWARDS), del.rewards)?;
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::DELTA_STAKE),
        del.delta_stake,
    )?;
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::NEXT_DELTA_STAKE),
        del.next_delta_stake,
    )?;
    // Pack delta_epoch + next_delta_epoch into one slot
    let mut epochs = [0u8; 32];
    epochs[0..8].copy_from_slice(&del.delta_epoch.to_be_bytes());
    epochs[8..16].copy_from_slice(&del.next_delta_epoch.to_be_bytes());
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::EPOCHS),
        U256::from_be_bytes(epochs),
    )?;
    Ok(())
}

fn write_list_node<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
    node: &ListNode,
) -> Result<(), PrecompileError> {
    let (slot6, slot7) = node.to_slots();
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::LIST_NODE),
        U256::from_be_bytes(slot6),
    )?;
    write_storage_u256(
        s,
        delegator_key(val_id, addr, delegator_offsets::LIST_NODE + 1),
        U256::from_be_bytes(slot7),
    )?;
    Ok(())
}

fn write_withdrawal_request<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
    wid: u8,
    amount: U256,
    acc: U256,
    epoch: u64,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, withdrawal_key(val_id, addr, wid, withdrawal_offsets::AMOUNT), amount)?;
    write_storage_u256(s, withdrawal_key(val_id, addr, wid, withdrawal_offsets::ACCUMULATOR), acc)?;
    write_storage_u64(s, withdrawal_key(val_id, addr, wid, withdrawal_offsets::EPOCH), epoch)?;
    Ok(())
}

fn clear_withdrawal_request<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
    wid: u8,
) -> Result<(), PrecompileError> {
    write_storage_u256(
        s,
        withdrawal_key(val_id, addr, wid, withdrawal_offsets::AMOUNT),
        U256::ZERO,
    )?;
    write_storage_u256(
        s,
        withdrawal_key(val_id, addr, wid, withdrawal_offsets::ACCUMULATOR),
        U256::ZERO,
    )?;
    write_storage_u256(
        s,
        withdrawal_key(val_id, addr, wid, withdrawal_offsets::EPOCH),
        U256::ZERO,
    )?;
    Ok(())
}

fn write_accumulator<S: StakingStorage>(
    s: &mut S,
    epoch: u64,
    val_id: u64,
    acc: &RefCountedAccumulator,
) -> Result<(), PrecompileError> {
    write_storage_u256(s, accumulator_key(epoch, val_id, 0), acc.value)?;
    // Refcount stored as u256 (right-aligned standard integer encoding)
    write_storage_u256(s, accumulator_key(epoch, val_id, 1), U256::from(acc.refcount))?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core Business Logic
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the activation epoch for new delegations.
///
/// Returns `epoch + 1` normally, or `epoch + 2` if in the epoch delay period.
fn get_activation_epoch<S: StorageReader>(s: &mut S) -> Result<u64, PrecompileError> {
    let epoch = read_epoch(s)?;
    let in_boundary = read_in_boundary(s)?;
    Ok(if in_boundary { epoch + 2 } else { epoch + 1 })
}

/// Check if an epoch is active (has passed).
const fn is_epoch_active(current_epoch: u64, active_epoch: u64) -> bool {
    active_epoch != 0 && active_epoch <= current_epoch
}

/// Calculate rewards using the accumulator formula.
///
/// `reward = (stake * (epoch_acc - last_acc)) / UNIT_BIAS`
fn calculate_rewards(stake: U256, epoch_acc: U256, last_acc: U256) -> U256 {
    if stake.is_zero() || epoch_acc <= last_acc {
        return U256::ZERO;
    }
    let diff = epoch_acc.saturating_sub(last_acc);
    stake.saturating_mul(diff) / UNIT_BIAS
}

/// Increment the accumulator refcount for a validator at the activation epoch.
///
/// Snapshots the current validator accumulator value and increments the refcount.
fn increment_accumulator_refcount<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
) -> Result<(), PrecompileError> {
    let epoch = get_activation_epoch(s)?;
    let mut acc = read_accumulator(s, epoch, val_id)?;
    acc.refcount += 1;
    acc.value =
        read_u256(s, validator_key(val_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN))?;
    write_accumulator(s, epoch, val_id, &acc)
}

/// Decrement the accumulator refcount and return the snapshotted value.
///
/// If refcount reaches 0, clears the accumulator storage.
fn decrement_accumulator_refcount<S: StakingStorage>(
    s: &mut S,
    epoch: u64,
    val_id: u64,
) -> Result<U256, PrecompileError> {
    let mut acc = read_accumulator(s, epoch, val_id)?;
    let value = acc.value;
    if acc.refcount == 0 {
        return Ok(U256::ZERO);
    }
    acc.refcount -= 1;
    if acc.refcount == 0 {
        // Clear storage
        write_storage_u256(s, accumulator_key(epoch, val_id, 0), U256::ZERO)?;
        write_storage_u256(s, accumulator_key(epoch, val_id, 1), U256::ZERO)?;
    } else {
        write_accumulator(s, epoch, val_id, &acc)?;
    }
    Ok(value)
}

/// Apply compound: calculate rewards from delta track activation and fold into active stake.
fn apply_compound<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    del: &mut Delegator,
) -> Result<U256, PrecompileError> {
    let epoch_acc = decrement_accumulator_refcount(s, del.delta_epoch, val_id)?;
    let rewards = calculate_rewards(del.stake, epoch_acc, del.accumulated_reward_per_token);
    del.accumulated_reward_per_token = epoch_acc;

    // Compound: active_stake += delta_stake
    del.stake = del.stake.saturating_add(del.delta_stake);

    // Promote next_delta → delta
    del.delta_stake = del.next_delta_stake;
    del.next_delta_stake = U256::ZERO;
    del.delta_epoch = del.next_delta_epoch;
    del.next_delta_epoch = 0;

    Ok(rewards)
}

/// Pull delegator state up to date.
///
/// Promotes pending delta stakes if their activation epochs have passed,
/// and calculates accumulated rewards.
fn pull_delegator_up_to_date<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    addr: &Address,
) -> Result<Delegator, PrecompileError> {
    let mut del = read_delegator(s, val_id, addr)?;
    let current_epoch = read_epoch(s)?;

    // Can promote next_delta → delta?
    let can_promote = del.delta_epoch == 0 && del.next_delta_epoch <= current_epoch + 1;
    if can_promote && del.next_delta_epoch != 0 {
        del.delta_stake = del.next_delta_stake;
        del.next_delta_stake = U256::ZERO;
        del.delta_epoch = del.next_delta_epoch;
        del.next_delta_epoch = 0;
    }

    // Track validator unclaimed_rewards for reward_invariant
    let mut unclaimed_rewards =
        read_u256(s, validator_key(val_id, validator_offsets::UNCLAIMED_REWARDS))?;

    // Check which delta tracks can be compounded.
    // Both checks use the state BEFORE any compounding.
    let can_compound = is_epoch_active(current_epoch, del.delta_epoch) && del.delta_epoch != 0;
    let can_compound_boundary =
        is_epoch_active(current_epoch, del.next_delta_epoch) && del.next_delta_epoch != 0;

    // Compound boundary track first.
    // When both tracks are active, apply_compound is called twice:
    //   1st: uses delta_epoch, promotes next_delta → delta
    //   2nd: uses the newly promoted delta (was next_delta)
    if can_compound_boundary {
        let rewards = apply_compound(s, val_id, &mut del)?;
        // reward_invariant: check solvency and deduct from unclaimed
        if unclaimed_rewards < rewards {
            return Err(PrecompileError::Other("solvency error".into()));
        }
        unclaimed_rewards = unclaimed_rewards.saturating_sub(rewards);
        del.rewards = del.rewards.saturating_add(rewards);
    }

    // Compound main delta track
    if can_compound {
        let rewards = apply_compound(s, val_id, &mut del)?;
        // reward_invariant: check solvency and deduct from unclaimed
        if unclaimed_rewards < rewards {
            return Err(PrecompileError::Other("solvency error".into()));
        }
        unclaimed_rewards = unclaimed_rewards.saturating_sub(rewards);
        del.rewards = del.rewards.saturating_add(rewards);
    }

    // Accrue rewards for active stake
    if !del.stake.is_zero() {
        let val_acc =
            read_u256(s, validator_key(val_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN))?;
        let rewards = calculate_rewards(del.stake, val_acc, del.accumulated_reward_per_token);
        // reward_invariant: check solvency and deduct from unclaimed
        if unclaimed_rewards < rewards {
            return Err(PrecompileError::Other("solvency error".into()));
        }
        unclaimed_rewards = unclaimed_rewards.saturating_sub(rewards);
        del.accumulated_reward_per_token = val_acc;
        del.rewards = del.rewards.saturating_add(rewards);
    }

    // Write back updated unclaimed_rewards
    write_validator_unclaimed_rewards(s, val_id, unclaimed_rewards)?;
    write_delegator(s, val_id, addr, &del)?;
    Ok(del)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linked List Mutation
// ═══════════════════════════════════════════════════════════════════════════════

/// Insert a delegation into both linked lists.
///
/// - Validator → Delegator list (anext/aprev pointers)
/// - Delegator → Validator list (inext/iprev pointers)
fn linked_list_insert<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    delegator: &Address,
) -> Result<(), PrecompileError> {
    // Insert delegator into validator's list (key=val_id, ptr=delegator address)
    linked_list_insert_address(s, val_id, delegator)?;
    // Insert validator into delegator's list (key=delegator, ptr=val_id)
    linked_list_insert_val_id(s, val_id, delegator)?;
    Ok(())
}

/// Insert an address into the validator's delegator list.
fn linked_list_insert_address<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    ptr: &Address,
) -> Result<(), PrecompileError> {
    if *ptr == Address::ZERO || *ptr == ListNode::SENTINEL_ADDRESS {
        return Err(PrecompileError::Other("invalid input".into()));
    }

    let mut this_node = read_list_node(s, val_id, ptr)?;
    // If aprev != empty, already in list
    if this_node.aprev != Address::ZERO {
        return Ok(());
    }

    let mut sentinel = read_list_node(s, val_id, &ListNode::SENTINEL_ADDRESS)?;
    let next_ptr = sentinel.anext;

    if next_ptr != Address::ZERO {
        let mut next_node = read_list_node(s, val_id, &next_ptr)?;
        next_node.aprev = *ptr;
        write_list_node(s, val_id, &next_ptr, &next_node)?;
    }

    this_node.aprev = ListNode::SENTINEL_ADDRESS;
    this_node.anext = next_ptr;
    sentinel.anext = *ptr;

    write_list_node(s, val_id, ptr, &this_node)?;
    write_list_node(s, val_id, &ListNode::SENTINEL_ADDRESS, &sentinel)?;
    Ok(())
}

/// Insert a validator ID into the delegator's validator list.
fn linked_list_insert_val_id<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    delegator: &Address,
) -> Result<(), PrecompileError> {
    if val_id == 0 || val_id == ListNode::SENTINEL_VAL_ID {
        return Err(PrecompileError::Other("invalid input".into()));
    }

    let mut this_node = read_list_node(s, val_id, delegator)?;
    // If iprev != empty, already in list
    if this_node.iprev != 0 {
        return Ok(());
    }

    let mut sentinel = read_list_node(s, ListNode::SENTINEL_VAL_ID, delegator)?;
    let next_ptr = sentinel.inext;

    if next_ptr != 0 {
        let mut next_node = read_list_node(s, next_ptr, delegator)?;
        next_node.iprev = val_id;
        write_list_node(s, next_ptr, delegator, &next_node)?;
    }

    this_node.iprev = ListNode::SENTINEL_VAL_ID;
    this_node.inext = next_ptr;
    sentinel.inext = val_id;

    write_list_node(s, val_id, delegator, &this_node)?;
    write_list_node(s, ListNode::SENTINEL_VAL_ID, delegator, &sentinel)?;
    Ok(())
}

/// Remove a delegation from both linked lists.
fn linked_list_remove<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    delegator: &Address,
) -> Result<(), PrecompileError> {
    linked_list_remove_address(s, val_id, delegator)?;
    linked_list_remove_val_id(s, val_id, delegator)?;
    Ok(())
}

/// Remove an address from the validator's delegator list.
fn linked_list_remove_address<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    ptr: &Address,
) -> Result<(), PrecompileError> {
    let mut this_node = read_list_node(s, val_id, ptr)?;
    // If aprev == empty, not in list
    if this_node.aprev == Address::ZERO {
        return Ok(());
    }

    let prev_ptr = this_node.aprev;
    let next_ptr = this_node.anext;

    let mut prev_node = read_list_node(s, val_id, &prev_ptr)?;
    prev_node.anext = next_ptr;
    write_list_node(s, val_id, &prev_ptr, &prev_node)?;

    if next_ptr != Address::ZERO {
        let mut next_node = read_list_node(s, val_id, &next_ptr)?;
        next_node.aprev = prev_ptr;
        write_list_node(s, val_id, &next_ptr, &next_node)?;
    }

    // Clear this node's list pointers
    this_node.aprev = Address::ZERO;
    this_node.anext = Address::ZERO;
    write_list_node(s, val_id, ptr, &this_node)?;
    Ok(())
}

/// Remove a validator ID from the delegator's validator list.
fn linked_list_remove_val_id<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    delegator: &Address,
) -> Result<(), PrecompileError> {
    let mut this_node = read_list_node(s, val_id, delegator)?;
    // If iprev == empty, not in list
    if this_node.iprev == 0 {
        return Ok(());
    }

    let prev_ptr = this_node.iprev;
    let next_ptr = this_node.inext;

    let mut prev_node = read_list_node(s, prev_ptr, delegator)?;
    prev_node.inext = next_ptr;
    write_list_node(s, prev_ptr, delegator, &prev_node)?;

    if next_ptr != 0 {
        let mut next_node = read_list_node(s, next_ptr, delegator)?;
        next_node.iprev = prev_ptr;
        write_list_node(s, next_ptr, delegator, &next_node)?;
    }

    this_node.iprev = 0;
    this_node.inext = 0;
    write_list_node(s, val_id, delegator, &this_node)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Valset Management
// ═══════════════════════════════════════════════════════════════════════════════

/// Add a validator to the execution valset bitset. Returns true if newly inserted.
fn add_to_valset<S: StakingStorage>(s: &mut S, val_id: u64) -> Result<bool, PrecompileError> {
    let bucket_key = bitset_bucket_key(val_id);
    let set = read_u256(s, bucket_key)?;
    let bit = val_id & 0xFF;
    let mask = U256::from(1u64) << bit;
    let inserted = (set & mask).is_zero();
    let new_set = set | mask;
    write_storage_u256(s, bucket_key, new_set)?;

    if inserted {
        // Append to execution valset array
        let len = read_u64(s, valset_slots::EXECUTION)?;
        write_storage_u64(s, valset_slots::EXECUTION + U256::from(1 + len), val_id)?;
        write_storage_u64(s, valset_slots::EXECUTION, len + 1)?;
    }
    Ok(inserted)
}

/// Remove a validator from the execution valset bitset.
fn remove_from_valset<S: StakingStorage>(s: &mut S, val_id: u64) -> Result<(), PrecompileError> {
    let bucket_key = bitset_bucket_key(val_id);
    let set = read_u256(s, bucket_key)?;
    let bit = val_id & 0xFF;
    let mask = !(U256::from(1u64) << bit);
    let new_set = set & mask;
    write_storage_u256(s, bucket_key, new_set)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event Emission Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn emit_event<S: StakingStorage>(
    s: &mut S,
    topics: Vec<B256>,
    data: Vec<u8>,
) -> Result<(), PrecompileError> {
    s.emit_log(Log { address: STAKING_ADDRESS, data: LogData::new(topics, data.into()).unwrap() })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validator Flag Management
// ═══════════════════════════════════════════════════════════════════════════════

/// Update validator flags based on current state. Returns the new flags.
fn update_validator_flags_after_delegate<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    val: &Validator,
    del: &Delegator,
    caller: &Address,
) -> Result<u64, PrecompileError> {
    let mut flags = val.flags;

    // Clear STAKE_TOO_LOW if total stake meets threshold
    if val.stake >= ACTIVE_VALIDATOR_STAKE {
        flags &= !validator_flags::STAKE_TOO_LOW;
    }

    // Clear WITHDRAWN if auth address and their next-epoch stake meets minimum
    if *caller == val.auth_address && del.total_stake() >= MIN_AUTH_ADDRESS_STAKE {
        flags &= !validator_flags::WITHDRAWN;
    }

    if flags != val.flags {
        write_validator_flags(s, val_id, val, flags)?;
        // Emit ValidatorStatusChanged event
        let topics = vec![ValidatorStatusChanged::SIGNATURE_HASH, B256::from(U256::from(val_id))];
        let data = U256::from(flags).to_be_bytes::<32>().to_vec();
        emit_event(s, topics, data)?;
    }
    Ok(flags)
}

fn update_validator_flags_after_undelegate<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    val: &Validator,
    del: &Delegator,
    caller: &Address,
) -> Result<u64, PrecompileError> {
    let mut flags = val.flags;

    // Set STAKE_TOO_LOW if below threshold
    if val.stake < ACTIVE_VALIDATOR_STAKE {
        flags |= validator_flags::STAKE_TOO_LOW;
    }

    // Set WITHDRAWN if auth address and their next-epoch stake is below minimum
    if *caller == val.auth_address && del.total_stake() < MIN_AUTH_ADDRESS_STAKE {
        flags |= validator_flags::WITHDRAWN;
    }

    if flags != val.flags {
        write_validator_flags(s, val_id, val, flags)?;
        let topics = vec![ValidatorStatusChanged::SIGNATURE_HASH, B256::from(U256::from(val_id))];
        let data = U256::from(flags).to_be_bytes::<32>().to_vec();
        emit_event(s, topics, data)?;
    }
    Ok(flags)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal delegate logic (shared by delegate, compound, addValidator)
// ═══════════════════════════════════════════════════════════════════════════════

fn internal_delegate<S: StakingStorage>(
    s: &mut S,
    val_id: u64,
    caller: &Address,
    amount: U256,
) -> Result<(), PrecompileError> {
    let mut del = pull_delegator_up_to_date(s, val_id, caller)?;
    let in_boundary = read_in_boundary(s)?;
    let activation_epoch = get_activation_epoch(s)?;

    // Add to appropriate delta track
    if in_boundary {
        let need_acc = del.next_delta_epoch == 0;
        del.next_delta_stake = del.next_delta_stake.saturating_add(amount);
        del.next_delta_epoch = activation_epoch;
        write_delegator(s, val_id, caller, &del)?;
        if need_acc {
            increment_accumulator_refcount(s, val_id)?;
        }
    } else {
        let need_acc = del.delta_epoch == 0;
        del.delta_stake = del.delta_stake.saturating_add(amount);
        del.delta_epoch = activation_epoch;
        write_delegator(s, val_id, caller, &del)?;
        if need_acc {
            increment_accumulator_refcount(s, val_id)?;
        }
    }

    // Update validator total stake
    let mut val = read_validator(s, val_id)?;
    val.stake = val.stake.saturating_add(amount);
    write_validator_stake(s, val_id, val.stake)?;

    // Update flags
    let flags = update_validator_flags_after_delegate(s, val_id, &val, &del, caller)?;

    // Add to valset if flags are now OK
    if flags == validator_flags::OK {
        add_to_valset(s, val_id)?;
    }

    // Insert into linked lists
    linked_list_insert(s, val_id, caller)?;

    // Emit Delegate event
    let topics = vec![
        Delegate::SIGNATURE_HASH,
        B256::from(U256::from(val_id)),
        B256::from(U256::from_be_slice(caller.as_slice())),
    ];
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&amount.to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(activation_epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Write Function Handlers
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle changeCommission(uint64, uint256) => bool
pub fn handle_change_commission<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::CHANGE_COMMISSION {
        return Err(PrecompileError::OutOfGas);
    }

    let call = changeCommissionCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let val = read_validator(s, call.validatorId)?;
    if !val.exists() {
        return Err(PrecompileError::Other("unknown validator".into()));
    }
    if *caller != val.auth_address {
        return Err(PrecompileError::Other("requires auth address".into()));
    }
    if call.commission > MAX_COMMISSION {
        return Err(PrecompileError::Other("commission too high".into()));
    }

    let old_commission = val.commission;
    if call.commission != old_commission {
        write_validator_commission(s, call.validatorId, call.commission)?;
        // Emit CommissionChanged event
        let topics =
            vec![CommissionChanged::SIGNATURE_HASH, B256::from(U256::from(call.validatorId))];
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&old_commission.to_be_bytes::<32>());
        data.extend_from_slice(&call.commission.to_be_bytes::<32>());
        emit_event(s, topics, data)?;
    }

    let encoded = changeCommissionCall::abi_encode_returns(&true);
    Ok((gas::CHANGE_COMMISSION, encoded.into()))
}

/// Handle claimRewards(uint64) => bool
pub fn handle_claim_rewards<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::CLAIM_REWARDS {
        return Err(PrecompileError::OutOfGas);
    }

    let call = claimRewardsCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let mut del = pull_delegator_up_to_date(s, call.validatorId, caller)?;

    if !del.rewards.is_zero() {
        let rewards = del.rewards;

        // Transfer (solvency already checked in pull_delegator_up_to_date via reward_invariant)
        s.transfer(STAKING_ADDRESS, *caller, rewards)?;

        // Clear rewards
        del.rewards = U256::ZERO;
        write_delegator(s, call.validatorId, caller, &del)?;

        // Emit ClaimRewards event
        let epoch = read_epoch(s)?;
        let topics = vec![
            ClaimRewards::SIGNATURE_HASH,
            B256::from(U256::from(call.validatorId)),
            B256::from(U256::from_be_slice(caller.as_slice())),
        ];
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&rewards.to_be_bytes::<32>());
        data.extend_from_slice(&U256::from(epoch).to_be_bytes::<32>());
        emit_event(s, topics, data)?;
    }

    let encoded = claimRewardsCall::abi_encode_returns(&true);
    Ok((gas::CLAIM_REWARDS, encoded.into()))
}

/// Handle externalReward(uint64) => bool
pub fn handle_external_reward<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
    call_value: U256,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::EXTERNAL_REWARD {
        return Err(PrecompileError::OutOfGas);
    }

    let call = externalRewardCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let val = read_validator(s, call.validatorId)?;
    if !val.exists() {
        return Err(PrecompileError::Other("unknown validator".into()));
    }

    // Get active stake from consensus/snapshot view
    let in_boundary = read_in_boundary(s)?;
    let view_key = if in_boundary {
        snapshot_view_key(call.validatorId, 0)
    } else {
        consensus_view_key(call.validatorId, 0)
    };
    let active_stake = read_u256(s, view_key)?;
    if active_stake.is_zero() {
        return Err(PrecompileError::Other("not in validator set".into()));
    }

    if call_value < MIN_EXTERNAL_REWARD {
        return Err(PrecompileError::Other("external reward too small".into()));
    }
    if call_value > MAX_EXTERNAL_REWARD {
        return Err(PrecompileError::Other("external reward too large".into()));
    }

    // Apply reward: accumulator += (reward * UNIT_BIAS) / active_stake
    let reward_acc = call_value.saturating_mul(UNIT_BIAS) / active_stake;
    let new_acc = val.accumulated_reward_per_token.saturating_add(reward_acc);
    write_validator_acc(s, call.validatorId, new_acc)?;
    write_validator_unclaimed_rewards(
        s,
        call.validatorId,
        val.unclaimed_rewards.saturating_add(call_value),
    )?;

    // Emit ValidatorRewarded event
    let epoch = read_epoch(s)?;
    let topics = vec![
        ValidatorRewarded::SIGNATURE_HASH,
        B256::from(U256::from(call.validatorId)),
        B256::from(U256::from_be_slice(caller.as_slice())),
    ];
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&call_value.to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    let encoded = externalRewardCall::abi_encode_returns(&true);
    Ok((gas::EXTERNAL_REWARD, encoded.into()))
}

/// Handle delegate(uint64) => bool
pub fn handle_delegate<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
    call_value: U256,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::DELEGATE {
        return Err(PrecompileError::OutOfGas);
    }

    let call = delegateCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let val = read_validator(s, call.validatorId)?;
    if !val.exists() {
        return Err(PrecompileError::Other("unknown validator".into()));
    }
    // Zero-value delegate is a success no-op
    if call_value.is_zero() {
        let encoded = delegateCall::abi_encode_returns(&true);
        return Ok((gas::DELEGATE, encoded.into()));
    }
    if call_value < DUST_THRESHOLD {
        return Err(PrecompileError::Other("delegation is too small".into()));
    }

    internal_delegate(s, call.validatorId, caller, call_value)?;

    let encoded = delegateCall::abi_encode_returns(&true);
    Ok((gas::DELEGATE, encoded.into()))
}

/// Handle undelegate(uint64, uint256, uint8) => bool
pub fn handle_undelegate<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::UNDELEGATE {
        return Err(PrecompileError::OutOfGas);
    }

    let call = undelegateCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    // No-op for zero amount
    if call.amount.is_zero() {
        let encoded = undelegateCall::abi_encode_returns(&true);
        return Ok((gas::UNDELEGATE, encoded.into()));
    }

    let val = read_validator(s, call.validatorId)?;
    if !val.exists() {
        return Err(PrecompileError::Other("unknown validator".into()));
    }

    // Check withdrawal ID doesn't exist
    let existing_wr = read_u256(
        s,
        withdrawal_key(call.validatorId, caller, call.withdrawId, withdrawal_offsets::AMOUNT),
    )?;
    if !existing_wr.is_zero() {
        return Err(PrecompileError::Other("withdrawal id exists".into()));
    }

    // Pull delegator up to date
    let mut del = pull_delegator_up_to_date(s, call.validatorId, caller)?;

    // Check sufficient stake
    let mut amount = call.amount;
    if del.stake < amount {
        return Err(PrecompileError::Other("insufficient stake".into()));
    }

    // Dust collection
    let remaining = del.stake.saturating_sub(amount);
    if !remaining.is_zero() && remaining < DUST_THRESHOLD {
        amount = del.stake; // collect all including dust
    }

    // Save the accumulator before potential reset (needed for withdrawal request)
    let wr_acc = del.accumulated_reward_per_token;

    // Update delegator stake
    del.stake = del.stake.saturating_sub(amount);
    if del.stake.is_zero() {
        del.accumulated_reward_per_token = U256::ZERO;
    }

    // Update validator stake
    let mut val = read_validator(s, call.validatorId)?;
    val.stake = val.stake.saturating_sub(amount);
    write_validator_stake(s, call.validatorId, val.stake)?;

    // Update flags. Don't remove from valset here — removal happens at snapshot time
    // via pop-and-swap compaction at snapshot time.
    update_validator_flags_after_undelegate(s, call.validatorId, &val, &del, caller)?;

    // Create withdrawal request with the pre-reset accumulator
    let activation_epoch = get_activation_epoch(s)?;
    write_withdrawal_request(
        s,
        call.validatorId,
        caller,
        call.withdrawId,
        amount,
        wr_acc,
        activation_epoch,
    )?;
    increment_accumulator_refcount(s, call.validatorId)?;

    // Write updated delegator
    write_delegator(s, call.validatorId, caller, &del)?;

    // Remove from linked lists if no more stake
    if !del.exists() {
        linked_list_remove(s, call.validatorId, caller)?;
    }

    // Emit Undelegate event
    let topics = vec![
        Undelegate::SIGNATURE_HASH,
        B256::from(U256::from(call.validatorId)),
        B256::from(U256::from_be_slice(caller.as_slice())),
    ];
    let mut data = Vec::with_capacity(96);
    data.extend_from_slice(&U256::from(call.withdrawId).to_be_bytes::<32>());
    data.extend_from_slice(&amount.to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(activation_epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    let encoded = undelegateCall::abi_encode_returns(&true);
    Ok((gas::UNDELEGATE, encoded.into()))
}

/// Handle withdraw(uint64, uint8) => bool
pub fn handle_withdraw<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::WITHDRAW {
        return Err(PrecompileError::OutOfGas);
    }

    let call = withdrawCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    // Load withdrawal request
    let wr_amount = read_u256(
        s,
        withdrawal_key(call.validatorId, caller, call.withdrawId, withdrawal_offsets::AMOUNT),
    )?;
    if wr_amount.is_zero() {
        return Err(PrecompileError::Other("unknown withdrawal id".into()));
    }
    let wr_acc = read_u256(
        s,
        withdrawal_key(call.validatorId, caller, call.withdrawId, withdrawal_offsets::ACCUMULATOR),
    )?;
    let wr_epoch = read_u64(
        s,
        withdrawal_key(call.validatorId, caller, call.withdrawId, withdrawal_offsets::EPOCH),
    )?;

    // Check withdrawal ready
    let current_epoch = read_epoch(s)?;
    if !is_epoch_active(current_epoch, wr_epoch) || wr_epoch + WITHDRAWAL_DELAY > current_epoch {
        return Err(PrecompileError::Other("withdrawal not ready".into()));
    }

    // Decrement accumulator and calculate rewards
    let epoch_acc = decrement_accumulator_refcount(s, wr_epoch, call.validatorId)?;
    let rewards = calculate_rewards(wr_amount, epoch_acc, wr_acc);

    // Check solvency
    let val = read_validator(s, call.validatorId)?;
    if val.unclaimed_rewards < rewards {
        return Err(PrecompileError::Other("solvency error".into()));
    }
    write_validator_unclaimed_rewards(
        s,
        call.validatorId,
        val.unclaimed_rewards.saturating_sub(rewards),
    )?;

    // Transfer total payout
    let total_payout = wr_amount.saturating_add(rewards);
    s.transfer(STAKING_ADDRESS, *caller, total_payout)?;

    // Clear withdrawal request
    clear_withdrawal_request(s, call.validatorId, caller, call.withdrawId)?;

    // Emit Withdraw event
    let topics = vec![
        Withdraw::SIGNATURE_HASH,
        B256::from(U256::from(call.validatorId)),
        B256::from(U256::from_be_slice(caller.as_slice())),
    ];
    let mut data = Vec::with_capacity(96);
    data.extend_from_slice(&U256::from(call.withdrawId).to_be_bytes::<32>());
    data.extend_from_slice(&total_payout.to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(current_epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    let encoded = withdrawCall::abi_encode_returns(&true);
    Ok((gas::WITHDRAW, encoded.into()))
}

/// Handle compound(uint64) => bool
pub fn handle_compound<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::COMPOUND {
        return Err(PrecompileError::OutOfGas);
    }

    let call = compoundCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let mut del = pull_delegator_up_to_date(s, call.validatorId, caller)?;

    if !del.rewards.is_zero() {
        let rewards = del.rewards;

        // Clear rewards (solvency already checked in pull_delegator_up_to_date via reward_invariant)
        del.rewards = U256::ZERO;
        write_delegator(s, call.validatorId, caller, &del)?;

        // Emit ClaimRewards event
        let epoch = read_epoch(s)?;
        let topics = vec![
            ClaimRewards::SIGNATURE_HASH,
            B256::from(U256::from(call.validatorId)),
            B256::from(U256::from_be_slice(caller.as_slice())),
        ];
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&rewards.to_be_bytes::<32>());
        data.extend_from_slice(&U256::from(epoch).to_be_bytes::<32>());
        emit_event(s, topics, data)?;

        // Re-delegate the rewards (compound)
        internal_delegate(s, call.validatorId, caller, rewards)?;
    }

    let encoded = compoundCall::abi_encode_returns(&true);
    Ok((gas::COMPOUND, encoded.into()))
}

/// Handle addValidator(bytes, bytes, bytes) => uint64
pub fn handle_add_validator<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    _caller: &Address,
    call_value: U256,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::ADD_VALIDATOR {
        return Err(PrecompileError::OutOfGas);
    }

    let call = addValidatorCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    // Decode payload: secp_pubkey(33) + bls_pubkey(48) + auth_address(20) + signed_stake(32) + commission(32) = 165 bytes
    let payload = &call.payload;
    if payload.len() < 165 {
        return Err(PrecompileError::Other("invalid input".into()));
    }

    let mut secp_pubkey = [0u8; 33];
    secp_pubkey.copy_from_slice(&payload[0..33]);
    let mut bls_pubkey = [0u8; 48];
    bls_pubkey.copy_from_slice(&payload[33..81]);
    let auth_address = Address::from_slice(&payload[81..101]);
    let signed_stake = U256::from_be_slice(&payload[101..133]);
    let commission = U256::from_be_slice(&payload[133..165]);

    // Validate
    if call_value != signed_stake {
        return Err(PrecompileError::Other("invalid input".into()));
    }
    if call_value < MIN_AUTH_ADDRESS_STAKE {
        return Err(PrecompileError::Other("insufficient stake".into()));
    }
    if commission > MAX_COMMISSION {
        return Err(PrecompileError::Other("commission too high".into()));
    }

    // Skip signature verification (per design decision)

    // Derive addresses from pubkeys for existence check
    // For secp: use keccak256 of uncompressed pubkey to get eth address
    // For now, use the auth_address directly as the secp lookup key
    let secp_addr = auth_address; // simplified: use auth_address
    let bls_addr = Address::from_slice(&bls_pubkey[0..20]); // simplified: first 20 bytes

    // Check validator doesn't already exist
    let existing_secp = read_u64(s, val_id_secp_key(&secp_addr))?;
    if existing_secp != 0 {
        return Err(PrecompileError::Other("validator exists".into()));
    }
    let existing_bls = read_u64(s, super::storage::val_id_bls_key(&bls_addr))?;
    if existing_bls != 0 {
        return Err(PrecompileError::Other("validator exists".into()));
    }

    // Increment last_val_id
    let last_val_id = read_u64(s, global_slots::LAST_VAL_ID)?;
    let new_val_id = last_val_id + 1;
    write_storage_u64(s, global_slots::LAST_VAL_ID, new_val_id)?;

    // Store ID mappings
    write_storage_u64(s, val_id_secp_key(&secp_addr), new_val_id)?;
    write_storage_u64(s, super::storage::val_id_bls_key(&bls_addr), new_val_id)?;

    // Create validator
    let val = Validator {
        stake: U256::ZERO,
        accumulated_reward_per_token: U256::ZERO,
        commission,
        secp_pubkey,
        bls_pubkey,
        auth_address,
        flags: validator_flags::STAKE_TOO_LOW,
        unclaimed_rewards: U256::ZERO,
    };
    write_validator_full(s, new_val_id, &val)?;

    // Emit ValidatorCreated event
    let topics = vec![
        ValidatorCreated::SIGNATURE_HASH,
        B256::from(U256::from(new_val_id)),
        B256::from(U256::from_be_slice(auth_address.as_slice())),
    ];
    let data = commission.to_be_bytes::<32>().to_vec();
    emit_event(s, topics, data)?;

    // Delegate initial stake (from auth_address, not caller)
    internal_delegate(s, new_val_id, &auth_address, call_value)?;

    let encoded = addValidatorCall::abi_encode_returns(&new_val_id);
    Ok((gas::ADD_VALIDATOR, encoded.into()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// getDelegator Write Handler
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle getDelegator as a write operation.
///
/// The canonical implementation calls `pull_delegator_up_to_date` which writes settled
/// state to storage. The gas cost (184,900) includes warm_sstores.
pub fn handle_get_delegator_write<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::GET_DELEGATOR {
        return Err(PrecompileError::OutOfGas);
    }

    let call = getDelegatorCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    // pull_delegator_up_to_date persists settled state
    let del = pull_delegator_up_to_date(s, call.validatorId, &call.delegator)?;

    let encoded = getDelegatorCall::abi_encode_returns(&getDelegatorReturn {
        stake: del.stake,
        accRewardPerToken: del.accumulated_reward_per_token,
        unclaimedRewards: del.rewards,
        deltaStake: del.delta_stake,
        nextDeltaStake: del.next_delta_stake,
        deltaEpoch: del.delta_epoch,
        nextDeltaEpoch: del.next_delta_epoch,
    });
    Ok((gas::GET_DELEGATOR, encoded.into()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Syscall Handlers
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle syscallReward(address) — distribute block rewards.
///
/// The block reward amount is resolved from two sources (in priority order):
/// 1. **Extended calldata**: If `input` is >= 68 bytes, the reward is read from
///    bytes `[36..68]` (appended after the standard ABI data). This is used by
///    [`crate::api::block::apply_syscall_reward`] for `SystemCallEvm` integration.
/// 2. **`msg.value`** (`call_value`): Fallback for direct calls (e.g., Foundry
///    `vm.prank(SYSTEM_ADDRESS)` with `{value: reward}`).
pub fn handle_syscall_reward<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
    call_value: U256,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::SYSCALL_REWARD {
        return Err(PrecompileError::OutOfGas);
    }
    if *caller != SYSTEM_ADDRESS {
        return Err(PrecompileError::Other("Unauthorized: not system address".into()));
    }

    let call = syscallRewardCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let block_author = call.blockAuthor;

    // Resolve block reward: extended calldata (36..68) takes priority over msg.value
    // Standard ABI: 4 (selector) + 32 (address) = 36 bytes
    // Extended:     4 (selector) + 32 (address) + 32 (reward) = 68 bytes
    let block_reward =
        if input.len() >= 68 { U256::from_be_slice(&input[36..68]) } else { call_value };

    // Look up validator from block author address
    let val_id = read_u64(s, val_id_secp_key(&block_author))?;
    if val_id == 0 {
        // Unknown author — return NotInValidatorSet error
        return Err(PrecompileError::Other("not in validator set".into()));
    }

    // Get active stake from the appropriate view
    let in_boundary = read_in_boundary(s)?;
    let view_key =
        if in_boundary { snapshot_view_key(val_id, 0) } else { consensus_view_key(val_id, 0) };
    let active_stake = read_u256(s, view_key)?;
    if active_stake.is_zero() {
        // Zero active stake — return NotInValidatorSet error
        return Err(PrecompileError::Other("not in validator set".into()));
    }

    // Mint tokens: add block_reward to staking contract balance
    // Note: for syscalls, the balance isn't transferred via msg.value,
    // so we need to mint directly
    // We handle this by adding to staking address balance
    // (In the real implementation, mint_tokens adds to the STAKING_CA balance)

    // Get commission rate
    let commission_view_key =
        if in_boundary { snapshot_view_key(val_id, 1) } else { consensus_view_key(val_id, 1) };
    let commission_rate = read_u256(s, commission_view_key)?;

    // Calculate commission: commission_amount = (block_reward * commission_rate) / MON
    let commission_amount = block_reward.saturating_mul(commission_rate) / MON;
    let del_reward = block_reward.saturating_sub(commission_amount);

    // Credit commission to auth address delegator rewards
    let val = read_validator(s, val_id)?;
    let auth_addr = val.auth_address;
    let mut auth_del = read_delegator(s, val_id, &auth_addr)?;
    auth_del.rewards = auth_del.rewards.saturating_add(commission_amount);
    write_delegator(s, val_id, &auth_addr, &auth_del)?;

    // Add del_reward (not block_reward) to unclaimed_rewards
    write_validator_unclaimed_rewards(s, val_id, val.unclaimed_rewards.saturating_add(del_reward))?;

    // Apply reward to accumulator: acc += (del_reward * UNIT_BIAS) / active_stake
    if !del_reward.is_zero() && !active_stake.is_zero() {
        let reward_acc = del_reward.saturating_mul(UNIT_BIAS) / active_stake;
        let new_acc = val.accumulated_reward_per_token.saturating_add(reward_acc);
        write_validator_acc(s, val_id, new_acc)?;
    }

    // Write proposer_val_id
    write_storage_u64(s, global_slots::PROPOSER_VAL_ID, val_id)?;

    // Emit ValidatorRewarded event
    let epoch = read_epoch(s)?;
    let topics = vec![
        ValidatorRewarded::SIGNATURE_HASH,
        B256::from(U256::from(val_id)),
        B256::from(U256::from_be_slice(SYSTEM_ADDRESS.as_slice())),
    ];
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&del_reward.to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    Ok((gas::SYSCALL_REWARD, Bytes::new()))
}

/// Handle syscallSnapshot() — take epoch boundary snapshot.
pub fn handle_syscall_snapshot<S: StakingStorage>(
    s: &mut S,
    _input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::SYSCALL_SNAPSHOT {
        return Err(PrecompileError::OutOfGas);
    }
    if *caller != SYSTEM_ADDRESS {
        return Err(PrecompileError::Other("Unauthorized: not system address".into()));
    }

    // Check not already in boundary
    let in_boundary = read_in_boundary(s)?;
    if in_boundary {
        return Err(PrecompileError::Other("called snapshot while in boundary".into()));
    }

    // Set in_epoch_delay_period = true (left-aligned bool: byte 0 = 1)
    let mut boundary_true = [0u8; 32];
    boundary_true[0] = 1;
    write_storage_u256(s, global_slots::IN_BOUNDARY, U256::from_be_bytes(boundary_true))?;

    // Read consensus set for snapshot
    let consensus_len = read_u64(s, valset_slots::CONSENSUS)?;

    // 1. Clear old snapshot set: pop each entry and clear its view slots
    let old_snapshot_len = read_u64(s, valset_slots::SNAPSHOT)?;
    for i in 0..old_snapshot_len {
        let old_val_id = read_u64(s, valset_slots::SNAPSHOT + U256::from(1 + i))?;
        write_storage_u256(s, snapshot_view_key(old_val_id, 0), U256::ZERO)?;
        write_storage_u256(s, snapshot_view_key(old_val_id, 1), U256::ZERO)?;
        write_storage_u64(s, valset_slots::SNAPSHOT + U256::from(1 + i), 0)?;
    }

    // 2. Copy consensus → snapshot (both array and view)
    write_storage_u64(s, valset_slots::SNAPSHOT, consensus_len)?;
    for i in 0..consensus_len {
        let val_id = read_u64(s, valset_slots::CONSENSUS + U256::from(1 + i))?;
        write_storage_u64(s, valset_slots::SNAPSHOT + U256::from(1 + i), val_id)?;

        // Copy consensus view → snapshot view
        let c_stake = read_u256(s, consensus_view_key(val_id, 0))?;
        let c_commission = read_u256(s, consensus_view_key(val_id, 1))?;
        write_storage_u256(s, snapshot_view_key(val_id, 0), c_stake)?;
        write_storage_u256(s, snapshot_view_key(val_id, 1), c_commission)?;
    }

    // 3. Clear old consensus set: pop each entry and clear its view slots
    for i in 0..consensus_len {
        let old_val_id = read_u64(s, valset_slots::CONSENSUS + U256::from(1 + i))?;
        write_storage_u256(s, consensus_view_key(old_val_id, 0), U256::ZERO)?;
        write_storage_u256(s, consensus_view_key(old_val_id, 1), U256::ZERO)?;
        write_storage_u64(s, valset_slots::CONSENSUS + U256::from(1 + i), 0)?;
    }

    // 4. Build new consensus set from execution set (top N by stake)
    let exec_len = read_u64(s, valset_slots::EXECUTION)?;
    let mut validators: Vec<(u64, U256)> = Vec::with_capacity(exec_len as usize);
    for i in 0..exec_len {
        let val_id = read_u64(s, valset_slots::EXECUTION + U256::from(1 + i))?;
        let stake = read_u256(s, validator_key(val_id, validator_offsets::STAKE))?;
        // Only include validators with OK flags
        let addr_flags = read_u256(s, validator_key(val_id, validator_offsets::ADDRESS_FLAGS))?
            .to_be_bytes::<32>();
        let flags = u64::from_be_bytes(addr_flags[20..28].try_into().unwrap());
        if flags == validator_flags::OK {
            validators.push((val_id, stake));
        }
    }

    // Sort by stake descending, then val_id ascending as tie-breaker
    validators.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Take top ACTIVE_VALSET_SIZE
    let new_consensus_len = validators.len().min(ACTIVE_VALSET_SIZE as usize);

    // 5. Write new consensus set with view
    write_storage_u64(s, valset_slots::CONSENSUS, new_consensus_len as u64)?;
    for (i, (val_id, _)) in validators.iter().take(new_consensus_len).enumerate() {
        write_storage_u64(s, valset_slots::CONSENSUS + U256::from(1 + i as u64), *val_id)?;
        // Write consensus view
        let val = read_validator(s, *val_id)?;
        write_storage_u256(s, consensus_view_key(*val_id, 0), val.stake)?;
        write_storage_u256(s, consensus_view_key(*val_id, 1), val.commission)?;
    }

    // Compact execution set: remove non-OK validators using pop-and-swap.
    // Re-read exec_len since we need the current value.
    let exec_len = read_u64(s, valset_slots::EXECUTION)?;
    let mut removals: Vec<u64> = Vec::new();
    for i in 0..exec_len {
        let vid = read_u64(s, valset_slots::EXECUTION + U256::from(1 + i))?;
        let addr_flags =
            read_u256(s, validator_key(vid, validator_offsets::ADDRESS_FLAGS))?.to_be_bytes::<32>();
        let flags = u64::from_be_bytes(addr_flags[20..28].try_into().unwrap());
        if flags != validator_flags::OK {
            removals.push(i);
        }
    }
    let mut current_len = exec_len;
    for &idx in removals.iter().rev() {
        let removed_vid = read_u64(s, valset_slots::EXECUTION + U256::from(1 + idx))?;
        remove_from_valset(s, removed_vid)?;
        current_len -= 1;
        if idx < current_len {
            let last_vid = read_u64(s, valset_slots::EXECUTION + U256::from(1 + current_len))?;
            write_storage_u64(s, valset_slots::EXECUTION + U256::from(1 + idx), last_vid)?;
        }
        write_storage_u64(s, valset_slots::EXECUTION + U256::from(1 + current_len), 0)?;
    }
    write_storage_u64(s, valset_slots::EXECUTION, current_len)?;

    Ok((gas::SYSCALL_SNAPSHOT, Bytes::new()))
}

/// Handle syscallOnEpochChange(uint64) — finalize epoch transition.
pub fn handle_syscall_on_epoch_change<S: StakingStorage>(
    s: &mut S,
    input: &[u8],
    gas_limit: u64,
    caller: &Address,
) -> Result<(u64, Bytes), PrecompileError> {
    if gas_limit < gas::SYSCALL_ON_EPOCH_CHANGE {
        return Err(PrecompileError::OutOfGas);
    }
    if *caller != SYSTEM_ADDRESS {
        return Err(PrecompileError::Other("Unauthorized: not system address".into()));
    }

    let call = syscallOnEpochChangeCall::abi_decode_raw(&input[4..])
        .map_err(|e| PrecompileError::Other(format!("Invalid input: {e}").into()))?;

    let current_epoch = read_epoch(s)?;
    // Allow any strictly increasing epoch (next > last), not just +1
    if call.epoch <= current_epoch {
        return Err(PrecompileError::Other("invalid epoch change".into()));
    }

    let next_epoch = call.epoch;
    let next_next_epoch = next_epoch + 1;

    // Emit EpochChanged event before state changes
    let topics = vec![EpochChanged::SIGNATURE_HASH];
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&U256::from(current_epoch).to_be_bytes::<32>());
    data.extend_from_slice(&U256::from(next_epoch).to_be_bytes::<32>());
    emit_event(s, topics, data)?;

    // Update accumulator values for snapshot validators.
    // For each validator in the snapshot set, update existing accumulator entries
    // for next_epoch and next_epoch+1 to the current validator acc value.
    // This ensures delta_stake activations use the accumulator at epoch transition,
    // not the stale value from delegation time.
    let snapshot_len = read_u64(s, valset_slots::SNAPSHOT)?;
    for i in 0..snapshot_len {
        let val_slot = valset_slots::SNAPSHOT.saturating_add(U256::from(i + 1));
        let val_id = read_u64(s, val_slot)?;
        let current_val_acc =
            read_u256(s, validator_key(val_id, validator_offsets::ACCUMULATED_REWARD_PER_TOKEN))?;

        // Update accumulator for next_epoch if it has refcount > 0
        let acc_next = read_accumulator(s, next_epoch, val_id)?;
        if acc_next.refcount > 0 {
            write_accumulator(
                s,
                next_epoch,
                val_id,
                &RefCountedAccumulator { value: current_val_acc, refcount: acc_next.refcount },
            )?;
        }

        // Update accumulator for next_epoch+1 if it has refcount > 0
        let acc_next_next = read_accumulator(s, next_next_epoch, val_id)?;
        if acc_next_next.refcount > 0 {
            write_accumulator(
                s,
                next_next_epoch,
                val_id,
                &RefCountedAccumulator { value: current_val_acc, refcount: acc_next_next.refcount },
            )?;
        }
    }

    // Clear in_epoch_delay_period and set new epoch
    write_storage_u256(s, global_slots::IN_BOUNDARY, U256::ZERO)?;
    write_storage_u64(s, global_slots::EPOCH, next_epoch)?;

    Ok((gas::SYSCALL_ON_EPOCH_CHANGE, Bytes::new()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dispatch Entry Point for Write Operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Check if a selector corresponds to a payable function.
pub const fn is_payable_selector(selector: [u8; 4]) -> bool {
    matches!(
        selector,
        delegateCall::SELECTOR | addValidatorCall::SELECTOR | externalRewardCall::SELECTOR
    )
}

/// Check if a selector corresponds to a syscall.
pub const fn is_syscall_selector(selector: [u8; 4]) -> bool {
    matches!(
        selector,
        syscallRewardCall::SELECTOR
            | syscallSnapshotCall::SELECTOR
            | syscallOnEpochChangeCall::SELECTOR
    )
}

/// Check if a selector corresponds to a write function (user-callable, syscall, or mutating getter).
pub const fn is_write_selector(selector: [u8; 4]) -> bool {
    matches!(
        selector,
        delegateCall::SELECTOR
            | undelegateCall::SELECTOR
            | withdrawCall::SELECTOR
            | compoundCall::SELECTOR
            | claimRewardsCall::SELECTOR
            | addValidatorCall::SELECTOR
            | changeCommissionCall::SELECTOR
            | externalRewardCall::SELECTOR
            | syscallRewardCall::SELECTOR
            | syscallSnapshotCall::SELECTOR
            | syscallOnEpochChangeCall::SELECTOR
            | getDelegatorCall::SELECTOR
    )
}

/// Check that msg.value is zero (non-payable method guard).
/// Matches C++ `function_not_payable` which is called inside each non-payable method.
fn function_not_payable(call_value: &U256) -> Result<(), PrecompileError> {
    if !call_value.is_zero() {
        return Err(PrecompileError::Other("value non-zero".into()));
    }
    Ok(())
}

/// Create a fallback result for unknown/short selectors.
/// Consumes FALLBACK gas (40k) and returns "method not supported" revert.
fn fallback_result(gas_limit: u64) -> InterpreterResult {
    if gas_limit < gas::FALLBACK {
        return InterpreterResult {
            result: InstructionResult::PrecompileOOG,
            output: Bytes::new(),
            gas: Gas::new(gas_limit),
        };
    }
    let mut gas = Gas::new(gas_limit);
    let _ = gas.record_cost(gas_limit);
    InterpreterResult {
        result: InstructionResult::Revert,
        output: Bytes::from("method not supported"),
        gas,
    }
}

/// Dispatch a write operation using the StakingStorage trait.
///
/// Returns `Ok(InterpreterResult)` or `Err(String)`.
pub fn run_staking_write<S: StakingStorage>(
    input: &[u8],
    gas_limit: u64,
    storage: &mut S,
    caller: &Address,
    call_value: U256,
) -> Result<InterpreterResult, String> {
    // Short input routes to fallback with 40k gas cost
    let selector: [u8; 4] = match input.get(..4).and_then(|s| s.try_into().ok()) {
        Some(s) => s,
        None => return Ok(fallback_result(gas_limit)),
    };

    // Dispatch to handler. Payability is checked per-method (matching C++ dispatch-first
    // semantics). Unknown/fallback selectors don't check payability.
    let result = match selector {
        // Payable methods (accept msg.value)
        delegateCall::SELECTOR => handle_delegate(storage, input, gas_limit, caller, call_value),
        addValidatorCall::SELECTOR => {
            handle_add_validator(storage, input, gas_limit, caller, call_value)
        }
        externalRewardCall::SELECTOR => {
            handle_external_reward(storage, input, gas_limit, caller, call_value)
        }
        // Non-payable methods (reject msg.value > 0 inside dispatch)
        changeCommissionCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_change_commission(storage, input, gas_limit, caller)),
        claimRewardsCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_claim_rewards(storage, input, gas_limit, caller)),
        undelegateCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_undelegate(storage, input, gas_limit, caller)),
        withdrawCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_withdraw(storage, input, gas_limit, caller)),
        compoundCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_compound(storage, input, gas_limit, caller)),
        // Syscalls: reward accepts value, snapshot/epoch-change are non-payable.
        syscallRewardCall::SELECTOR => {
            handle_syscall_reward(storage, input, gas_limit, caller, call_value)
        }
        syscallSnapshotCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_syscall_snapshot(storage, input, gas_limit, caller)),
        syscallOnEpochChangeCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_syscall_on_epoch_change(storage, input, gas_limit, caller)),
        // Mutating getter (non-payable)
        getDelegatorCall::SELECTOR => function_not_payable(&call_value)
            .and_then(|_| handle_get_delegator_write(storage, input, gas_limit)),
        // Unknown selector → fallback (no payability check, just "method not supported")
        _ => return Ok(fallback_result(gas_limit)),
    };

    match result {
        Ok((gas_used, output)) => {
            let mut ir = InterpreterResult {
                result: InstructionResult::Return,
                gas: Gas::new(gas_limit),
                output,
            };
            if !ir.gas.record_cost(gas_used) {
                ir.result = InstructionResult::PrecompileOOG;
            }
            Ok(ir)
        }
        Err(e) => {
            // Consume all gas on revert (gas_left = 0)
            let mut gas = Gas::new(gas_limit);
            let _ = gas.record_cost(gas_limit);
            Ok(InterpreterResult {
                result: if e.is_oog() {
                    InstructionResult::PrecompileOOG
                } else {
                    InstructionResult::Revert
                },
                gas,
                output: Bytes::copy_from_slice(e.to_string().as_bytes()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct MockStorage {
        slots: HashMap<U256, U256>,
        logs: Vec<Log>,
        transfers: Vec<(Address, Address, U256)>,
    }

    impl MockStorage {
        fn set_u64_left(&mut self, key: U256, value: u64) {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&value.to_be_bytes());
            self.slots.insert(key, U256::from_be_bytes(bytes));
        }

        fn get_u64_left(&self, key: U256) -> u64 {
            let value = self.slots.get(&key).copied().unwrap_or(U256::ZERO);
            let bytes = value.to_be_bytes::<32>();
            u64::from_be_bytes(bytes[..8].try_into().unwrap())
        }
    }

    impl StorageReader for MockStorage {
        fn sload(&mut self, key: U256) -> Result<U256, PrecompileError> {
            Ok(self.slots.get(&key).copied().unwrap_or(U256::ZERO))
        }
    }

    impl StakingStorage for MockStorage {
        fn sstore(&mut self, key: U256, value: U256) -> Result<(), PrecompileError> {
            self.slots.insert(key, value);
            Ok(())
        }

        fn transfer(
            &mut self,
            from: Address,
            to: Address,
            amount: U256,
        ) -> Result<(), PrecompileError> {
            self.transfers.push((from, to, amount));
            Ok(())
        }

        fn emit_log(&mut self, log: Log) -> Result<(), PrecompileError> {
            self.logs.push(log);
            Ok(())
        }
    }

    fn address_flags_slot(auth: Address, flags: u64) -> U256 {
        let mut bytes = [0u8; 32];
        bytes[..20].copy_from_slice(auth.as_slice());
        bytes[20..28].copy_from_slice(&flags.to_be_bytes());
        U256::from_be_bytes(bytes)
    }

    #[test]
    fn test_syscall_reward_accepts_nonzero_call_value() {
        let mut storage = MockStorage::default();

        let block_author = Address::new([0x11; 20]);
        let auth_address = Address::new([0x22; 20]);
        let val_id = 7u64;
        let reward = U256::from(5u64);

        // Resolve block author -> validator ID
        storage.set_u64_left(val_id_secp_key(&block_author), val_id);

        // Active in consensus set with zero commission.
        storage.slots.insert(consensus_view_key(val_id, 0), U256::from(10u64));
        storage.slots.insert(consensus_view_key(val_id, 1), U256::ZERO);

        // Validator metadata (exists + auth address).
        storage.slots.insert(
            validator_key(val_id, validator_offsets::ADDRESS_FLAGS),
            address_flags_slot(auth_address, validator_flags::OK),
        );

        // Standard ABI calldata (36 bytes). Reward comes from call_value.
        let input = syscallRewardCall { blockAuthor: block_author }.abi_encode();
        let result = run_staking_write(
            &input,
            gas::SYSCALL_REWARD + 1_000,
            &mut storage,
            &SYSTEM_ADDRESS,
            reward,
        )
        .expect("syscall reward should execute");

        assert_eq!(result.result, InstructionResult::Return);
        assert_eq!(result.output, Bytes::new());
        assert_eq!(storage.get_u64_left(global_slots::PROPOSER_VAL_ID), val_id);
        assert_eq!(
            storage
                .slots
                .get(&validator_key(val_id, validator_offsets::UNCLAIMED_REWARDS))
                .copied()
                .unwrap_or(U256::ZERO),
            reward
        );
    }

    #[test]
    fn test_snapshot_clears_stale_snapshot_and_consensus_views() {
        let mut storage = MockStorage::default();

        let old_snapshot_val = 11u64;
        let old_consensus_val = 22u64;
        let new_consensus_val = 33u64;

        // Old snapshot set with stale view.
        storage.set_u64_left(valset_slots::SNAPSHOT, 1);
        storage.set_u64_left(valset_slots::SNAPSHOT + U256::from(1u64), old_snapshot_val);
        storage.slots.insert(snapshot_view_key(old_snapshot_val, 0), U256::from(101u64));
        storage.slots.insert(snapshot_view_key(old_snapshot_val, 1), U256::from(202u64));

        // Old consensus set and view.
        storage.set_u64_left(valset_slots::CONSENSUS, 1);
        storage.set_u64_left(valset_slots::CONSENSUS + U256::from(1u64), old_consensus_val);
        storage.slots.insert(consensus_view_key(old_consensus_val, 0), U256::from(303u64));
        storage.slots.insert(consensus_view_key(old_consensus_val, 1), U256::from(404u64));

        // Execution set with one OK validator to become the new consensus entry.
        storage.set_u64_left(valset_slots::EXECUTION, 1);
        storage.set_u64_left(valset_slots::EXECUTION + U256::from(1u64), new_consensus_val);
        storage
            .slots
            .insert(validator_key(new_consensus_val, validator_offsets::STAKE), U256::from(999u64));
        storage.slots.insert(
            validator_key(new_consensus_val, validator_offsets::COMMISSION),
            U256::from(77u64),
        );
        storage.slots.insert(
            validator_key(new_consensus_val, validator_offsets::ADDRESS_FLAGS),
            address_flags_slot(Address::new([0x33; 20]), validator_flags::OK),
        );

        let (_gas, output) = handle_syscall_snapshot(
            &mut storage,
            &[],
            gas::SYSCALL_SNAPSHOT + 1_000,
            &SYSTEM_ADDRESS,
        )
        .expect("snapshot should succeed");
        assert!(output.is_empty());

        // Old snapshot val view cleared.
        assert_eq!(
            storage
                .slots
                .get(&snapshot_view_key(old_snapshot_val, 0))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::ZERO
        );
        assert_eq!(
            storage
                .slots
                .get(&snapshot_view_key(old_snapshot_val, 1))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::ZERO
        );

        // Old consensus val view cleared.
        assert_eq!(
            storage
                .slots
                .get(&consensus_view_key(old_consensus_val, 0))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::ZERO
        );
        assert_eq!(
            storage
                .slots
                .get(&consensus_view_key(old_consensus_val, 1))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::ZERO
        );

        // Snapshot now contains previous consensus val.
        assert_eq!(storage.get_u64_left(valset_slots::SNAPSHOT), 1);
        assert_eq!(
            storage.get_u64_left(valset_slots::SNAPSHOT + U256::from(1u64)),
            old_consensus_val
        );
        assert_eq!(
            storage
                .slots
                .get(&snapshot_view_key(old_consensus_val, 0))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::from(303u64)
        );
        assert_eq!(
            storage
                .slots
                .get(&snapshot_view_key(old_consensus_val, 1))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::from(404u64)
        );

        // Consensus rebuilt from execution.
        assert_eq!(storage.get_u64_left(valset_slots::CONSENSUS), 1);
        assert_eq!(
            storage.get_u64_left(valset_slots::CONSENSUS + U256::from(1u64)),
            new_consensus_val
        );
        assert_eq!(
            storage
                .slots
                .get(&consensus_view_key(new_consensus_val, 0))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::from(999u64)
        );
        assert_eq!(
            storage
                .slots
                .get(&consensus_view_key(new_consensus_val, 1))
                .copied()
                .unwrap_or(U256::ZERO),
            U256::from(77u64)
        );
    }
}
