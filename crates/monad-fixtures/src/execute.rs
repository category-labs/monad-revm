//! Valid-block execution and state/receipt commitment checks.

use crate::fixture::{Account, Allocation, Fixture};
use alloy_consensus::{
    proofs::{calculate_receipt_root, calculate_transaction_root, calculate_withdrawals_root},
    transaction::SignerRecoverable,
    Receipt, ReceiptEnvelope, Transaction, TxType, EMPTY_OMMER_ROOT_HASH,
};
use alloy_primitives::Bloom;
use alloy_trie::{
    root::{state_root_unhashed, storage_root_unhashed},
    TrieAccount,
};
use eyre::{ensure, eyre, Result, WrapErr};
use monad_revm::{
    monad_context_with_db, MonadBuilder, MonadCfgEnv, MonadChainContext, MonadJournalTr,
};
use revm::{
    context::{BlockEnv, TxEnv},
    context_interface::{either::Either, result::ExecutionResult, ContextTr, JournalTr},
    database::InMemoryDB,
    handler::system_call::{SystemCallEvm, SYSTEM_ADDRESS},
    primitives::{address, b256, keccak256, Address, Bytes, HashSet, B256, U256},
    state::{AccountInfo, Bytecode, EvmState},
    DatabaseCommit, ExecuteEvm,
};
use std::collections::BTreeSet;

const BEACON_ROOTS: Address = address!("000f3df6d732807ef1319fb7b8bb8522d0beac02");
const HISTORY_STORAGE: Address = address!("0000f90827f1c53a10cb7a02335b175320002935");
const EMPTY_REQUESTS_HASH: B256 =
    b256!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

#[derive(Debug)]
pub(crate) struct Stats {
    pub blocks: usize,
    pub transactions: usize,
}

pub(crate) fn execute(fixture: &Fixture) -> Result<Stats> {
    let mut db = InMemoryDB::default();
    for (address, account) in &fixture.pre {
        let code = Bytecode::new_raw_checked(account.code.clone())
            .wrap_err_with(|| format!("account {address}: invalid pre-state bytecode"))?;
        db.insert_account_info(
            *address,
            AccountInfo {
                balance: account.balance,
                nonce: account.nonce,
                code_hash: keccak256(&account.code),
                code: Some(code),
                ..Default::default()
            },
        );
        for (slot, value) in &account.storage {
            db.insert_account_storage(*address, *slot, *value)?;
        }
    }
    ensure!(
        state_root(&fixture.pre) == fixture.genesis.state_root,
        "genesis state root differs from pre allocation"
    );
    let mut parent = &fixture.genesis;
    let mut parent_participants = HashSet::default();
    let mut grandparent_participants = HashSet::default();
    db.cache.block_hashes.insert(U256::from(parent.number), parent.hash_slow());
    let mut stats = Stats { blocks: 0, transactions: 0 };

    for (index, block) in fixture.blocks.iter().enumerate() {
        let header = &block.header;
        ensure!(
            parent.number.checked_add(1) == Some(header.number),
            "block {index}: nonconsecutive block number"
        );
        ensure!(header.parent_hash == parent.hash_slow(), "block {index}: parent hash mismatch");
        ensure!(header.timestamp >= parent.timestamp, "block {index}: timestamp precedes parent");
        ensure!(
            header.difficulty.is_zero() && header.nonce.is_zero(),
            "block {index}: nonzero proof-of-work fields"
        );
        ensure!(header.ommers_hash == EMPTY_OMMER_ROOT_HASH, "block {index}: nonempty ommers hash");
        ensure!(header.extra_data.len() <= 32, "block {index}: extraData exceeds 32 bytes");
        ensure!(
            header.blob_gas_used.unwrap_or_default() == 0,
            "block {index}: nonzero blob gas used"
        );
        if let Some(hash) = header.requests_hash {
            ensure!(hash == EMPTY_REQUESTS_HASH, "block {index}: expected an empty requests hash");
        }
        let txs = &block.body.transactions;
        ensure!(
            calculate_transaction_root(txs) == header.transactions_root,
            "block {index}: transactions root mismatch"
        );
        let withdrawals = block.body.withdrawals.as_ref().map_or(&[][..], |w| w.as_slice());
        if let Some(expected) = header.withdrawals_root {
            ensure!(
                calculate_withdrawals_root(withdrawals) == expected,
                "block {index}: withdrawals root mismatch"
            );
        } else {
            ensure!(withdrawals.is_empty(), "block {index}: withdrawals without a header root");
        }

        let senders = txs
            .iter()
            .map(|tx| tx.recover_signer().wrap_err("invalid transaction signature"))
            .collect::<Result<Vec<_>>>()?;
        let authorities = txs
            .iter()
            .map(|tx| {
                tx.authorization_list()
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|auth| auth.recover_authority().ok())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let participants =
            senders.iter().copied().chain(authorities.iter().flatten().copied()).collect();
        let spec = fixture.network.at(header.timestamp);
        let mut context = monad_context_with_db(&mut db);
        context.cfg = MonadCfgEnv::new_with_spec(spec).with_chain_id(fixture.chain_id);
        context.journaled_state.set_monad_spec(spec);
        context.journaled_state.set_spec_id(spec.into());
        context.block = BlockEnv {
            number: U256::from(header.number),
            beneficiary: header.beneficiary,
            timestamp: U256::from(header.timestamp),
            gas_limit: header.gas_limit,
            basefee: header
                .base_fee_per_gas
                .ok_or_else(|| eyre!("block {index}: missing base fee"))?,
            difficulty: header.difficulty,
            prevrandao: Some(header.mix_hash),
            ..Default::default()
        };
        context.chain = MonadChainContext {
            parent_senders_and_authorities: parent_participants,
            grandparent_senders_and_authorities: grandparent_participants,
            current_block_senders: senders.clone(),
            current_block_authorities: authorities,
            ..Default::default()
        };
        let mut evm = context.build_monad();
        for (target, data) in [
            (BEACON_ROOTS, header.parent_beacon_block_root),
            (HISTORY_STORAGE, Some(header.parent_hash)),
        ] {
            let has_code = evm.0.ctx.db().cache.accounts.get(&target).is_some_and(|account| {
                account.info().is_some() && account.info.code_hash != keccak256([])
            });
            if let Some(data) = data.filter(|_| has_code) {
                let mut result = evm
                    .system_call_with_caller(
                        SYSTEM_ADDRESS,
                        target,
                        Bytes::copy_from_slice(data.as_slice()),
                    )
                    .wrap_err_with(|| format!("block {index}: system call to {target}"))?;
                normalize_changes(&mut result.state);
                evm.0.ctx.db_mut().commit(result.state);
            }
        }
        let mut cumulative_gas_used = 0u64;
        let mut receipts = Vec::with_capacity(txs.len());
        let mut outcomes = Vec::with_capacity(txs.len());
        let mut bloom = Bloom::ZERO;
        for (tx_index, (tx, sender)) in txs.iter().zip(senders).enumerate() {
            ensure!(
                tx.tx_type() != TxType::Eip4844,
                "block {index}, transaction {tx_index}: blob transactions are invalid on Monad"
            );
            cumulative_gas_used = cumulative_gas_used
                .checked_add(tx.gas_limit())
                .ok_or_else(|| eyre!("block {index}: cumulative gas overflow"))?;
            ensure!(
                cumulative_gas_used <= header.gas_limit,
                "block {index}: transactions exceed block gas limit"
            );
            evm.0.ctx.chain.current_tx_index = tx_index;
            let env = TxEnv {
                tx_type: tx.tx_type() as u8,
                caller: sender,
                gas_limit: tx.gas_limit(),
                gas_price: tx.max_fee_per_gas(),
                kind: tx.kind(),
                value: tx.value(),
                data: tx.input().clone(),
                nonce: tx.nonce(),
                chain_id: tx.chain_id(),
                access_list: tx.access_list().cloned().unwrap_or_default(),
                gas_priority_fee: tx.max_priority_fee_per_gas(),
                authorization_list: tx
                    .authorization_list()
                    .unwrap_or_default()
                    .iter()
                    .cloned()
                    .map(Either::Left)
                    .collect(),
                ..Default::default()
            };
            let mut result = evm.transact(env).wrap_err_with(|| {
                format!("block {index}, transaction {tx_index}: unexpected transaction rejection")
            })?;
            outcomes.push(match &result.result {
                ExecutionResult::Success { reason, .. } => {
                    format!("{tx_index}: success ({reason:?})")
                }
                ExecutionResult::Revert { .. } => format!("{tx_index}: revert"),
                ExecutionResult::Halt { reason, .. } => format!("{tx_index}: halt ({reason:?})"),
            });
            // Monad receipt gas includes the entire gas limit, independently of VM gas spent.
            let receipt = Receipt {
                status: result.result.is_success().into(),
                cumulative_gas_used,
                logs: result.result.into_logs(),
            }
            .with_bloom();
            bloom |= receipt.logs_bloom;
            receipts.push(ReceiptEnvelope::from_typed(tx.tx_type(), receipt));
            normalize_changes(&mut result.state);
            evm.0.ctx.db_mut().commit(result.state);
            stats.transactions += 1;
        }
        grandparent_participants =
            std::mem::take(&mut evm.0.ctx.chain.parent_senders_and_authorities);
        parent_participants = participants;
        drop(evm);

        for withdrawal in withdrawals {
            if withdrawal.amount == 0 {
                continue;
            }
            let account = db.load_account(withdrawal.address)?;
            let amount = U256::from(withdrawal.amount) * U256::from(1_000_000_000u64);
            let mut info = account.info().unwrap_or_default();
            info.balance = info
                .balance
                .checked_add(amount)
                .ok_or_else(|| eyre!("withdrawal balance overflow"))?;
            db.insert_account_info(withdrawal.address, info);
        }
        ensure!(
            cumulative_gas_used == header.gas_used,
            "block {index}: gas used mismatch: expected {}, got {cumulative_gas_used}",
            header.gas_used
        );
        ensure!(
            bloom == header.logs_bloom,
            "block {index}: logs bloom mismatch; transactions [{}]",
            outcomes.join(", ")
        );
        let receipt_root = calculate_receipt_root(&receipts);
        ensure!(receipt_root == header.receipts_root, "block {index}: receipts root mismatch: expected {}, got {receipt_root}; transactions [{}]", header.receipts_root, outcomes.join(", "));
        let actual_root = state_root(&snapshot(&db));
        ensure!(
            actual_root == header.state_root,
            "block {index}: state root mismatch: expected {}, got {actual_root}",
            header.state_root
        );
        db.cache.block_hashes.insert(U256::from(header.number), header.hash_slow());
        parent = header;
        stats.blocks += 1;
    }
    ensure!(parent.hash_slow() == fixture.last_block_hash, "lastblockhash mismatch");
    let actual = snapshot(&db);
    if let Some(expected) = &fixture.post {
        compare_accounts(&actual, expected)?;
    }
    if let Some(expected) = fixture.post_hash {
        ensure!(state_root(&actual) == expected, "postStateHash mismatch");
    }
    Ok(stats)
}

fn normalize_changes(state: &mut EvmState) {
    // CacheDB does not apply EIP-161's deletion of touched empty accounts itself.
    for account in state.values_mut() {
        if account.is_touched() && account.info.is_empty() {
            account.mark_selfdestruct();
        }
    }
}

fn snapshot(db: &InMemoryDB) -> Allocation {
    db.cache
        .accounts
        .iter()
        .filter_map(|(address, account)| {
            let info = account.info()?;
            let code = db
                .cache
                .contracts
                .get(&info.code_hash)
                .map(Bytecode::original_bytes)
                .unwrap_or_default();
            let storage = account
                .storage
                .iter()
                .filter(|(_, value)| !value.is_zero())
                .map(|(k, v)| (*k, *v))
                .collect();
            Some((*address, Account { nonce: info.nonce, balance: info.balance, code, storage }))
        })
        .collect()
}

fn state_root(accounts: &Allocation) -> B256 {
    state_root_unhashed(accounts.iter().map(|(address, account)| {
        let storage_root = storage_root_unhashed(
            account
                .storage
                .iter()
                .map(|(key, value)| (B256::from(key.to_be_bytes::<32>()), *value)),
        );
        (
            *address,
            TrieAccount {
                nonce: account.nonce,
                balance: account.balance,
                code_hash: keccak256(&account.code),
                storage_root,
            },
        )
    }))
}

fn compare_accounts(actual: &Allocation, expected: &Allocation) -> Result<()> {
    let addresses = actual.keys().chain(expected.keys()).collect::<BTreeSet<_>>();
    for address in addresses {
        match (actual.get(address), expected.get(address)) {
            (Some(actual), Some(expected)) => {
                ensure!(
                    actual.nonce == expected.nonce,
                    "account {address}: nonce expected {}, got {}",
                    expected.nonce,
                    actual.nonce
                );
                ensure!(
                    actual.balance == expected.balance,
                    "account {address}: balance expected {}, got {}",
                    expected.balance,
                    actual.balance
                );
                ensure!(actual.code == expected.code, "account {address}: code mismatch");
                let slots =
                    actual.storage.keys().chain(expected.storage.keys()).collect::<BTreeSet<_>>();
                for slot in slots {
                    let actual = actual.storage.get(slot).copied().unwrap_or_default();
                    let expected = expected.storage.get(slot).copied().unwrap_or_default();
                    ensure!(actual == expected, "account {address}, storage {slot:#x}: expected {expected:#x}, got {actual:#x}");
                }
            }
            (Some(_), None) => return Err(eyre!("unexpected account {address}")),
            (None, Some(_)) => return Err(eyre!("missing account {address}")),
            (None, None) => unreachable!(),
        }
    }
    Ok(())
}
