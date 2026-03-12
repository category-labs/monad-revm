//! Monad journal wrapper with reserve-balance tracking.

use crate::reserve_balance::tracker::ReserveBalanceTracker;
use revm::{
    bytecode::Bytecode,
    context::{journal::JournalInner, Journal},
    context_interface::{
        context::{SStoreResult, SelfDestructResult, StateLoad},
        journaled_state::{
            entry::JournalEntry, AccountInfoLoad, AccountLoad, JournalCheckpoint, JournalLoadError,
            JournalTr, TransferError,
        },
    },
    database_interface::Database,
    inspector::JournalExt,
    primitives::{
        hardfork::SpecId, Address, HashMap, HashSet, Log, StorageKey, StorageValue, B256, U256,
    },
    state::{Account, EvmState},
};
use std::ops::{Deref, DerefMut};

/// Monad journal extension used by reserve-balance-aware execution.
pub trait MonadJournalTr: JournalTr<State = EvmState> {
    /// Returns the reserve-balance tracker.
    fn reserve_balance(&self) -> &ReserveBalanceTracker;

    /// Returns the reserve-balance tracker mutably.
    fn reserve_balance_mut(&mut self) -> &mut ReserveBalanceTracker;
}

/// Monad journal wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonadJournal<DB: Database> {
    inner: Journal<DB>,
    reserve_balance: ReserveBalanceTracker,
}

impl<DB: Database> MonadJournal<DB> {
    /// Consumes the journal and returns the wrapped database.
    pub fn into_database(self) -> DB {
        self.inner.database
    }

    /// Creates a new Monad journal from an existing journal inner state.
    pub fn new_with_inner(
        database: DB,
        inner: JournalInner<JournalEntry>,
        reserve_balance: ReserveBalanceTracker,
    ) -> Self {
        Self { inner: Journal::new_with_inner(database, inner), reserve_balance }
    }

    fn on_transfer(&mut self, from: Address, to: Address) {
        let state = &self.inner.state;
        self.reserve_balance.on_debit(state.get(&from), from);
        self.reserve_balance.on_credit(state.get(&to), to);
    }

    fn on_checkpoint_revert(&mut self, checkpoint: JournalCheckpoint) {
        let reverted_addresses: Vec<_> = self
            .inner
            .journal
            .get(checkpoint.journal_i..)
            .into_iter()
            .flatten()
            .flat_map(reverted_addresses_from_entry)
            .collect();
        self.inner.checkpoint_revert(checkpoint);
        self.reserve_balance.on_checkpoint_revert(reverted_addresses, &self.inner.state);
    }
}

impl<DB: Database> Deref for MonadJournal<DB> {
    type Target = Journal<DB>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<DB: Database> DerefMut for MonadJournal<DB> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<DB: Database> MonadJournalTr for MonadJournal<DB> {
    fn reserve_balance(&self) -> &ReserveBalanceTracker {
        &self.reserve_balance
    }

    fn reserve_balance_mut(&mut self) -> &mut ReserveBalanceTracker {
        &mut self.reserve_balance
    }
}

impl<DB: Database> JournalExt for MonadJournal<DB> {
    fn journal(&self) -> &[JournalEntry] {
        &self.inner.journal
    }

    fn evm_state(&self) -> &EvmState {
        &self.inner.state
    }

    fn evm_state_mut(&mut self) -> &mut EvmState {
        &mut self.inner.state
    }
}

impl<DB: Database> JournalTr for MonadJournal<DB> {
    type Database = DB;
    type State = EvmState;
    type JournaledAccount<'a>
        = <Journal<DB> as JournalTr>::JournaledAccount<'a>
    where
        DB: 'a;

    fn new(database: DB) -> Self {
        Self { inner: Journal::new(database), reserve_balance: ReserveBalanceTracker::default() }
    }

    fn db(&self) -> &Self::Database {
        self.inner.db()
    }

    fn db_mut(&mut self) -> &mut Self::Database {
        self.inner.db_mut()
    }

    fn sload(
        &mut self,
        address: Address,
        key: StorageKey,
    ) -> Result<StateLoad<StorageValue>, <Self::Database as Database>::Error> {
        self.inner.sload(address, key)
    }

    fn sstore(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
    ) -> Result<StateLoad<SStoreResult>, <Self::Database as Database>::Error> {
        self.inner.sstore(address, key, value)
    }

    fn tload(&mut self, address: Address, key: StorageKey) -> StorageValue {
        self.inner.tload(address, key)
    }

    fn tstore(&mut self, address: Address, key: StorageKey, value: StorageValue) {
        self.inner.tstore(address, key, value)
    }

    fn log(&mut self, log: Log) {
        self.inner.log(log)
    }

    fn take_logs(&mut self) -> Vec<Log> {
        self.inner.take_logs()
    }

    fn logs(&self) -> &[Log] {
        self.inner.logs()
    }

    fn selfdestruct(
        &mut self,
        address: Address,
        target: Address,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SelfDestructResult>, JournalLoadError<<Self::Database as Database>::Error>>
    {
        let result = self.inner.selfdestruct(address, target, skip_cold_load)?;
        self.on_transfer(address, target);
        Ok(result)
    }

    fn warm_access_list(&mut self, access_list: HashMap<Address, HashSet<StorageKey>>) {
        self.inner.warm_access_list(access_list)
    }

    fn warm_coinbase_account(&mut self, address: Address) {
        self.inner.warm_coinbase_account(address)
    }

    fn warm_precompiles(&mut self, precompiles: HashSet<Address>) {
        self.inner.warm_precompiles(precompiles)
    }

    fn precompile_addresses(&self) -> &HashSet<Address> {
        self.inner.precompile_addresses()
    }

    fn set_spec_id(&mut self, spec_id: SpecId) {
        self.inner.set_spec_id(spec_id)
    }

    fn touch_account(&mut self, address: Address) {
        self.inner.touch_account(address)
    }

    fn transfer(
        &mut self,
        from: Address,
        to: Address,
        balance: U256,
    ) -> Result<Option<TransferError>, DB::Error> {
        let result = self.inner.transfer(from, to, balance)?;
        if result.is_none() {
            self.on_transfer(from, to);
        }
        Ok(result)
    }

    fn transfer_loaded(
        &mut self,
        from: Address,
        to: Address,
        balance: U256,
    ) -> Option<TransferError> {
        let result = self.inner.transfer_loaded(from, to, balance);
        if result.is_none() {
            self.on_transfer(from, to);
        }
        result
    }

    #[allow(deprecated)]
    fn caller_accounting_journal_entry(
        &mut self,
        address: Address,
        old_balance: U256,
        bump_nonce: bool,
    ) {
        self.inner.caller_accounting_journal_entry(address, old_balance, bump_nonce)
    }

    fn balance_incr(
        &mut self,
        address: Address,
        balance: U256,
    ) -> Result<(), <Self::Database as Database>::Error> {
        self.inner.balance_incr(address, balance)?;
        self.reserve_balance.on_credit(self.inner.state.get(&address), address);
        Ok(())
    }

    #[allow(deprecated)]
    fn nonce_bump_journal_entry(&mut self, address: Address) {
        self.inner.nonce_bump_journal_entry(address)
    }

    fn load_account(
        &mut self,
        address: Address,
    ) -> Result<StateLoad<&Account>, <Self::Database as Database>::Error> {
        self.inner.load_account(address)
    }

    fn load_account_mut_skip_cold_load(
        &mut self,
        address: Address,
        skip_cold_load: bool,
    ) -> Result<StateLoad<Self::JournaledAccount<'_>>, <Self::Database as Database>::Error> {
        self.inner.load_account_mut_skip_cold_load(address, skip_cold_load)
    }

    fn load_account_mut_optional_code(
        &mut self,
        address: Address,
        load_code: bool,
    ) -> Result<StateLoad<Self::JournaledAccount<'_>>, <Self::Database as Database>::Error> {
        self.inner.load_account_mut_optional_code(address, load_code)
    }

    fn load_account_with_code(
        &mut self,
        address: Address,
    ) -> Result<StateLoad<&Account>, <Self::Database as Database>::Error> {
        self.inner.load_account_with_code(address)
    }

    fn load_account_delegated(
        &mut self,
        address: Address,
    ) -> Result<StateLoad<AccountLoad>, <Self::Database as Database>::Error> {
        self.inner.load_account_delegated(address)
    }

    fn checkpoint(&mut self) -> JournalCheckpoint {
        self.inner.checkpoint()
    }

    fn checkpoint_commit(&mut self) {
        self.inner.checkpoint_commit()
    }

    fn checkpoint_revert(&mut self, checkpoint: JournalCheckpoint) {
        self.on_checkpoint_revert(checkpoint)
    }

    fn set_code_with_hash(&mut self, address: Address, code: Bytecode, hash: B256) {
        let tracker_code = code.clone();
        self.inner.set_code_with_hash(address, code, hash);
        self.reserve_balance.on_set_code(self.inner.state.get(&address), address, &tracker_code);
    }

    fn create_account_checkpoint(
        &mut self,
        caller: Address,
        address: Address,
        balance: U256,
        spec_id: SpecId,
    ) -> Result<JournalCheckpoint, TransferError> {
        let checkpoint = self.inner.create_account_checkpoint(caller, address, balance, spec_id)?;
        self.on_transfer(caller, address);
        Ok(checkpoint)
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn commit_tx(&mut self) {
        self.inner.commit_tx();
        self.reserve_balance.clear();
    }

    fn discard_tx(&mut self) {
        self.inner.discard_tx();
        self.reserve_balance.clear();
    }

    fn finalize(&mut self) -> Self::State {
        self.reserve_balance.clear();
        self.inner.finalize()
    }

    fn sload_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, JournalLoadError<<Self::Database as Database>::Error>>
    {
        self.inner.sload_skip_cold_load(address, key, skip_cold_load)
    }

    fn sstore_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, JournalLoadError<<Self::Database as Database>::Error>>
    {
        self.inner.sstore_skip_cold_load(address, key, value, skip_cold_load)
    }

    fn load_account_info_skip_cold_load(
        &mut self,
        address: Address,
        load_code: bool,
        skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, JournalLoadError<<Self::Database as Database>::Error>> {
        self.inner.load_account_info_skip_cold_load(address, load_code, skip_cold_load)
    }
}

fn reverted_addresses_from_entry(entry: &JournalEntry) -> Vec<Address> {
    match entry {
        JournalEntry::AccountWarmed { address }
        | JournalEntry::AccountTouched { address }
        | JournalEntry::BalanceChange { address, .. }
        | JournalEntry::NonceChange { address, .. }
        | JournalEntry::NonceBump { address }
        | JournalEntry::AccountCreated { address, .. }
        | JournalEntry::StorageChanged { address, .. }
        | JournalEntry::StorageWarmed { address, .. }
        | JournalEntry::TransientStorageChange { address, .. }
        | JournalEntry::CodeChange { address } => vec![*address],
        JournalEntry::BalanceTransfer { from, to, .. } => vec![*from, *to],
        JournalEntry::AccountDestroyed { address, target, .. } => vec![*address, *target],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::{
        context_interface::journaled_state::JournalCheckpoint, database_interface::EmptyDB,
    };

    #[test]
    fn checkpoint_revert_without_entries_is_noop() {
        let mut journal = MonadJournal::new(EmptyDB::new());
        let checkpoint = journal.checkpoint();

        journal.checkpoint_revert(checkpoint);

        assert!(journal.journal().is_empty());
    }

    #[test]
    fn checkpoint_revert_ignores_out_of_bounds_journal_index() {
        let mut journal = MonadJournal::new(EmptyDB::new());

        journal.checkpoint_revert(JournalCheckpoint { log_i: 0, journal_i: 4 });

        assert!(journal.journal().is_empty());
    }
}
