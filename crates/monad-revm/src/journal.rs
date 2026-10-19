//! Monad journal wrapper with reserve-balance, page-access, and call-frame tracking.

use crate::{
    page::{PageAccessTracker, StoragePageKey},
    reserve_balance::tracker::ReserveBalanceTracker,
    MonadHardfork,
};
use alloc::{vec, vec::Vec};
use core::ops::{Deref, DerefMut};
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
        hardfork::SpecId, Address, AddressMap, AddressSet, HashSet, Log, StorageKey, StorageValue,
        B256, U256,
    },
    state::{Account, EvmState},
};

/// Monad journal extension used by Monad execution.
///
/// Checkpoint rollback must notify [`ReserveBalanceTracker::before_checkpoint_revert`] before
/// restoring state unless reserve tracking is preserved for a synthetic frame. It must always
/// notify [`ReserveBalanceTracker::on_checkpoint_revert`] after restoring state.
pub trait MonadJournalTr: JournalTr<State = EvmState> {
    /// Returns the reserve-balance tracker.
    fn reserve_balance(&self) -> &ReserveBalanceTracker;

    /// Returns the reserve-balance tracker mutably.
    fn reserve_balance_mut(&mut self) -> &mut ReserveBalanceTracker;

    /// Returns the active Monad hardfork for journal-specific behavior.
    fn monad_spec(&self) -> MonadHardfork;

    /// Updates the active Monad hardfork for journal-specific behavior.
    fn set_monad_spec(&mut self, spec: MonadHardfork);

    /// Returns the MIP-8 page access tracker.
    fn page_access(&self) -> &PageAccessTracker;

    /// Returns the MIP-8 page access tracker mutably.
    fn page_access_mut(&mut self) -> &mut PageAccessTracker;

    /// Returns the caller recorded for an active frame at the given zero-based depth.
    ///
    /// The frame must have been recorded with [`Self::set_call_frame_sender`].
    fn call_frame_sender(&self, depth: usize) -> Address;

    /// Records a frame's CALLER value at its zero-based depth for MIP-18.
    ///
    /// Called by the EVM only when MIP-18 is active (`MonadNext`).
    /// Entries are overwritten on frame entry, not reverted with state checkpoints. Only
    /// depths at or below the executing frame may be read. Clear entries at transaction end.
    fn set_call_frame_sender(&mut self, depth: usize, sender: Address);

    /// Snapshots execution state without increasing the observable call depth.
    ///
    /// This checkpoint must be opened after transaction fees, CALL nonce advancement, and
    /// authorizations, and settled after all execution frames have returned.
    fn checkpoint_execution(&mut self) -> JournalCheckpoint;

    /// Accepts the execution snapshot without changing the call depth.
    fn commit_execution(&mut self);

    /// Reverts the execution snapshot without changing the call depth.
    fn revert_execution(&mut self, checkpoint: JournalCheckpoint);

    /// Returns whether transaction boundaries preserve the reserve-balance tracker.
    fn preserves_reserve_balance_tracker(&self) -> bool {
        false
    }

    /// Sets whether transaction boundaries preserve the reserve-balance tracker.
    ///
    /// This is intended for synthetic transactions that model an internal call while retaining
    /// the enclosing protocol transaction's reserve-balance state. Such calls defer reserve
    /// enforcement: their completion does not finalize the enclosing transaction. The embedder
    /// must check `reserve_balance().has_violation()` before accepting the enclosing execution
    /// checkpoint. Restore this to `false` before executing the next ordinary transaction.
    fn set_preserve_reserve_balance_tracker(&mut self, _preserve: bool) {}
}

/// Monad journal wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonadJournal<DB: Database> {
    inner: Journal<DB>,
    reserve_balance: ReserveBalanceTracker,
    preserve_reserve_balance_tracker: bool,
    monad_spec: MonadHardfork,
    page_access: PageAccessTracker,
    call_frame_senders: Vec<Address>,
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
        Self {
            inner: Journal::new_with_inner(database, inner),
            reserve_balance,
            preserve_reserve_balance_tracker: false,
            monad_spec: MonadHardfork::default(),
            page_access: PageAccessTracker::default(),
            call_frame_senders: Vec::new(),
        }
    }

    #[inline]
    const fn page_access_enabled(&self) -> bool {
        MonadHardfork::MonadTen.is_enabled_in(self.monad_spec)
    }

    #[inline]
    fn page_key(address: Address, key: StorageKey) -> StoragePageKey {
        StoragePageKey::from_slot(address, key)
    }

    fn on_transfer(&mut self, from: Address, to: Address) {
        let state = &self.inner.state;
        self.reserve_balance.on_debit(state.get(&from), from);
        self.reserve_balance.on_credit(state.get(&to), to);
    }

    fn on_checkpoint_revert(&mut self, checkpoint: JournalCheckpoint) {
        if !self.preserve_reserve_balance_tracker {
            self.reserve_balance.before_checkpoint_revert(self.inner.depth());
        }
        let reverted_addresses: Vec<_> = self
            .inner
            .journal
            .get(checkpoint.journal_i..)
            .into_iter()
            .flatten()
            .flat_map(reverted_addresses_from_entry)
            .collect();
        self.page_access.checkpoint_revert();
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

    fn monad_spec(&self) -> MonadHardfork {
        self.monad_spec
    }

    fn set_monad_spec(&mut self, spec: MonadHardfork) {
        self.monad_spec = spec;
    }

    fn page_access(&self) -> &PageAccessTracker {
        &self.page_access
    }

    fn page_access_mut(&mut self) -> &mut PageAccessTracker {
        &mut self.page_access
    }

    fn call_frame_sender(&self, depth: usize) -> Address {
        self.call_frame_senders[depth]
    }

    fn set_call_frame_sender(&mut self, depth: usize, sender: Address) {
        if depth >= self.call_frame_senders.len() {
            self.call_frame_senders.resize(depth + 1, Address::ZERO);
        }
        self.call_frame_senders[depth] = sender;
    }

    fn checkpoint_execution(&mut self) -> JournalCheckpoint {
        self.page_access.checkpoint();
        let checkpoint = self.inner.checkpoint();
        self.inner.checkpoint_commit();
        checkpoint
    }

    fn commit_execution(&mut self) {
        self.page_access.checkpoint_commit();
    }

    fn revert_execution(&mut self, checkpoint: JournalCheckpoint) {
        // Balance the depth decrement performed by the ordinary checkpoint revert.
        self.inner.checkpoint();
        self.on_checkpoint_revert(checkpoint);
    }

    fn preserves_reserve_balance_tracker(&self) -> bool {
        self.preserve_reserve_balance_tracker
    }

    fn set_preserve_reserve_balance_tracker(&mut self, preserve: bool) {
        self.preserve_reserve_balance_tracker = preserve;
    }
}

impl<DB: Database> JournalExt for MonadJournal<DB> {
    fn journal(&self) -> &[JournalEntry] {
        &self.inner.journal
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
        Self {
            inner: Journal::new(database),
            reserve_balance: ReserveBalanceTracker::default(),
            preserve_reserve_balance_tracker: false,
            monad_spec: MonadHardfork::default(),
            page_access: PageAccessTracker::default(),
            call_frame_senders: Vec::new(),
        }
    }

    fn db(&self) -> &Self::Database {
        self.inner.db()
    }

    fn db_mut(&mut self) -> &mut Self::Database {
        self.inner.db_mut()
    }

    fn db_and_state(&self) -> (&Self::Database, &Self::State) {
        self.inner.db_and_state()
    }

    fn db_and_state_mut(&mut self) -> (&mut Self::Database, &mut Self::State) {
        self.inner.db_and_state_mut()
    }

    fn sload(
        &mut self,
        address: Address,
        key: StorageKey,
    ) -> Result<StateLoad<StorageValue>, <Self::Database as Database>::Error> {
        self.sload_skip_cold_load(address, key, false).map_err(JournalLoadError::unwrap_db_error)
    }

    fn sstore(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
    ) -> Result<StateLoad<SStoreResult>, <Self::Database as Database>::Error> {
        self.sstore_skip_cold_load(address, key, value, false)
            .map_err(JournalLoadError::unwrap_db_error)
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

    fn warm_access_list(&mut self, access_list: AddressMap<HashSet<StorageKey>>) {
        if self.page_access_enabled() {
            self.page_access.warm_access_list(&access_list);
        }
        self.inner.warm_access_list(access_list)
    }

    fn warm_coinbase_account(&mut self, address: Address) {
        self.inner.warm_coinbase_account(address)
    }

    fn warm_precompiles(&mut self, precompiles: &AddressSet) {
        self.inner.warm_precompiles(precompiles)
    }

    fn precompile_addresses(&self) -> &AddressSet {
        self.inner.precompile_addresses()
    }

    fn set_spec_id(&mut self, spec_id: SpecId) {
        self.inner.set_spec_id(spec_id)
    }

    fn set_eip7708_config(&mut self, disabled: bool, delayed_burn_disabled: bool) {
        self.inner.set_eip7708_config(disabled, delayed_burn_disabled)
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
    ) -> Result<
        StateLoad<Self::JournaledAccount<'_>>,
        JournalLoadError<<Self::Database as Database>::Error>,
    > {
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
        self.page_access.checkpoint();
        self.inner.checkpoint()
    }

    fn checkpoint_commit(&mut self) {
        self.page_access.checkpoint_commit();
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
        self.page_access.checkpoint();
        let checkpoint =
            match self.inner.create_account_checkpoint(caller, address, balance, spec_id) {
                Ok(checkpoint) => checkpoint,
                Err(err) => {
                    self.page_access.checkpoint_revert();
                    return Err(err);
                }
            };
        self.on_transfer(caller, address);
        Ok(checkpoint)
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn commit_tx(&mut self) {
        self.inner.commit_tx();
        self.call_frame_senders.clear();
        self.page_access.clear();
        if !self.preserve_reserve_balance_tracker {
            self.reserve_balance.clear();
        }
    }

    fn discard_tx(&mut self) {
        let reverted_addresses = self.preserve_reserve_balance_tracker.then(|| {
            self.inner.journal.iter().flat_map(reverted_addresses_from_entry).collect::<Vec<_>>()
        });
        self.inner.discard_tx();
        self.call_frame_senders.clear();
        self.page_access.clear();
        if let Some(reverted_addresses) = reverted_addresses {
            self.reserve_balance.on_checkpoint_revert(reverted_addresses, &self.inner.state);
        } else {
            self.reserve_balance.clear();
        }
    }

    fn finalize(&mut self) -> Self::State {
        self.call_frame_senders.clear();
        self.page_access.clear();
        if !self.preserve_reserve_balance_tracker {
            self.reserve_balance.clear();
        }
        self.inner.finalize()
    }

    fn sload_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, JournalLoadError<<Self::Database as Database>::Error>>
    {
        if !self.page_access_enabled() {
            return self.inner.sload_skip_cold_load(address, key, skip_cold_load);
        }

        let page_key = Self::page_key(address, key);
        let is_cold = !self.page_access.is_read_accessed(&page_key);
        if is_cold && skip_cold_load {
            return Err(JournalLoadError::ColdLoadSkipped);
        }

        let storage = self.inner.sload_skip_cold_load(address, key, false)?;
        if is_cold {
            self.page_access.mark_read_accessed(page_key);
        }

        Ok(StateLoad::new(storage.data, is_cold))
    }

    fn sstore_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, JournalLoadError<<Self::Database as Database>::Error>>
    {
        if !self.page_access_enabled() {
            return self.inner.sstore_skip_cold_load(address, key, value, skip_cold_load);
        }

        let page_key = Self::page_key(address, key);
        let is_cold = !self.page_access.is_read_accessed(&page_key);
        if is_cold && skip_cold_load {
            return Err(JournalLoadError::ColdLoadSkipped);
        }

        let storage = self.inner.sstore_skip_cold_load(address, key, value, false)?;
        if is_cold {
            self.page_access.mark_read_accessed(page_key);
        }

        Ok(StateLoad::new(storage.data, is_cold))
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
        | JournalEntry::CodeChange { address, .. } => vec![*address],
        JournalEntry::BalanceTransfer { from, to, .. } => vec![*from, *to],
        JournalEntry::AccountDestroyed { address, target, .. } => vec![*address, *target],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        monad_context_with_db, reserve_balance::tracker::ReserveBalanceInit, MonadBuilder,
        MonadCfgEnv, MonadChainContext, MonadContext, MonadHardfork,
    };
    use revm::{
        bytecode::opcode,
        context::TxEnv,
        context_interface::journaled_state::JournalCheckpoint,
        database::{EmptyDB, InMemoryDB},
        inspector::InspectEvm,
        interpreter::Interpreter,
        primitives::address,
        state::AccountInfo,
        Inspector,
    };
    use rstest::rstest;

    fn journal_with_account(spec: MonadHardfork, address: Address) -> MonadJournal<InMemoryDB> {
        let mut db = InMemoryDB::default();
        db.insert_account_info(address, AccountInfo::default());

        let mut journal = MonadJournal::new(db);
        journal.set_monad_spec(spec);
        journal.set_spec_id(spec.into());
        journal.load_account(address).expect("account should load");
        journal
    }

    #[derive(Debug, Default)]
    struct CallerSnapshots(Vec<Vec<Address>>);

    impl Inspector<MonadContext<InMemoryDB>> for CallerSnapshots {
        fn initialize_interp(&mut self, _: &mut Interpreter, ctx: &mut MonadContext<InMemoryDB>) {
            self.0.push(ctx.journaled_state.call_frame_senders.clone());
        }
    }

    #[rstest]
    #[case(MonadHardfork::MonadEight, false)]
    #[case(MonadHardfork::MonadNine, false)]
    #[case(MonadHardfork::MonadTen, false)]
    #[case(MonadHardfork::MonadNext, true)]
    fn caller_tracking_requires_monad_next(
        #[case] spec: MonadHardfork,
        #[case] track_callers: bool,
    ) {
        let sender = Address::repeat_byte(0x11);
        let root = Address::repeat_byte(0x22);
        let child = Address::repeat_byte(0x33);
        let mut root_code = vec![opcode::PUSH0; 5];
        root_code.push(opcode::PUSH20);
        root_code.extend_from_slice(child.as_slice());
        root_code.extend_from_slice(&[opcode::GAS, opcode::CALL, opcode::STOP]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            sender,
            AccountInfo { balance: U256::from(1_000_000), ..Default::default() },
        );
        for (address, code) in [(root, root_code), (child, vec![opcode::STOP])] {
            db.insert_account_info(
                address,
                AccountInfo::default().with_code(Bytecode::new_raw(code.into())),
            );
        }
        let ctx = monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(spec));
        let mut evm = ctx.build_monad_with_inspector(CallerSnapshots::default());
        let tx =
            TxEnv::builder().caller(sender).to(root).gas_limit(100_000).gas_price(0).build_fill();
        let result = evm.inspect_one_tx(tx).unwrap();
        assert!(result.is_success(), "{result:?}");
        let expected = if track_callers {
            vec![vec![sender], vec![sender, root]]
        } else {
            vec![vec![], vec![]]
        };
        assert_eq!(evm.0.inspector.0, expected);
    }

    #[test]
    fn call_frame_senders_survive_reverts_and_replace_siblings() {
        let mut journal = MonadJournal::new(EmptyDB::new());
        let origin = Address::repeat_byte(0x11);
        let child = Address::repeat_byte(0x22);
        let sibling = Address::repeat_byte(0x33);
        journal.set_call_frame_sender(0, origin);
        let checkpoint = journal.checkpoint();
        journal.set_call_frame_sender(1, child);
        journal.checkpoint_revert(checkpoint);
        assert_eq!(journal.call_frame_sender(0), origin);
        assert_eq!(journal.call_frame_sender(1), child);
        journal.set_call_frame_sender(1, sibling);
        assert_eq!(journal.call_frame_sender(1), sibling);
    }

    #[test]
    fn transaction_boundaries_clear_call_frame_senders() {
        let mut journal = MonadJournal::new(EmptyDB::new());
        let caller = Address::repeat_byte(0x11);
        for preserve_reserve in [false, true] {
            journal.set_preserve_reserve_balance_tracker(preserve_reserve);
            journal.set_call_frame_sender(0, caller);
            journal.commit_tx();
            assert!(journal.call_frame_senders.is_empty());
            journal.set_call_frame_sender(0, caller);
            journal.discard_tx();
            assert!(journal.call_frame_senders.is_empty());
            journal.set_call_frame_sender(0, caller);
            journal.finalize();
            assert!(journal.call_frame_senders.is_empty());
        }
    }

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

        journal.checkpoint_revert(JournalCheckpoint {
            log_i: 0,
            journal_i: 4,
            selfdestructed_i: 0,
        });

        assert!(journal.journal().is_empty());
    }

    #[test]
    fn discarded_synthetic_transaction_reverts_tracker_changes() {
        let sender = Address::from([0x11; 20]);
        let tracked = Address::from([0x22; 20]);
        let recipient = Address::from([0x33; 20]);
        let mut db = InMemoryDB::default();
        db.insert_account_info(
            tracked,
            AccountInfo { balance: U256::from(1_000), ..Default::default() },
        );

        let mut journal = MonadJournal::new(db);
        let chain = MonadChainContext::default();
        journal.reserve_balance_mut().init(ReserveBalanceInit {
            chain: &chain,
            spec: MonadHardfork::MonadNine,
            sender,
            effective_gas_price: 0,
            gas_limit: 0,
            sender_is_delegated: false,
            sender_account: None,
        });
        journal.set_preserve_reserve_balance_tracker(true);
        journal.transfer(tracked, recipient, U256::from(1)).expect("transfer should succeed");
        assert!(journal.reserve_balance().has_violation());

        journal.discard_tx();

        assert!(!journal.reserve_balance().has_violation());
        assert!(journal.reserve_balance().tracking_enabled());
    }

    #[test]
    fn preserved_root_rollback_does_not_finalize_enclosing_execution() {
        let sender = Address::repeat_byte(0x11);
        let tracked = Address::repeat_byte(0x22);
        let recipient = Address::repeat_byte(0x33);
        let mut db = InMemoryDB::default();
        db.insert_account_info(
            tracked,
            AccountInfo { balance: U256::from(1_000), ..Default::default() },
        );
        let mut journal = MonadJournal::new(db);
        journal.reserve_balance_mut().init(ReserveBalanceInit {
            chain: &MonadChainContext::default(),
            spec: MonadHardfork::MonadTen,
            sender,
            effective_gas_price: 0,
            gas_limit: 0,
            sender_is_delegated: false,
            sender_account: None,
        });
        journal.reserve_balance_mut().start_execution(1);

        for preserve in [false, true] {
            let mut parent = journal.clone();
            let mut isolated = parent.clone();
            isolated.set_preserve_reserve_balance_tracker(preserve);
            let checkpoint = isolated.checkpoint();
            assert_eq!(isolated.depth(), 1);
            assert!(isolated.transfer(tracked, recipient, U256::from(1)).unwrap().is_none());
            assert!(isolated.reserve_balance().has_violation());

            isolated.checkpoint_revert(checkpoint);
            assert!(!isolated.reserve_balance().has_violation());
            assert_eq!(isolated.state[&tracked].info.balance, U256::from(1_000));
            assert_eq!(isolated.state[&recipient].info.balance, U256::ZERO);
            *parent.reserve_balance_mut() = isolated.reserve_balance().clone();
            assert_eq!(parent.reserve_balance_mut().take_execution_violation(), !preserve);
        }
    }

    #[test]
    fn same_page_sload_is_warm_in_monad_ten() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadTen, address);

        let first = journal.sload(address, U256::ZERO).unwrap();
        let second = journal.sload(address, U256::from(127)).unwrap();

        assert!(first.is_cold);
        assert!(!second.is_cold);
    }

    #[test]
    fn same_page_sload_uses_slot_warmth_before_monad_ten() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadNine, address);

        let first = journal.sload(address, U256::ZERO).unwrap();
        let second = journal.sload(address, U256::from(127)).unwrap();

        assert!(first.is_cold);
        assert!(second.is_cold);
    }

    #[test]
    fn different_page_sload_remains_cold_in_monad_ten() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadTen, address);

        let first = journal.sload(address, U256::ZERO).unwrap();
        let second = journal.sload(address, U256::from(128)).unwrap();

        assert!(first.is_cold);
        assert!(second.is_cold);
    }

    #[test]
    fn access_list_warms_entire_page_in_monad_ten() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadTen, address);
        let mut access_list = AddressMap::default();
        access_list.insert(address, HashSet::from_iter([U256::ZERO]));

        journal.warm_access_list(access_list);

        assert!(!journal.sload(address, U256::from(127)).unwrap().is_cold);
    }

    #[test]
    fn checkpoint_revert_restores_page_warmth() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadTen, address);
        let checkpoint = journal.checkpoint();
        assert!(journal.sload(address, U256::ZERO).unwrap().is_cold);

        journal.checkpoint_revert(checkpoint);

        assert!(journal.sload(address, U256::from(1)).unwrap().is_cold);
    }

    #[test]
    fn transaction_boundary_clears_page_warmth() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadTen, address);
        assert!(journal.sload(address, U256::ZERO).unwrap().is_cold);
        journal.commit_tx();

        assert!(journal.sload(address, U256::from(1)).unwrap().is_cold);
    }

    #[test]
    fn monad_next_inherits_page_warmth_from_monad_ten() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut journal = journal_with_account(MonadHardfork::MonadNext, address);

        assert!(journal.sload(address, U256::ZERO).unwrap().is_cold);
        assert!(!journal.sload(address, U256::from(127)).unwrap().is_cold);
    }
}
