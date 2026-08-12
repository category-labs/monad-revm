//! Reserve-balance tracker.

use crate::{chain::MonadChainContext, staking::STAKING_ADDRESS, MonadHardfork};
use revm::{
    bytecode::Bytecode,
    primitives::{Address, HashMap, HashSet, KECCAK_EMPTY, U256},
    state::Account,
};

/// Input data used to initialize the reserve-balance tracker for a transaction.
#[derive(Clone, Copy, Debug)]
pub struct ReserveBalanceInit<'a> {
    /// Monad chain metadata for sender-dip checks and reserve policy.
    pub chain: &'a MonadChainContext,
    /// Active Monad hardfork.
    pub spec: MonadHardfork,
    /// Transaction sender.
    pub sender: Address,
    /// Effective gas price used to charge the transaction.
    pub effective_gas_price: u128,
    /// Transaction gas limit.
    pub gas_limit: u64,
    /// Whether the sender is delegated.
    pub sender_is_delegated: bool,
    /// Optional loaded sender account.
    pub sender_account: Option<&'a Account>,
}

/// Cached reserve-balance state for the current transaction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReserveBalanceTracker {
    tracking_enabled: bool,
    transaction: ReserveBalanceTransactionContext,
    policy: ReserveBalancePolicy,
    violation_thresholds: HashMap<Address, Option<U256>>,
    failed: HashSet<Address>,
}

/// Transaction-scoped inputs that remain stable across frame hardfork changes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ReserveBalanceTransactionContext {
    chain: MonadChainContext,
    sender: Address,
    sender_gas_fees: U256,
    sender_is_delegated: bool,
    sender_can_dip: bool,
}

/// Hardfork-dependent rules used to evaluate reserve-balance violations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReserveBalancePolicy {
    subject_code: SubjectCodePolicy,
    init_selfdestruct: InitSelfdestructPolicy,
}

impl ReserveBalancePolicy {
    /// Derives reserve-balance behavior exhaustively for a Monad hardfork.
    const fn for_spec(spec: MonadHardfork) -> Self {
        match spec {
            MonadHardfork::MonadEight => Self {
                subject_code: SubjectCodePolicy::Current,
                init_selfdestruct: InitSelfdestructPolicy::EnforceReserve,
            },
            MonadHardfork::MonadNine | MonadHardfork::MonadNext => Self {
                subject_code: SubjectCodePolicy::Current,
                init_selfdestruct: InitSelfdestructPolicy::Exempt,
            },
        }
    }
}

/// Account code version used to determine whether reserve balance applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SubjectCodePolicy {
    /// Use the code present before the transaction.
    #[default]
    Original,
    /// Use the most recently journaled code.
    Current,
}

/// Treatment of contracts created and self-destructed in the same transaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum InitSelfdestructPolicy {
    /// Enforce the normal reserve requirement.
    #[default]
    EnforceReserve,
    /// Exempt the account from the reserve requirement.
    Exempt,
}

impl ReserveBalanceTracker {
    /// Returns true if tracking is enabled.
    pub const fn tracking_enabled(&self) -> bool {
        self.tracking_enabled
    }

    /// Returns true if the transaction is currently violating reserve balance.
    pub fn has_violation(&self) -> bool {
        !self.failed.is_empty()
    }

    /// Clears all cached state.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Initializes the tracker for a new transaction.
    pub fn init(&mut self, init: ReserveBalanceInit<'_>) {
        self.clear();
        self.tracking_enabled = true;
        self.transaction = ReserveBalanceTransactionContext {
            chain: init.chain.clone(),
            sender: init.sender,
            sender_gas_fees: U256::from(init.effective_gas_price) * U256::from(init.gas_limit),
            sender_is_delegated: init.sender_is_delegated,
            sender_can_dip: init.chain.sender_can_dip(init.sender, init.sender_is_delegated),
        };
        self.policy = ReserveBalancePolicy::for_spec(init.spec);
        self.update_loaded_account(init.sender_account, init.sender);
    }

    /// Rebases tracked accounts onto replacement journal state and chain metadata.
    ///
    /// This preserves the enclosing transaction's sender and gas invariants while discarding
    /// cached thresholds derived from the previous state. Accounts that were merely loaded but
    /// never affected by reserve tracking remain untracked.
    pub fn rebase(&mut self, chain: &MonadChainContext, state: &revm::state::EvmState) {
        if !self.tracking_enabled {
            return;
        }

        self.transaction.chain = chain.clone();
        self.transaction.sender_can_dip =
            chain.sender_can_dip(self.transaction.sender, self.transaction.sender_is_delegated);
        self.recompute_tracked_accounts(state);
    }

    /// Reconfigures hardfork-dependent reserve policy for the active frame.
    ///
    /// This preserves transaction-scoped sender, gas, and chain invariants while recomputing
    /// accounts already affected by reserve tracking under the selected hardfork.
    pub fn reconfigure(&mut self, spec: MonadHardfork, state: &revm::state::EvmState) {
        if !self.tracking_enabled {
            return;
        }

        let policy = ReserveBalancePolicy::for_spec(spec);
        if self.policy == policy {
            return;
        }

        self.policy = policy;
        self.recompute_tracked_accounts(state);
    }

    fn recompute_tracked_accounts(&mut self, state: &revm::state::EvmState) {
        let tracked = core::mem::take(&mut self.violation_thresholds);
        self.failed.clear();

        for address in tracked.into_keys() {
            self.update_loaded_account(state.get(&address), address);
        }
    }

    /// Recomputes the violation status of an address after a debit.
    pub fn on_debit(&mut self, account: Option<&Account>, address: Address) {
        self.update_loaded_account(account, address);
    }

    /// Recomputes the violation status of an address after a credit if it was failing.
    pub fn on_credit(&mut self, account: Option<&Account>, address: Address) {
        if self.failed.contains(&address) {
            self.update_loaded_account(account, address);
        }
    }

    /// Recomputes the violation status of an address after code changes.
    pub fn on_set_code(&mut self, account: Option<&Account>, address: Address, code: &Bytecode) {
        if !self.tracking_enabled || self.policy.subject_code != SubjectCodePolicy::Current {
            return;
        }

        if is_smart_contract_code(code) {
            self.violation_thresholds.insert(address, Some(U256::ZERO));
            self.failed.remove(&address);
            return;
        }

        self.violation_thresholds.remove(&address);
        self.update_loaded_account(account, address);
    }

    /// Recomputes violation status for reverted addresses.
    pub fn on_checkpoint_revert<I>(&mut self, reverted_addresses: I, state: &revm::state::EvmState)
    where
        I: IntoIterator<Item = Address>,
    {
        if !self.tracking_enabled {
            return;
        }

        for address in reverted_addresses {
            self.violation_thresholds.remove(&address);
            self.update_loaded_account(state.get(&address), address);
        }
    }

    fn update_loaded_account(&mut self, account: Option<&Account>, address: Address) {
        if !self.tracking_enabled {
            return;
        }

        let Some(account) = account else {
            self.failed.remove(&address);
            self.violation_thresholds.remove(&address);
            return;
        };

        if self.policy.init_selfdestruct == InitSelfdestructPolicy::Exempt
            && account.is_selfdestructed()
            && account.is_created_locally()
        {
            self.failed.remove(&address);
            self.violation_thresholds.insert(address, Some(U256::ZERO));
            return;
        }

        let threshold = match self.violation_thresholds.get(&address).copied() {
            Some(threshold) => threshold,
            None => {
                let threshold = self.compute_violation_threshold(account, address);
                self.violation_thresholds.insert(address, threshold);
                threshold
            }
        };

        let Some(threshold) = threshold else {
            self.failed.insert(address);
            return;
        };

        if threshold.is_zero() || account.info.balance >= threshold {
            self.failed.remove(&address);
        } else {
            self.failed.insert(address);
        }
    }

    fn pretx_reserve(&self, address: Address, account: &Account) -> U256 {
        self.transaction.chain.max_reserve_balance(address).min(account.original_info().balance)
    }

    fn compute_violation_threshold(&self, account: &Account, address: Address) -> Option<U256> {
        if !self.is_subject_account(account, address) {
            return Some(U256::ZERO);
        }

        let mut reserve = self.pretx_reserve(address, account);
        if address == self.transaction.sender {
            if self.transaction.sender_can_dip {
                return Some(U256::ZERO);
            }
            reserve = reserve.checked_sub(self.transaction.sender_gas_fees)?;
        }
        Some(reserve)
    }

    fn is_subject_account(&self, account: &Account, address: Address) -> bool {
        if address == STAKING_ADDRESS {
            return false;
        }

        let effective_code_hash = match self.policy.subject_code {
            SubjectCodePolicy::Original => account.original_info().code_hash,
            SubjectCodePolicy::Current => account.info.code_hash,
        };
        if effective_code_hash.is_zero() || effective_code_hash == KECCAK_EMPTY {
            return true;
        }

        account
            .info
            .code
            .as_ref()
            .or(account.original_info().code.as_ref())
            .is_some_and(Bytecode::is_eip7702)
    }
}

fn is_smart_contract_code(code: &Bytecode) -> bool {
    !code.original_bytes().is_empty() && !code.is_eip7702()
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::state::{AccountInfo, EvmState};

    fn debited_account(original: u64, current: u64) -> Account {
        let mut account =
            Account::from(AccountInfo { balance: U256::from(original), ..Default::default() });
        account.info.balance = U256::from(current);
        account
    }

    fn sender_chain(sender: Address) -> MonadChainContext {
        MonadChainContext {
            parent_senders_and_authorities: [sender].into_iter().collect(),
            ..Default::default()
        }
    }

    fn init_tracker(
        tracker: &mut ReserveBalanceTracker,
        chain: &MonadChainContext,
        sender: Address,
        account: &Account,
        delegated: bool,
    ) {
        init_tracker_at(
            tracker,
            chain,
            MonadHardfork::MonadNine,
            sender,
            account,
            delegated,
            (0, 0),
        );
    }

    fn init_tracker_with_fees(
        tracker: &mut ReserveBalanceTracker,
        chain: &MonadChainContext,
        sender: Address,
        account: &Account,
        delegated: bool,
        effective_gas_price: u128,
        gas_limit: u64,
    ) {
        init_tracker_at(
            tracker,
            chain,
            MonadHardfork::MonadNine,
            sender,
            account,
            delegated,
            (effective_gas_price, gas_limit),
        );
    }

    fn init_tracker_at(
        tracker: &mut ReserveBalanceTracker,
        chain: &MonadChainContext,
        spec: MonadHardfork,
        sender: Address,
        account: &Account,
        delegated: bool,
        gas: (u128, u64),
    ) {
        let (effective_gas_price, gas_limit) = gas;
        tracker.init(ReserveBalanceInit {
            chain,
            spec,
            sender,
            effective_gas_price,
            gas_limit,
            sender_is_delegated: delegated,
            sender_account: Some(account),
        });
    }

    #[test]
    fn rebase_updates_sender_eligibility_in_both_directions() {
        let sender = Address::with_last_byte(1);
        let account = debited_account(12, 9);
        let fresh_chain = MonadChainContext::default();
        let restricted_chain = sender_chain(sender);
        let state = EvmState::from_iter([(sender, account.clone())]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &fresh_chain, sender, &account, false);
        assert!(!tracker.has_violation());

        tracker.rebase(&restricted_chain, &state);
        assert!(tracker.has_violation());

        tracker.rebase(&fresh_chain, &state);
        assert!(!tracker.has_violation());
    }

    #[test]
    fn rebase_preserves_delegated_sender_restriction() {
        let sender = Address::with_last_byte(1);
        let account = debited_account(12, 9);
        let fresh_chain = MonadChainContext::default();
        let state = EvmState::from_iter([(sender, account.clone())]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &fresh_chain, sender, &account, true);
        assert!(tracker.has_violation());

        tracker.rebase(&fresh_chain, &state);
        assert!(tracker.has_violation());
    }

    #[test]
    fn rebase_preserves_sender_gas_fee_allowance() {
        let sender = Address::with_last_byte(1);
        let account = debited_account(12, 11);
        let chain = sender_chain(sender);
        let state = EvmState::from_iter([(sender, account.clone())]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker_with_fees(&mut tracker, &chain, sender, &account, false, 1, 2);
        assert!(!tracker.has_violation());

        tracker.rebase(&chain, &state);
        assert!(!tracker.has_violation());
    }

    #[test]
    fn rebase_drops_accounts_absent_from_replacement_state() {
        let sender = Address::with_last_byte(1);
        let tracked = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let tracked_account = debited_account(12, 9);
        let chain = sender_chain(sender);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &chain, sender, &sender_account, false);
        tracker.on_debit(Some(&tracked_account), tracked);
        assert!(tracker.has_violation());

        let state = EvmState::from_iter([(sender, sender_account)]);
        tracker.rebase(&chain, &state);
        assert!(!tracker.has_violation());
    }

    #[test]
    fn rebase_preserves_tracked_violation_in_replacement_state() {
        let sender = Address::with_last_byte(1);
        let tracked = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let tracked_account = debited_account(12, 9);
        let chain = sender_chain(sender);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &chain, sender, &sender_account, false);
        tracker.on_debit(Some(&tracked_account), tracked);
        assert!(tracker.has_violation());

        let state = EvmState::from_iter([(sender, sender_account), (tracked, tracked_account)]);
        tracker.rebase(&chain, &state);
        assert!(tracker.has_violation());
    }

    #[test]
    fn rebase_recomputes_thresholds_from_replacement_original_state() {
        let sender = Address::with_last_byte(1);
        let tracked = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let tracked_account = debited_account(12, 9);
        let replacement_account = debited_account(8, 8);
        let chain = sender_chain(sender);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &chain, sender, &sender_account, false);
        tracker.on_debit(Some(&tracked_account), tracked);
        assert!(tracker.has_violation());

        let state = EvmState::from_iter([(sender, sender_account), (tracked, replacement_account)]);
        tracker.rebase(&chain, &state);
        assert!(!tracker.has_violation());
    }

    #[test]
    fn rebase_does_not_track_unaffected_loaded_accounts() {
        let sender = Address::with_last_byte(1);
        let unrelated = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let unrelated_account = debited_account(12, 9);
        let chain = MonadChainContext::default();
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker(&mut tracker, &chain, sender, &sender_account, false);
        let state =
            EvmState::from_iter([(sender, sender_account), (unrelated, unrelated_account.clone())]);
        tracker.rebase(&chain, &state);
        assert!(!tracker.has_violation());

        tracker.on_debit(Some(&unrelated_account), unrelated);
        assert!(tracker.has_violation());
    }

    #[test]
    fn rebase_is_a_noop_when_tracking_is_disabled() {
        let mut tracker = ReserveBalanceTracker::default();
        let chain = sender_chain(Address::with_last_byte(1));
        tracker.rebase(&chain, &EvmState::default());
        assert_eq!(tracker, ReserveBalanceTracker::default());
    }

    #[test]
    fn reconfigure_updates_init_selfdestruct_policy_in_both_directions() {
        let sender = Address::with_last_byte(1);
        let tracked = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let mut tracked_account = debited_account(12, 9);
        tracked_account.mark_created_locally();
        tracked_account.mark_selfdestructed_locally();
        let chain = sender_chain(sender);
        let state = EvmState::from_iter([
            (sender, sender_account.clone()),
            (tracked, tracked_account.clone()),
        ]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker_at(
            &mut tracker,
            &chain,
            MonadHardfork::MonadEight,
            sender,
            &sender_account,
            false,
            (1, 2),
        );
        tracker.on_debit(Some(&tracked_account), tracked);
        assert!(tracker.has_violation());
        let transaction = tracker.transaction.clone();

        tracker.reconfigure(MonadHardfork::MonadNine, &state);
        assert!(!tracker.has_violation());
        assert_eq!(tracker.transaction, transaction);

        tracker.reconfigure(MonadHardfork::MonadEight, &state);
        assert!(tracker.has_violation());
        assert_eq!(tracker.transaction, transaction);

        tracker.reconfigure(MonadHardfork::MonadNine, &state);
        assert!(!tracker.has_violation());
        assert_eq!(tracker.transaction, transaction);
    }

    #[test]
    fn reserve_policy_is_exhaustive_for_supported_hardforks() {
        let monad_eight = ReserveBalancePolicy::for_spec(MonadHardfork::MonadEight);
        assert_eq!(monad_eight.subject_code, SubjectCodePolicy::Current);
        assert_eq!(monad_eight.init_selfdestruct, InitSelfdestructPolicy::EnforceReserve);

        let monad_nine = ReserveBalancePolicy::for_spec(MonadHardfork::MonadNine);
        assert_eq!(monad_nine.subject_code, SubjectCodePolicy::Current);
        assert_eq!(monad_nine.init_selfdestruct, InitSelfdestructPolicy::Exempt);
        assert_eq!(ReserveBalancePolicy::for_spec(MonadHardfork::MonadNext), monad_nine);
    }

    #[test]
    fn reconfigure_is_a_noop_for_hardforks_with_the_same_policy() {
        let sender = Address::with_last_byte(1);
        let tracked = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let tracked_account = debited_account(12, 9);
        let chain = sender_chain(sender);
        let state = EvmState::from_iter([
            (sender, sender_account.clone()),
            (tracked, tracked_account.clone()),
        ]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker_at(
            &mut tracker,
            &chain,
            MonadHardfork::MonadNine,
            sender,
            &sender_account,
            false,
            (1, 2),
        );
        tracker.on_debit(Some(&tracked_account), tracked);
        let before = tracker.clone();

        tracker.reconfigure(MonadHardfork::MonadNext, &state);
        assert_eq!(tracker, before);
    }

    #[test]
    fn reconfigure_does_not_enroll_untracked_accounts() {
        let sender = Address::with_last_byte(1);
        let unrelated = Address::with_last_byte(2);
        let sender_account = debited_account(12, 12);
        let unrelated_account = debited_account(12, 9);
        let chain = sender_chain(sender);
        let state = EvmState::from_iter([
            (sender, sender_account.clone()),
            (unrelated, unrelated_account.clone()),
        ]);
        let mut tracker = ReserveBalanceTracker::default();

        init_tracker_at(
            &mut tracker,
            &chain,
            MonadHardfork::MonadEight,
            sender,
            &sender_account,
            false,
            (0, 0),
        );
        tracker.reconfigure(MonadHardfork::MonadNine, &state);
        assert!(!tracker.has_violation());

        tracker.on_debit(Some(&unrelated_account), unrelated);
        assert!(tracker.has_violation());
    }

    #[test]
    fn reconfigure_is_a_noop_when_tracking_is_disabled() {
        let mut tracker = ReserveBalanceTracker::default();
        tracker.reconfigure(MonadHardfork::MonadNine, &EvmState::default());
        assert_eq!(tracker, ReserveBalanceTracker::default());
    }
}
