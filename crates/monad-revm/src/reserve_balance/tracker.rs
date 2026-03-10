//! Reserve-balance tracker.

use crate::{chain::MonadChainContext, staking::STAKING_ADDRESS, MonadSpecId};
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
    pub spec: MonadSpecId,
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
    chain: MonadChainContext,
    use_recent_code_hash: bool,
    sender: Address,
    sender_gas_fees: U256,
    sender_can_dip: bool,
    allow_init_selfdestruct_exemption: bool,
    violation_thresholds: HashMap<Address, Option<U256>>,
    failed: HashSet<Address>,
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
        self.chain = init.chain.clone();
        self.use_recent_code_hash = MonadSpecId::MonadEight.is_enabled_in(init.spec);
        self.sender = init.sender;
        self.allow_init_selfdestruct_exemption = MonadSpecId::MonadNine.is_enabled_in(init.spec);
        self.sender_gas_fees = U256::from(init.effective_gas_price) * U256::from(init.gas_limit);
        self.sender_can_dip = init.chain.sender_can_dip(self.sender, init.sender_is_delegated);
        self.update_loaded_account(init.sender_account, self.sender);
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
        if !self.tracking_enabled || !self.use_recent_code_hash {
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

        if self.allow_init_selfdestruct_exemption
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
        self.chain.max_reserve_balance(address).min(account.original_info.balance)
    }

    fn compute_violation_threshold(&self, account: &Account, address: Address) -> Option<U256> {
        if !self.is_subject_account(account, address) {
            return Some(U256::ZERO);
        }

        let mut reserve = self.pretx_reserve(address, account);
        if address == self.sender {
            if self.sender_can_dip {
                return Some(U256::ZERO);
            }
            reserve = reserve.checked_sub(self.sender_gas_fees)?;
        }
        Some(reserve)
    }

    fn is_subject_account(&self, account: &Account, address: Address) -> bool {
        if address == STAKING_ADDRESS {
            return false;
        }

        let effective_code_hash = if self.use_recent_code_hash {
            account.info.code_hash
        } else {
            account.original_info.code_hash
        };
        if effective_code_hash.is_zero() || effective_code_hash == KECCAK_EMPTY {
            return true;
        }

        account
            .info
            .code
            .as_ref()
            .or(account.original_info.code.as_ref())
            .is_some_and(Bytecode::is_eip7702)
    }
}

fn is_smart_contract_code(code: &Bytecode) -> bool {
    !code.original_bytes().is_empty() && !code.is_eip7702()
}
