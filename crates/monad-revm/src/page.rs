//! Helpers for [MIP-8] page-based storage access tracking.
//!
//! [MIP-8]: https://mips.monad.xyz/MIPs/MIP-8

use revm::{
    context_interface::context::SStoreResult,
    primitives::{Address, AddressMap, HashMap, HashSet, StorageKey},
};

/// Number of bits shifted right to derive a 4 KiB page index from a storage slot.
pub const PAGE_SHIFT: usize = 7;

/// Number of 32-byte storage words in a page.
pub const WORDS_PER_PAGE: u64 = 1 << PAGE_SHIFT;

/// Base cost charged for every storage access.
pub const BASE_COST: u64 = 100;

/// Cost charged for the first write to a page in a transaction.
pub const PAGE_WRITE_COST: u64 = 2_800;

/// Cost charged when a page reaches a new high-water mark of net state growth.
pub const STATE_GROWTH_COST: u64 = 17_000;

/// Returns the page index for a storage slot.
#[inline]
pub fn page_index(slot: StorageKey) -> StorageKey {
    slot >> PAGE_SHIFT
}

/// Returns the word offset of a storage slot within its page.
#[inline]
pub fn page_offset(slot: StorageKey) -> StorageKey {
    slot & StorageKey::from(WORDS_PER_PAGE - 1)
}

/// Transaction-local page key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StoragePageKey {
    /// Account address owning the storage page.
    pub address: Address,
    /// Page index derived from the storage slot.
    pub page: StorageKey,
}

impl StoragePageKey {
    /// Creates a page key from an account address and storage slot.
    #[inline]
    pub fn from_slot(address: Address, slot: StorageKey) -> Self {
        Self { address, page: page_index(slot) }
    }
}

/// Transaction-local page access and state-growth tracker.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageAccessTracker {
    read_accessed_pages: HashSet<StoragePageKey>,
    write_accessed_pages: HashSet<StoragePageKey>,
    current_state_growth: HashMap<StoragePageKey, i32>,
    net_state_growth: HashMap<StoragePageKey, i32>,
    change_journal: alloc::vec::Vec<PageTrackerChange>,
    checkpoint_stack: alloc::vec::Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageTrackerChange {
    ReadAccessed(StoragePageKey),
    WriteAccessed(StoragePageKey),
    CurrentStateGrowth { key: StoragePageKey, previous: i32, existed: bool },
    NetStateGrowth { key: StoragePageKey, previous: i32, existed: bool },
}

impl PageAccessTracker {
    /// Returns `true` if a page has been read during the current transaction.
    #[inline]
    pub fn is_read_accessed(&self, key: &StoragePageKey) -> bool {
        self.read_accessed_pages.contains(key)
    }

    /// Marks a page as read and records the transition for checkpoint reverts.
    #[inline]
    pub fn mark_read_accessed(&mut self, key: StoragePageKey) {
        if self.read_accessed_pages.insert(key) {
            self.change_journal.push(PageTrackerChange::ReadAccessed(key));
        }
    }

    /// Marks every page represented by an access list as read-accessed.
    pub fn warm_access_list(&mut self, access_list: &AddressMap<HashSet<StorageKey>>) {
        for (address, keys) in access_list {
            for key in keys {
                self.read_accessed_pages.insert(StoragePageKey::from_slot(*address, *key));
            }
        }
    }

    /// Records a new call-frame checkpoint.
    #[inline]
    pub fn checkpoint(&mut self) {
        self.checkpoint_stack.push(self.change_journal.len());
    }

    /// Commits the latest call-frame checkpoint.
    #[inline]
    pub fn checkpoint_commit(&mut self) {
        let _ = self.checkpoint_stack.pop();
    }

    /// Reverts all page tracker changes made since the latest checkpoint.
    pub fn checkpoint_revert(&mut self) {
        let Some(journal_len) = self.checkpoint_stack.pop() else {
            return;
        };

        while self.change_journal.len() > journal_len {
            let change = self.change_journal.pop().expect("change journal length checked");
            match change {
                PageTrackerChange::ReadAccessed(key) => {
                    self.read_accessed_pages.remove(&key);
                }
                PageTrackerChange::WriteAccessed(key) => {
                    self.write_accessed_pages.remove(&key);
                }
                PageTrackerChange::CurrentStateGrowth { key, previous, existed } => {
                    restore_counter(&mut self.current_state_growth, key, previous, existed);
                }
                PageTrackerChange::NetStateGrowth { key, previous, existed } => {
                    restore_counter(&mut self.net_state_growth, key, previous, existed);
                }
            }
        }
    }

    /// Returns the MIP-8 dynamic SSTORE cost for a storage transition.
    pub fn sstore_gas(&mut self, key: StoragePageKey, result: &SStoreResult) -> u64 {
        let mut gas = BASE_COST;

        if result.new_values_changes_present() && self.mark_write_accessed(key) {
            gas += PAGE_WRITE_COST;
        }

        let growth_delta = match (result.is_present_zero(), result.is_new_zero()) {
            (true, false) => 1,
            (false, true) => -1,
            _ => 0,
        };
        if growth_delta == 0 {
            return gas;
        }

        let current_growth = self.current_state_growth(key) + growth_delta;
        self.set_current_state_growth(key, current_growth);

        if current_growth > self.net_state_growth(key) {
            gas += STATE_GROWTH_COST;
            self.set_net_state_growth(key, current_growth);
        }

        gas
    }

    /// Clears all transaction-local page state.
    pub fn clear(&mut self) {
        self.read_accessed_pages.clear();
        self.write_accessed_pages.clear();
        self.current_state_growth.clear();
        self.net_state_growth.clear();
        self.change_journal.clear();
        self.checkpoint_stack.clear();
    }

    #[inline]
    fn mark_write_accessed(&mut self, key: StoragePageKey) -> bool {
        if self.write_accessed_pages.insert(key) {
            self.change_journal.push(PageTrackerChange::WriteAccessed(key));
            true
        } else {
            false
        }
    }

    #[inline]
    fn current_state_growth(&self, key: StoragePageKey) -> i32 {
        self.current_state_growth.get(&key).copied().unwrap_or_default()
    }

    #[inline]
    fn net_state_growth(&self, key: StoragePageKey) -> i32 {
        self.net_state_growth.get(&key).copied().unwrap_or_default()
    }

    fn set_current_state_growth(&mut self, key: StoragePageKey, value: i32) {
        let previous = self.current_state_growth.get(&key).copied();
        if previous == Some(value) || (previous.is_none() && value == 0) {
            return;
        }
        self.change_journal.push(PageTrackerChange::CurrentStateGrowth {
            key,
            previous: previous.unwrap_or_default(),
            existed: previous.is_some(),
        });
        set_counter(&mut self.current_state_growth, key, value);
    }

    fn set_net_state_growth(&mut self, key: StoragePageKey, value: i32) {
        let previous = self.net_state_growth.get(&key).copied();
        if previous == Some(value) || (previous.is_none() && value == 0) {
            return;
        }
        self.change_journal.push(PageTrackerChange::NetStateGrowth {
            key,
            previous: previous.unwrap_or_default(),
            existed: previous.is_some(),
        });
        set_counter(&mut self.net_state_growth, key, value);
    }
}

fn set_counter(counters: &mut HashMap<StoragePageKey, i32>, key: StoragePageKey, value: i32) {
    if value == 0 {
        counters.remove(&key);
    } else {
        counters.insert(key, value);
    }
}

fn restore_counter(
    counters: &mut HashMap<StoragePageKey, i32>,
    key: StoragePageKey,
    previous: i32,
    existed: bool,
) {
    if existed {
        counters.insert(key, previous);
    } else {
        counters.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::{address, U256};

    fn transition(original: u64, present: u64, new: u64) -> SStoreResult {
        SStoreResult {
            original_value: U256::from(original),
            present_value: U256::from(present),
            new_value: U256::from(new),
        }
    }

    #[test]
    fn page_math_matches_spec() {
        let slot = U256::from(0x181);
        assert_eq!(page_index(slot), U256::from(3));
        assert_eq!(page_offset(slot), U256::from(1));
    }

    #[test]
    fn every_sstore_charges_base_cost() {
        let key = StoragePageKey::from_slot(Address::ZERO, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        assert_eq!(tracker.sstore_gas(key, &transition(0, 0, 0)), BASE_COST);
        assert_eq!(tracker.sstore_gas(key, &transition(0, 1, 0)), BASE_COST + PAGE_WRITE_COST);
    }

    #[test]
    fn first_new_slot_charges_write_and_growth() {
        let key = StoragePageKey::from_slot(Address::ZERO, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        assert_eq!(
            tracker.sstore_gas(key, &transition(0, 0, 1)),
            BASE_COST + PAGE_WRITE_COST + STATE_GROWTH_COST
        );
    }

    #[test]
    fn clearing_existing_slot_offsets_new_growth_in_same_page() {
        let key = StoragePageKey::from_slot(Address::ZERO, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        assert_eq!(tracker.sstore_gas(key, &transition(1, 1, 0)), BASE_COST + PAGE_WRITE_COST);
        assert_eq!(tracker.sstore_gas(key, &transition(0, 0, 1)), BASE_COST);
    }

    #[test]
    fn restoring_new_slot_does_not_recharge_growth() {
        let key = StoragePageKey::from_slot(Address::ZERO, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        let _ = tracker.sstore_gas(key, &transition(0, 0, 1));
        assert_eq!(tracker.sstore_gas(key, &transition(0, 1, 0)), BASE_COST);
        assert_eq!(tracker.sstore_gas(key, &transition(0, 0, 2)), BASE_COST);
    }

    #[test]
    fn checkpoint_revert_restores_all_page_state() {
        let address = address!("1234567890123456789012345678901234567890");
        let key = StoragePageKey::from_slot(address, U256::ZERO);
        let mut tracker = PageAccessTracker::default();
        tracker.checkpoint();
        tracker.mark_read_accessed(key);
        let _ = tracker.sstore_gas(key, &transition(0, 0, 1));

        tracker.checkpoint_revert();

        assert!(!tracker.is_read_accessed(&key));
        assert_eq!(
            tracker.sstore_gas(key, &transition(0, 0, 1)),
            BASE_COST + PAGE_WRITE_COST + STATE_GROWTH_COST
        );
    }
}
