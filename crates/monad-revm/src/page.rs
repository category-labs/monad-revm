//! Helpers for paged storage access tracking.

use revm::{
    context_interface::context::SStoreResult,
    primitives::{Address, HashMap, HashSet, StorageKey},
};

/// Number of bits shifted right to derive the 4KB page index from a storage slot.
pub const PAGE_SHIFT: usize = 7;

/// Number of 32-byte storage words in a single page.
pub const WORDS_PER_PAGE: u64 = 1 << PAGE_SHIFT;

/// Temporary MIP-8 assumption for the SSTORE base cost.
pub const BASE_SSTORE_COST: u64 = 100;

/// Temporary MIP-8 assumption for the first page write charge in a transaction.
pub const PAGE_WRITE_COST: u64 = 2800;

/// Temporary MIP-8 assumption for net new slot growth within a page.
pub const NEW_SLOT_COST: u64 = 17100;

/// Returns the 4KB page index for a storage slot.
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
    /// Creates a storage page key from an account address and storage slot.
    #[inline]
    pub fn from_slot(address: Address, slot: StorageKey) -> Self {
        Self { address, page: page_index(slot) }
    }
}

/// Transaction-local page warmth tracker with checkpoint undo support.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageAccessTracker {
    warmed_pages: HashSet<StoragePageKey>,
    written_pages: HashSet<StoragePageKey>,
    slot_delta_counter: HashMap<StoragePageKey, i32>,
    max_nonzero_slots: HashMap<StoragePageKey, u32>,
    change_journal: Vec<PageTrackerChange>,
    checkpoint_stack: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageTrackerChange {
    Warmed(StoragePageKey),
    WriteCharged(StoragePageKey),
    SlotDelta { key: StoragePageKey, previous: i32, existed: bool },
    MaxNonZero { key: StoragePageKey, previous: u32, existed: bool },
}

impl PageAccessTracker {
    /// Returns `true` if the page is warm in the current transaction.
    #[inline]
    pub fn is_warm(&self, key: &StoragePageKey) -> bool {
        self.warmed_pages.contains(key)
    }

    /// Marks the page as warm and records the transition if it is newly warmed.
    #[inline]
    pub fn warm_page(&mut self, key: StoragePageKey) {
        if self.warmed_pages.insert(key) {
            self.change_journal.push(PageTrackerChange::Warmed(key));
        }
    }

    /// Marks the pages from an access list as warm for the whole transaction.
    pub fn warm_access_list(&mut self, access_list: &HashMap<Address, HashSet<StorageKey>>) {
        for (address, keys) in access_list {
            for key in keys {
                self.warmed_pages.insert(StoragePageKey::from_slot(*address, *key));
            }
        }
    }

    /// Records a new checkpoint.
    #[inline]
    pub fn checkpoint(&mut self) {
        self.checkpoint_stack.push(self.change_journal.len());
    }

    /// Commits the latest checkpoint.
    #[inline]
    pub fn checkpoint_commit(&mut self) {
        let _ = self.checkpoint_stack.pop();
    }

    /// Reverts the latest checkpoint.
    pub fn checkpoint_revert(&mut self) {
        let Some(journal_len) = self.checkpoint_stack.pop() else {
            return;
        };

        while self.change_journal.len() > journal_len {
            let change =
                self.change_journal.pop().expect("change journal length checked before pop");
            match change {
                PageTrackerChange::Warmed(page) => {
                    self.warmed_pages.remove(&page);
                }
                PageTrackerChange::WriteCharged(page) => {
                    self.written_pages.remove(&page);
                }
                PageTrackerChange::SlotDelta { key, previous, existed } => {
                    if existed {
                        self.slot_delta_counter.insert(key, previous);
                    } else {
                        self.slot_delta_counter.remove(&key);
                    }
                }
                PageTrackerChange::MaxNonZero { key, previous, existed } => {
                    if existed {
                        self.max_nonzero_slots.insert(key, previous);
                    } else {
                        self.max_nonzero_slots.remove(&key);
                    }
                }
            }
        }
    }

    /// Applies MIP-8 page write and per-page growth charges for an SSTORE transition.
    pub fn sstore_gas(&mut self, key: StoragePageKey, result: &SStoreResult) -> u64 {
        let mut gas = 0;

        if result.new_values_changes_present() && self.charge_page_write(key) {
            gas += PAGE_WRITE_COST;
        }

        // Creating a new slot | 0 -> 0 -> Z |
        if result.is_original_zero() && result.is_present_zero() && !result.is_new_zero() {
            gas += BASE_SSTORE_COST;
            let new_delta = self.slot_delta(key) + 1;
            self.set_slot_delta(key, new_delta);
            if new_delta > self.max_nonzero_slots(key) as i32 {
                gas += NEW_SLOT_COST;
                self.set_max_nonzero_slots(key, new_delta as u32);
            }
            return gas;
        }

        // Clear an existing slot | X -> Y -> 0 | X -> X -> 0 |
        if !result.is_original_zero() && !result.is_present_zero() && result.is_new_zero() {
            gas += BASE_SSTORE_COST;
            self.set_slot_delta(key, self.slot_delta(key) - 1);
            return gas;
        }

        // Writing zero to zero | 0 -> Y -> 0 |
        if result.is_original_zero() && !result.is_present_zero() && result.is_new_zero() {
            self.set_slot_delta(key, self.slot_delta(key) - 1);
            return gas;
        }

        // Restoring a cleared slot | X -> 0 -> Z | X -> 0 -> X |
        if !result.is_original_zero() && result.is_present_zero() && !result.is_new_zero() {
            gas += BASE_SSTORE_COST;
            self.set_slot_delta(key, self.slot_delta(key) + 1);
            return gas;
        }

        // Write Nonzero to Nonzero and remaining cases | X -> Y -> X | X -> X -> Z |
        gas + BASE_SSTORE_COST
    }

    /// Clears all transaction-local page access state.
    pub fn clear(&mut self) {
        self.warmed_pages.clear();
        self.written_pages.clear();
        self.slot_delta_counter.clear();
        self.max_nonzero_slots.clear();
        self.change_journal.clear();
        self.checkpoint_stack.clear();
    }

    #[inline]
    fn charge_page_write(&mut self, key: StoragePageKey) -> bool {
        if self.written_pages.insert(key) {
            self.change_journal.push(PageTrackerChange::WriteCharged(key));
            true
        } else {
            false
        }
    }

    #[inline]
    fn slot_delta(&self, key: StoragePageKey) -> i32 {
        self.slot_delta_counter.get(&key).copied().unwrap_or_default()
    }

    #[inline]
    fn max_nonzero_slots(&self, key: StoragePageKey) -> u32 {
        self.max_nonzero_slots.get(&key).copied().unwrap_or_default()
    }

    fn set_slot_delta(&mut self, key: StoragePageKey, new_value: i32) {
        let previous = self.slot_delta_counter.get(&key).copied();
        if previous == Some(new_value) || (previous.is_none() && new_value == 0) {
            return;
        }

        self.change_journal.push(PageTrackerChange::SlotDelta {
            key,
            previous: previous.unwrap_or_default(),
            existed: previous.is_some(),
        });

        if new_value == 0 {
            self.slot_delta_counter.remove(&key);
        } else {
            self.slot_delta_counter.insert(key, new_value);
        }
    }

    fn set_max_nonzero_slots(&mut self, key: StoragePageKey, new_value: u32) {
        let previous = self.max_nonzero_slots.get(&key).copied();
        if previous == Some(new_value) || (previous.is_none() && new_value == 0) {
            return;
        }

        self.change_journal.push(PageTrackerChange::MaxNonZero {
            key,
            previous: previous.unwrap_or_default(),
            existed: previous.is_some(),
        });

        if new_value == 0 {
            self.max_nonzero_slots.remove(&key);
        } else {
            self.max_nonzero_slots.insert(key, new_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::{address, U256};

    #[test]
    fn page_math_matches_spec() {
        let slot = U256::from(0x181u64);
        assert_eq!(page_index(slot), U256::from(3));
        assert_eq!(page_offset(slot), U256::from(0x1));
    }

    #[test]
    fn checkpoint_revert_only_unwarms_pages_added_after_checkpoint() {
        let address = address!("1234567890123456789012345678901234567890");
        let mut tracker = PageAccessTracker::default();
        let page0 = StoragePageKey::from_slot(address, U256::ZERO);
        let page1 = StoragePageKey::from_slot(address, U256::from(128));

        tracker.warm_page(page0);
        tracker.checkpoint();
        tracker.warm_page(page1);
        tracker.checkpoint_revert();

        assert!(tracker.is_warm(&page0));
        assert!(!tracker.is_warm(&page1));
    }

    #[test]
    fn first_new_slot_in_page_charges_page_write_and_growth() {
        let address = address!("1234567890123456789012345678901234567890");
        let page = StoragePageKey::from_slot(address, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        let gas = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::ZERO,
                present_value: U256::ZERO,
                new_value: U256::from(1),
            },
        );

        assert_eq!(gas, BASE_SSTORE_COST + PAGE_WRITE_COST + NEW_SLOT_COST);
    }

    #[test]
    fn second_new_slot_in_written_page_charges_only_base_and_growth() {
        let address = address!("1234567890123456789012345678901234567890");
        let page = StoragePageKey::from_slot(address, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        let _ = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::ZERO,
                present_value: U256::ZERO,
                new_value: U256::from(1),
            },
        );

        let gas = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::ZERO,
                present_value: U256::ZERO,
                new_value: U256::from(2),
            },
        );

        assert_eq!(gas, BASE_SSTORE_COST + NEW_SLOT_COST);
    }

    #[test]
    fn clean_existing_slot_first_write_on_page_charges_page_write_and_base() {
        let address = address!("1234567890123456789012345678901234567890");
        let page = StoragePageKey::from_slot(address, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        let gas = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::from(1),
                present_value: U256::from(1),
                new_value: U256::from(2),
            },
        );

        assert_eq!(gas, BASE_SSTORE_COST + PAGE_WRITE_COST);
    }

    #[test]
    fn restoring_cleared_slot_does_not_recharge_growth() {
        let address = address!("1234567890123456789012345678901234567890");
        let page = StoragePageKey::from_slot(address, U256::ZERO);
        let mut tracker = PageAccessTracker::default();

        let clear_gas = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::from(1),
                present_value: U256::from(1),
                new_value: U256::ZERO,
            },
        );
        let restore_gas = tracker.sstore_gas(
            page,
            &SStoreResult {
                original_value: U256::from(1),
                present_value: U256::ZERO,
                new_value: U256::from(3),
            },
        );

        assert_eq!(clear_gas, BASE_SSTORE_COST + PAGE_WRITE_COST);
        assert_eq!(restore_gas, BASE_SSTORE_COST);
    }
}
