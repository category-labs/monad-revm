//! Helpers for paged storage access tracking.

use revm::primitives::{Address, HashMap, HashSet, StorageKey};

/// Number of bits shifted right to derive the 4KB page index from a storage slot.
pub const PAGE_SHIFT: usize = 7;

/// Number of 32-byte storage words in a single page.
pub const WORDS_PER_PAGE: u64 = 1 << PAGE_SHIFT;

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
    warm_journal: Vec<StoragePageKey>,
    checkpoint_stack: Vec<usize>,
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
            self.warm_journal.push(key);
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
        self.checkpoint_stack.push(self.warm_journal.len());
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

        while self.warm_journal.len() > journal_len {
            let page = self.warm_journal.pop().expect("warm journal length checked before pop");
            self.warmed_pages.remove(&page);
        }
    }

    /// Clears all transaction-local page access state.
    pub fn clear(&mut self) {
        self.warmed_pages.clear();
        self.warm_journal.clear();
        self.checkpoint_stack.clear();
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
}
