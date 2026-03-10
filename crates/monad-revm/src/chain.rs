//! Monad-specific chain context.

use revm::primitives::{Address, HashSet, U256};

/// Default max reserve balance: 10 MON.
pub const fn default_max_reserve_balance() -> U256 {
    U256::from_limbs([10_000_000_000_000_000_000u64, 0, 0, 0])
}

/// Monad chain context needed for reserve-balance decisions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonadChainContext {
    /// Combined senders and authorities from the parent block.
    pub parent_senders_and_authorities: HashSet<Address>,
    /// Combined senders and authorities from the grandparent block.
    pub grandparent_senders_and_authorities: HashSet<Address>,
    /// Senders already present in the current block.
    pub current_block_senders: Vec<Address>,
    /// Authorities already present in the current block.
    pub current_block_authorities: Vec<Vec<Address>>,
    /// Index of the current transaction within the current block metadata.
    pub current_tx_index: usize,
    /// Maximum reserve balance to enforce for accounts.
    pub max_reserve_balance: U256,
}

impl Default for MonadChainContext {
    fn default() -> Self {
        Self {
            parent_senders_and_authorities: HashSet::default(),
            grandparent_senders_and_authorities: HashSet::default(),
            current_block_senders: Vec::new(),
            current_block_authorities: Vec::new(),
            current_tx_index: 0,
            max_reserve_balance: default_max_reserve_balance(),
        }
    }
}

impl MonadChainContext {
    /// Returns the maximum reserve balance for the address.
    #[inline]
    pub const fn max_reserve_balance(&self, _address: Address) -> U256 {
        self.max_reserve_balance
    }

    /// Returns true if the sender may dip into reserve.
    pub fn sender_can_dip(&self, sender: Address, sender_is_delegated: bool) -> bool {
        if sender_is_delegated {
            return false;
        }

        if self.grandparent_senders_and_authorities.contains(&sender)
            || self.parent_senders_and_authorities.contains(&sender)
        {
            return false;
        }

        for (index, block_sender) in self.current_block_senders.iter().enumerate() {
            if index < self.current_tx_index && *block_sender == sender {
                return false;
            }
        }

        for (index, authorities) in self.current_block_authorities.iter().enumerate() {
            if index <= self.current_tx_index && authorities.contains(&sender) {
                return false;
            }
        }

        true
    }
}
