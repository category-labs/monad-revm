use crate::{address, word};
use monad_revm::MonadChainContext;
use revm::{
    context_interface::transaction::{AccessListItem, Authorization, SignedAuthorization},
    primitives::{Address, B256},
};
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AccessEntry {
    address: String,
    storage_keys: Vec<String>,
}

impl AccessEntry {
    pub(crate) fn parse(&self) -> Result<AccessListItem, String> {
        Ok(AccessListItem {
            address: address(&self.address)?,
            storage_keys: self
                .storage_keys
                .iter()
                .map(|key| {
                    B256::from_str(key).map_err(|e| format!("Invalid access-list storage key: {e}"))
                })
                .collect::<Result<_, _>>()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AuthorizationEntry {
    chain_id: String,
    address: String,
    nonce: String,
    y_parity: u8,
    r: String,
    s: String,
}

impl AuthorizationEntry {
    pub(crate) fn parse(&self) -> Result<SignedAuthorization, String> {
        Ok(SignedAuthorization::new_unchecked(
            Authorization {
                chain_id: word(&self.chain_id)?,
                address: address(&self.address)?,
                nonce: word(&self.nonce)?
                    .try_into()
                    .map_err(|_| "Authorization nonce exceeds u64")?,
            },
            self.y_parity,
            word(&self.r)?,
            word(&self.s)?,
        ))
    }
}

/// Block participants preceding this transaction. Current authorities are recovered in Rust.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ChainContext {
    #[serde(default)]
    parent_senders_and_authorities: Vec<String>,
    #[serde(default)]
    grandparent_senders_and_authorities: Vec<String>,
    #[serde(default)]
    current_block_senders: Vec<String>,
    #[serde(default)]
    current_block_authorities: Vec<Vec<String>>,
}

impl ChainContext {
    pub(crate) fn parse(
        &self,
        sender: Address,
        authorities: Vec<Address>,
    ) -> Result<MonadChainContext, String> {
        if self.current_block_senders.len() != self.current_block_authorities.len() {
            return Err("Current block senders and authority lists must have equal lengths".into());
        }
        let mut senders = self
            .current_block_senders
            .iter()
            .map(|value| address(value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut current_authorities = self
            .current_block_authorities
            .iter()
            .map(|values| values.iter().map(|value| address(value)).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()?;
        let current_tx_index = senders.len();
        senders.push(sender);
        current_authorities.push(authorities);
        Ok(MonadChainContext {
            parent_senders_and_authorities: self
                .parent_senders_and_authorities
                .iter()
                .map(|value| address(value))
                .collect::<Result<_, _>>()?,
            grandparent_senders_and_authorities: self
                .grandparent_senders_and_authorities
                .iter()
                .map(|value| address(value))
                .collect::<Result<_, _>>()?,
            current_block_senders: senders,
            current_block_authorities: current_authorities,
            current_tx_index,
            ..Default::default()
        })
    }
}
