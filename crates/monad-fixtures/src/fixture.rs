//! Fixture parsing and explicit capability checks.

use alloy_consensus::{Block, Header, TxEnvelope};
use eyre::{ensure, eyre, Result, WrapErr};
use monad_revm::MonadHardfork;
use revm::primitives::{hex, Address, Bytes, FixedBytes, B256, U256};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug)]
pub(crate) enum FixtureError {
    Unsupported(String),
    Failed(eyre::Report),
}

impl From<eyre::Report> for FixtureError {
    fn from(error: eyre::Report) -> Self {
        Self::Failed(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Account {
    pub nonce: u64,
    pub balance: U256,
    pub code: Bytes,
    pub storage: BTreeMap<U256, U256>,
}

pub(crate) type Allocation = BTreeMap<Address, Account>;

#[derive(Debug)]
pub(crate) struct Network {
    before: MonadHardfork,
    after: MonadHardfork,
    transition: u64,
}

impl Network {
    fn parse(name: &str) -> Result<Self, FixtureError> {
        fn fork(name: &str) -> Result<MonadHardfork, FixtureError> {
            match name {
                "MONAD_EIGHT" => Ok(MonadHardfork::MonadEight),
                "MONAD_NINE" => Ok(MonadHardfork::MonadNine),
                _ => Err(FixtureError::Unsupported(format!("network {name}: only MONAD_EIGHT and MONAD_NINE execution commitments are supported"))),
            }
        }
        if let Some((before, rest)) = name.split_once("To") {
            let Some((after, time)) = rest.split_once("AtTime") else {
                return Err(FixtureError::Unsupported(format!(
                    "unsupported fork transition {name}"
                )));
            };
            let transition = if let Some(seconds) = time.strip_suffix('k') {
                seconds.parse::<u64>().ok().and_then(|n| n.checked_mul(1000))
            } else {
                time.parse().ok()
            }
            .ok_or_else(|| eyre!("invalid transition timestamp in {name}"))?;
            Ok(Self { before: fork(before)?, after: fork(after)?, transition })
        } else {
            let spec = fork(name)?;
            Ok(Self { before: spec, after: spec, transition: 0 })
        }
    }

    pub(crate) const fn at(&self, timestamp: u64) -> MonadHardfork {
        if timestamp < self.transition {
            self.before
        } else {
            self.after
        }
    }
}

#[derive(Debug)]
pub(crate) struct Fixture {
    pub network: Network,
    pub chain_id: u64,
    pub genesis: Header,
    pub blocks: Vec<Block<TxEnvelope>>,
    pub pre: Allocation,
    pub post: Option<Allocation>,
    pub post_hash: Option<B256>,
    pub last_block_hash: B256,
}

pub(crate) fn network_name(value: &Value) -> Option<&str> {
    value.pointer("/config/network").or_else(|| value.get("network")).and_then(Value::as_str)
}

impl Fixture {
    pub(crate) fn parse(value: &Value) -> Result<Self, FixtureError> {
        if let Some(format) = value.pointer("/_info/fixture-format").and_then(Value::as_str) {
            if format != "blockchain_test" {
                return Err(FixtureError::Unsupported(format!("fixture format {format}")));
            }
        }
        if value.get("env").is_some() && value.get("transaction").is_some() {
            return Err(FixtureError::Unsupported(
                "state-test format; expected blockchain fixtures".into(),
            ));
        }
        check_fields(
            value,
            &[
                "_info",
                "network",
                "config",
                "blocks",
                "genesisBlockHeader",
                "genesisRLP",
                "lastblockhash",
                "pre",
                "postState",
                "post",
                "postStateHash",
                "sealEngine",
            ],
            "fixture",
        )?;
        let config = required(value, "config")?;
        check_fields(config, &["network", "chainid", "blobSchedule"], "config")?;
        // With zero excess blob gas the fee is one for every positive update fraction.
        // Target/max blob counts cannot affect execution because blob transactions are rejected.
        if let Some(schedule) = config.get("blobSchedule") {
            let entries =
                schedule.as_object().ok_or_else(|| eyre!("blobSchedule must be an object"))?;
            for parameters in entries.values() {
                check_fields(
                    parameters,
                    &["target", "max", "baseFeeUpdateFraction"],
                    "blobSchedule",
                )?;
                let target = quantity_u64(required(parameters, "target")?)?;
                let max = quantity_u64(required(parameters, "max")?)?;
                let fraction = quantity_u64(required(parameters, "baseFeeUpdateFraction")?)?;
                if target > max || fraction == 0 {
                    return Err(eyre!("invalid blobSchedule parameters").into());
                }
            }
        }
        if value.get("sealEngine").is_some_and(|engine| engine != "NoProof") {
            return Err(FixtureError::Unsupported("seal engine other than NoProof".into()));
        }
        let name = network_name(value).ok_or_else(|| eyre!("missing fixture network"))?;
        let network = Network::parse(name)?;
        if let Some(legacy_network) = value.get("network") {
            if legacy_network.as_str() != Some(name) {
                return Err(eyre!("network and config.network disagree").into());
            }
        }
        let blocks = required(value, "blocks")?
            .as_array()
            .ok_or_else(|| eyre!("blocks must be an array"))?;
        if blocks.iter().any(|block| block.get("expectException").is_some_and(|v| !v.is_null())) {
            return Err(FixtureError::Unsupported(
                "invalid-block expectations require consensus validation and exact error matching"
                    .into(),
            ));
        }
        let chain_id = quantity_u64(required(config, "chainid")?)?;
        let genesis = header(required(value, "genesisBlockHeader")?)?;
        if genesis.number != 0 {
            return Err(FixtureError::Unsupported("genesis above block zero requires historical block hashes and reserve-balance metadata".into()));
        }
        let genesis_rlp = bytes(required(value, "genesisRLP")?)?;
        let decoded_genesis = alloy_rlp::decode_exact::<Header>(&genesis_rlp)
            .or_else(|_| {
                alloy_rlp::decode_exact::<Block<TxEnvelope>>(&genesis_rlp).and_then(|block| {
                    if !block.body.transactions.is_empty()
                        || !block.body.ommers.is_empty()
                        || block.body.withdrawals.as_ref().is_some_and(|w| !w.is_empty())
                    {
                        return Err(alloy_rlp::Error::Custom("genesis block body must be empty"));
                    }
                    Ok(block.header)
                })
            })
            .wrap_err("decoding genesisRLP")?;
        if decoded_genesis != genesis {
            return Err(eyre!("genesisBlockHeader differs from genesisRLP").into());
        }

        let mut parsed_blocks = Vec::with_capacity(blocks.len());
        for (index, value) in blocks.iter().enumerate() {
            check_fields(
                value,
                &[
                    "blockHeader",
                    "rlp",
                    "transactions",
                    "uncleHeaders",
                    "withdrawals",
                    "expectException",
                ],
                "block",
            )?;
            let expected_header = header(required(value, "blockHeader")?)?;
            let rlp = bytes(required(value, "rlp")?)?;
            let block = alloy_rlp::decode_exact::<Block<TxEnvelope>>(&rlp)
                .wrap_err_with(|| format!("block {index}: decoding RLP"))?;
            if block.header != expected_header {
                return Err(eyre!("block {index}: blockHeader differs from RLP").into());
            }
            if !block.body.ommers.is_empty() {
                return Err(FixtureError::Unsupported(format!(
                    "block {index}: ommers are outside valid Monad block execution"
                )));
            }
            if block.header.excess_blob_gas.is_some_and(|gas| gas != 0) {
                return Err(FixtureError::Unsupported(format!(
                    "block {index}: nonzero excessBlobGas requires a fork-specific blob-base-fee environment"
                )));
            }
            parsed_blocks.push(block);
        }
        let post = match (value.get("postState"), value.get("post")) {
            (Some(_), Some(_)) => return Err(eyre!("both postState and post were supplied").into()),
            (post, other) => post.or(other).map(allocation).transpose()?,
        };
        Ok(Self {
            network,
            chain_id,
            genesis,
            blocks: parsed_blocks,
            pre: allocation(required(value, "pre")?)?,
            post,
            post_hash: value.get("postStateHash").map(fixed).transpose()?,
            last_block_hash: fixed(required(value, "lastblockhash")?)?,
        })
    }
}

fn check_fields(value: &Value, known: &[&str], context: &str) -> Result<(), FixtureError> {
    let object = value.as_object().ok_or_else(|| eyre!("{context} must be an object"))?;
    for key in object.keys() {
        if !known.contains(&key.as_str()) {
            return Err(FixtureError::Unsupported(format!("{context} field {key}")));
        }
    }
    Ok(())
}

fn header(value: &Value) -> Result<Header, FixtureError> {
    let fields = value.as_object().ok_or_else(|| eyre!("blockHeader must be an object"))?;
    for key in fields.keys() {
        if !matches!(
            key.as_str(),
            "parentHash"
                | "uncleHash"
                | "coinbase"
                | "stateRoot"
                | "transactionsTrie"
                | "receiptTrie"
                | "bloom"
                | "difficulty"
                | "number"
                | "gasLimit"
                | "gasUsed"
                | "timestamp"
                | "extraData"
                | "mixHash"
                | "nonce"
                | "baseFeePerGas"
                | "withdrawalsRoot"
                | "blobGasUsed"
                | "excessBlobGas"
                | "parentBeaconBlockRoot"
                | "requestsHash"
                | "hash"
        ) {
            return Err(FixtureError::Unsupported(format!("header field {key}")));
        }
    }
    let header = Header {
        parent_hash: fixed(required(value, "parentHash")?)?,
        ommers_hash: fixed(required(value, "uncleHash")?)?,
        beneficiary: Address::from(fixed::<20>(required(value, "coinbase")?)?),
        state_root: fixed(required(value, "stateRoot")?)?,
        transactions_root: fixed(required(value, "transactionsTrie")?)?,
        receipts_root: fixed(required(value, "receiptTrie")?)?,
        logs_bloom: fixed::<256>(required(value, "bloom")?)?.into(),
        difficulty: quantity(required(value, "difficulty")?)?,
        number: quantity_u64(required(value, "number")?)?,
        gas_limit: quantity_u64(required(value, "gasLimit")?)?,
        gas_used: quantity_u64(required(value, "gasUsed")?)?,
        timestamp: quantity_u64(required(value, "timestamp")?)?,
        extra_data: bytes(required(value, "extraData")?)?,
        mix_hash: fixed(required(value, "mixHash")?)?,
        nonce: fixed(required(value, "nonce")?)?,
        base_fee_per_gas: optional(value, "baseFeePerGas", quantity_u64)?,
        withdrawals_root: optional(value, "withdrawalsRoot", fixed)?,
        blob_gas_used: optional(value, "blobGasUsed", quantity_u64)?,
        excess_blob_gas: optional(value, "excessBlobGas", quantity_u64)?,
        parent_beacon_block_root: optional(value, "parentBeaconBlockRoot", fixed)?,
        requests_hash: optional(value, "requestsHash", fixed)?,
        ..Default::default()
    };
    if let Some(hash) = value.get("hash") {
        if header.hash_slow() != fixed::<32>(hash)? {
            return Err(eyre!("header hash differs from its encoded fields").into());
        }
    }
    Ok(header)
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| eyre!("missing field {key}"))
}

fn optional<T>(
    value: &Value,
    key: &str,
    parse: impl FnOnce(&Value) -> Result<T>,
) -> Result<Option<T>> {
    value.get(key).filter(|v| !v.is_null()).map(parse).transpose()
}

fn quantity(value: &Value) -> Result<U256> {
    if let Some(number) = value.as_u64() {
        return Ok(U256::from(number));
    }
    let text = value
        .as_str()
        .ok_or_else(|| eyre!("expected an integer or numeric string, got {value}"))?;
    let (digits, radix) = text.strip_prefix("0x").map_or((text, 10), |s| (s, 16));
    U256::from_str_radix(digits, radix).wrap_err_with(|| format!("invalid quantity {text}"))
}

fn quantity_u64(value: &Value) -> Result<u64> {
    quantity(value)?.try_into().map_err(|_| eyre!("quantity {value} exceeds u64"))
}

fn bytes(value: &Value) -> Result<Bytes> {
    let text = value.as_str().ok_or_else(|| eyre!("expected a hex byte string, got {value}"))?;
    hex::decode(text).map(Bytes::from).wrap_err("invalid hex byte string")
}

fn fixed<const N: usize>(value: &Value) -> Result<FixedBytes<N>> {
    let text = value.as_str().ok_or_else(|| eyre!("expected a hex string, got {value}"))?;
    let digits = text.strip_prefix("0x").unwrap_or(text);
    ensure!(digits.len() <= N * 2, "hex value exceeds {N} bytes");
    let digits = if digits.len() % 2 == 0 { digits.to_owned() } else { format!("0{digits}") };
    let decoded = hex::decode(&digits).wrap_err("invalid fixed-width hex string")?;
    let mut result = [0; N];
    result[N - decoded.len()..].copy_from_slice(&decoded);
    Ok(FixedBytes::from(result))
}

fn allocation(value: &Value) -> Result<Allocation> {
    let entries = value.as_object().ok_or_else(|| eyre!("allocation must be an object"))?;
    let mut accounts = BTreeMap::new();
    for (address, value) in entries {
        let address = Address::from(fixed::<20>(&Value::String(address.clone()))?);
        let fields =
            value.as_object().ok_or_else(|| eyre!("account {address} must be an object"))?;
        for key in fields.keys() {
            ensure!(
                matches!(key.as_str(), "nonce" | "balance" | "code" | "storage"),
                "unknown account field {key}"
            );
        }
        let mut storage = BTreeMap::new();
        if let Some(slots) = value.get("storage") {
            for (key, value) in
                slots.as_object().ok_or_else(|| eyre!("storage must be an object"))?
            {
                let key = U256::from_be_bytes(fixed::<32>(&Value::String(key.clone()))?.0);
                let value = quantity(value)?;
                ensure!(
                    storage.insert(key, value).is_none(),
                    "duplicate normalized storage key {key}"
                );
            }
        }
        storage.retain(|_, value| !value.is_zero());
        let account = Account {
            nonce: optional(value, "nonce", quantity_u64)?.unwrap_or_default(),
            balance: optional(value, "balance", quantity)?.unwrap_or_default(),
            code: optional(value, "code", bytes)?.unwrap_or_default(),
            storage,
        };
        ensure!(
            accounts.insert(address, account).is_none(),
            "duplicate normalized address {address}"
        );
    }
    Ok(accounts)
}
