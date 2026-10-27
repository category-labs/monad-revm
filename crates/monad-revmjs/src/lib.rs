//! JavaScript-accessible execution, state and inspection for a local Monad VM.

mod database;
mod trace;

use database::LocalDB;
use monad_revm::{monad_context_with_db, MonadBuilder, MonadCfgEnv, MonadHardfork};
use revm::{
    context::{result::ExecutionResult, BlockEnv, TxEnv},
    database::AccountState,
    inspector::InspectEvm,
    primitives::{hex, Address, Bytes, B256, U256},
    state::{AccountInfo, Bytecode},
    DatabaseCommit, ExecuteEvm,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, str::FromStr};
use trace::Trace;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Transaction {
    from: String,
    to: Option<String>,
    data: String,
    value: String,
    gas: u64,
    gas_price: String,
    #[serde(default)]
    priority_fee: Option<String>,
    nonce: u64,
    block_number: u64,
    timestamp: u64,
    block_gas_limit: u64,
    base_fee: String,
    beneficiary: String,
    prevrandao: String,
    commit: bool,
    trace: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Account {
    balance: String,
    nonce: u64,
    code: String,
    storage: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Snapshot {
    version: u32,
    spec: String,
    chain_id: u64,
    accounts: BTreeMap<String, Account>,
    block_hashes: BTreeMap<String, String>,
}

/// Owns local state; simulation and tracing do not mutate that state.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Engine {
    db: LocalDB,
    spec: MonadHardfork,
    chain_id: u64,
}

fn address(value: &str) -> Result<Address, String> {
    Address::from_str(value).map_err(|e| format!("Invalid address: {e}"))
}

fn word(value: &str) -> Result<U256, String> {
    U256::from_str(value).map_err(|e| format!("Invalid 256-bit quantity: {e}"))
}

#[wasm_bindgen]
impl Engine {
    /// Create an empty local VM with an explicit specification and chain ID.
    #[wasm_bindgen(constructor)]
    pub fn new(spec: &str, chain_id: u32) -> Result<Self, String> {
        let spec = match spec {
            "monad-ten" => MonadHardfork::MonadTen,
            "monad-next" => MonadHardfork::MonadNext,
            _ => return Err("Supported specifications: monad-ten, monad-next".into()),
        };
        Ok(Self { db: LocalDB::default(), spec, chain_id: chain_id.into() })
    }

    /// Set a development account's balance without replacing its code or nonce.
    pub fn fund(&mut self, account: &str, balance: &str) -> Result<(), String> {
        let account = address(account)?;
        let balance = word(balance)?;
        let mut info =
            self.db.cache.accounts.get(&account).map(|a| a.info.clone()).unwrap_or_default();
        info.balance = balance;
        self.db.insert_account_info(account, info);
        Ok(())
    }

    /// Read an account as JSON, including its bytecode and nonzero storage slots.
    pub fn account(&self, account: &str) -> Result<String, String> {
        serde_json::to_string(&self.read_account(address(account)?)).map_err(|e| e.to_string())
    }

    /// Supply a historical hash for the BLOCKHASH instruction.
    /// Unsupplied hashes return zero, including after restoring a snapshot.
    pub fn set_block_hash(&mut self, number: u32, hash: &str) -> Result<(), String> {
        let hash = B256::from_str(hash).map_err(|e| e.to_string())?;
        self.db.cache.block_hashes.insert(U256::from(number), hash);
        Ok(())
    }

    /// Execute against an isolated cache, applying state only for committed transactions.
    pub fn execute(&mut self, request: &str) -> Result<String, String> {
        let request: Transaction = serde_json::from_str(request).map_err(|e| e.to_string())?;
        let gas_price =
            word(&request.gas_price)?.try_into().map_err(|_| "Gas price exceeds u128")?;
        let priority_fee = request
            .priority_fee
            .as_deref()
            .map(word)
            .transpose()?
            .map(u128::try_from)
            .transpose()
            .map_err(|_| "Priority fee exceeds u128")?;
        let mut tx = TxEnv::builder()
            .caller(address(&request.from)?)
            .value(word(&request.value)?)
            .gas_limit(request.gas)
            .gas_price(gas_price)
            .nonce(request.nonce)
            .chain_id(Some(self.chain_id))
            .gas_priority_fee(priority_fee)
            .data(Bytes::from_str(&request.data).map_err(|e| e.to_string())?);
        tx = match request.to {
            Some(to) => tx.to(address(&to)?),
            None => tx.create(),
        };
        let mut cfg = MonadCfgEnv::new_with_spec(self.spec);
        cfg.chain_id = self.chain_id;
        let block = BlockEnv {
            number: U256::from(request.block_number),
            timestamp: U256::from(request.timestamp),
            gas_limit: request.block_gas_limit,
            basefee: word(&request.base_fee)?.try_into().map_err(|_| "Base fee exceeds u64")?,
            beneficiary: address(&request.beneficiary)?,
            prevrandao: Some(B256::from_str(&request.prevrandao).map_err(|e| e.to_string())?),
            ..Default::default()
        };
        let ctx = monad_context_with_db(self.db.clone()).with_cfg(cfg).with_block(block);
        let mut evm = ctx.build_monad_with_inspector(Trace::new(self.spec));
        let output = if request.trace {
            evm.inspect_tx(tx.build_fill())
        } else {
            evm.transact(tx.build_fill())
        }
        .map_err(|e| format!("Transaction validation failed: {e:?}"))?;
        let result = &output.result;
        let error = match result {
            ExecutionResult::Success { .. } => None,
            ExecutionResult::Revert { .. } => Some("revert".to_owned()),
            ExecutionResult::Halt { reason, .. } => Some(format!("{reason:?}")),
        };
        let trace = evm.0.inspector;
        let logs: Vec<Value> = result.logs().iter().map(|log| json!({
            "address": format!("{:#x}", log.address),
            "topics": log.data.topics().iter().map(|t| format!("{t:#x}")).collect::<Vec<_>>(),
            "data": hex::encode_prefixed(&log.data.data)
        })).collect();
        let response = json!({
            "success": result.is_success(), "error": error,
            "returnValue": result.output().map(hex::encode_prefixed).unwrap_or_else(|| "0x".into()),
            "createdAddress": result.created_address().map(|a| format!("{a:#x}")),
            "gasUsed": result.tx_gas_used(), "gasSpent": result.gas().total_gas_spent(),
            "logs": logs, "structLogs": trace.steps, "preimages": trace.preimages,
            "consoleCalls": trace.console_calls,
            "traceTruncated": trace.truncated
        });
        if request.commit {
            self.db.commit(output.state);
        }
        serde_json::to_string(&response).map_err(|e| e.to_string())
    }

    /// Export versioned JSON state, independent of any browser persistence mechanism.
    pub fn snapshot(&self) -> Result<String, String> {
        let accounts = self
            .db
            .cache
            .accounts
            .iter()
            .filter(|(_, account)| account.account_state != AccountState::NotExisting)
            .map(|(addr, _)| (format!("{addr:#x}"), self.read_account(*addr)))
            .collect();
        let block_hashes = self
            .db
            .cache
            .block_hashes
            .iter()
            .map(|(number, hash)| (format!("{number:#x}"), format!("{hash:#x}")))
            .collect();
        serde_json::to_string(&Snapshot {
            version: 1,
            spec: self.spec.to_string(),
            chain_id: self.chain_id,
            accounts,
            block_hashes,
        })
        .map_err(|e| e.to_string())
    }

    /// Validate completely before replacing the current database.
    pub fn restore(&mut self, snapshot: &str) -> Result<(), String> {
        let snapshot: Snapshot = serde_json::from_str(snapshot).map_err(|e| e.to_string())?;
        if snapshot.version != 1
            || snapshot.spec != self.spec.to_string()
            || snapshot.chain_id != self.chain_id
        {
            return Err("Incompatible Monad state version, specification or chain ID".into());
        }
        let mut db = LocalDB::default();
        for (addr, account) in snapshot.accounts {
            let addr = address(&addr)?;
            let code = Bytes::from_str(&account.code).map_err(|e| e.to_string())?;
            let code = Bytecode::new_raw_checked(code)
                .map_err(|e| format!("Invalid bytecode for {addr:#x}: {e}"))?;
            db.insert_account_info(
                addr,
                AccountInfo {
                    balance: word(&account.balance)?,
                    nonce: account.nonce,
                    ..Default::default()
                }
                .with_code(code),
            );
            for (slot, value) in account.storage {
                db.insert_account_storage(addr, word(&slot)?, word(&value)?)
                    .map_err(|e| e.to_string())?;
            }
        }
        for (number, hash) in snapshot.block_hashes {
            db.cache
                .block_hashes
                .insert(word(&number)?, B256::from_str(&hash).map_err(|e| e.to_string())?);
        }
        self.db = db;
        Ok(())
    }
}

impl Engine {
    fn read_account(&self, address: Address) -> Account {
        let Some(account) = self.db.cache.accounts.get(&address) else {
            return Account {
                balance: "0x0".into(),
                nonce: 0,
                code: "0x".into(),
                storage: BTreeMap::new(),
            };
        };
        let code = self
            .db
            .cache
            .contracts
            .get(&account.info.code_hash)
            .map(|c| hex::encode_prefixed(c.original_bytes()))
            .unwrap_or_else(|| "0x".into());
        Account {
            balance: format!("{:#x}", account.info.balance),
            nonce: account.info.nonce,
            code,
            storage: account
                .storage
                .iter()
                .filter(|(_, v)| !v.is_zero())
                .map(|(k, v)| (format!("0x{k:064x}"), format!("0x{v:064x}")))
                .collect(),
        }
    }
}
