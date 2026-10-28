use crate::{address, word, Engine};
use monad_revm::{
    api::block::{apply_syscall_on_epoch_change, apply_syscall_reward, apply_syscall_snapshot},
    monad_context_with_db,
    staking::STAKING_ADDRESS,
    MonadBuilder, MonadCfgEnv,
};
use revm::{
    context::{result::ExecutionResult, BlockEnv},
    primitives::{hex, U256},
    DatabaseCommit, ExecuteEvm,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Action {
    Snapshot {},
    EpochChange { epoch: String },
    Reward { author: String, amount: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    action: Action,
    block_number: u64,
    timestamp: u64,
    commit: bool,
}

pub(crate) fn execute(engine: &mut Engine, input: &str) -> Result<String, String> {
    let request: Request = serde_json::from_str(input).map_err(|e| e.to_string())?;
    // Reward helpers account for rewards in storage; their host must mint the backing balance.
    // Stage both effects in the same isolated database so failure cannot mint tokens.
    let mut db = engine.db.clone();
    let reward = if let Action::Reward { author, amount } = &request.action {
        let author = address(author)?;
        let amount = word(amount)?;
        let mut account =
            db.cache.accounts.get(&STAKING_ADDRESS).map(|a| a.info.clone()).unwrap_or_default();
        account.balance = account.balance.checked_add(amount).ok_or("Staking balance overflow")?;
        db.insert_account_info(STAKING_ADDRESS, account);
        Some((author, amount))
    } else {
        None
    };
    let mut cfg = MonadCfgEnv::new_with_spec(engine.spec);
    cfg.chain_id = engine.chain_id;
    let block = BlockEnv {
        number: U256::from(request.block_number),
        timestamp: U256::from(request.timestamp),
        gas_limit: 30_000_000,
        ..Default::default()
    };
    let mut evm = monad_context_with_db(db.clone()).with_cfg(cfg).with_block(block).build_monad();
    let result = match request.action {
        Action::Snapshot {} => apply_syscall_snapshot(&mut evm),
        Action::EpochChange { epoch } => {
            let epoch: u64 = word(&epoch)?.try_into().map_err(|_| "Epoch exceeds u64")?;
            // The native transition also prepares the following epoch's accumulators.
            if epoch == u64::MAX {
                return Err("Epoch must be below u64::MAX".into());
            }
            apply_syscall_on_epoch_change(&mut evm, epoch)
        }
        Action::Reward { .. } => {
            let (author, amount) = reward.expect("reward was validated above");
            apply_syscall_reward(&mut evm, author, amount)
        }
    }
    .map_err(|e| format!("Staking transition failed: {e:?}"))?;
    let state = evm.finalize();
    let error = match &result {
        ExecutionResult::Success { .. } => None,
        ExecutionResult::Revert { .. } => Some("revert".to_owned()),
        ExecutionResult::Halt { reason, .. } => Some(format!("{reason:?}")),
    };
    let logs: Vec<_> = result
        .logs()
        .iter()
        .map(|log| {
            json!({
                "address": format!("{:#x}", log.address),
                "topics": log.data.topics().iter().map(|t| format!("{t:#x}")).collect::<Vec<_>>(),
                "data": hex::encode_prefixed(&log.data.data)
            })
        })
        .collect();
    let response = json!({
        "success": result.is_success(), "error": error,
        "returnValue": result.output().map(hex::encode_prefixed).unwrap_or_else(|| "0x".into()),
        "gasSpent": result.gas().total_gas_spent(), "logs": logs
    });
    let response = serde_json::to_string(&response).map_err(|e| e.to_string())?;
    if request.commit && result.is_success() {
        db.commit(state);
        engine.db = db;
    }
    Ok(response)
}
