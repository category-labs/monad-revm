//! Native execution and state regressions for the browser API.

use monad_revmjs::Engine;
use serde_json::{json, Value};

const SENDER: &str = "0x1111111111111111111111111111111111111111";

fn engine(spec: &str) -> Engine {
    let mut engine = Engine::new(spec, 1337).unwrap();
    engine.fund(SENDER, "0x3635c9adc5dea00000").unwrap();
    engine
}

fn execute(engine: &mut Engine, to: Option<&str>, data: &str, commit: bool) -> Value {
    let sender: Value = serde_json::from_str(&engine.account(SENDER).unwrap()).unwrap();
    serde_json::from_str(
        &engine
            .execute(
                &json!({
                    "from": SENDER, "to": to, "data": data, "value": "0x0", "gas": 1_000_000,
                    "gasPrice": "0x1", "nonce": sender["nonce"], "blockNumber": 1,
                    "timestamp": 1_800_000_000, "commit": commit, "trace": true,
                    "blockGasLimit": 30_000_000, "baseFee": "0x0",
                    "beneficiary": "0x0000000000000000000000000000000000000000",
                    "prevrandao": format!("0x{:064x}", 0)
                })
                .to_string(),
            )
            .unwrap(),
    )
    .unwrap()
}

fn deploy(engine: &mut Engine, runtime: &str) -> String {
    let len = runtime.len() / 2;
    assert!(len < 256);
    let init = format!("0x60{len:02x}600c60003960{len:02x}6000f3{runtime}");
    let result = execute(engine, None, &init, true);
    assert_eq!(result["success"], true, "{result}");
    result["createdAddress"].as_str().unwrap().to_owned()
}

#[test]
fn commit_simulate_snapshot_and_full_limit_charge() {
    let mut vm = engine("monad-next");
    let contract = deploy(&mut vm, "602a5f555f545f5260205ff3");
    let before = vm.snapshot().unwrap();
    let call = execute(&mut vm, Some(&contract), "0x", false);
    assert_eq!(call["success"], true);
    assert_eq!(call["returnValue"], format!("0x{:064x}", 42));
    assert_eq!(vm.snapshot().unwrap(), before);
    let result = execute(&mut vm, Some(&contract), "0x", true);
    assert_eq!(result["gasUsed"], 1_000_000);
    assert!(result["gasSpent"].as_u64().unwrap() < 1_000_000);
    let state: Value = serde_json::from_str(&vm.account(&contract).unwrap()).unwrap();
    assert_eq!(state["storage"][format!("0x{:064x}", 0)], format!("0x{:064x}", 42));
    let snapshot = vm.snapshot().unwrap();
    let mut restored = engine("monad-next");
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.snapshot().unwrap(), snapshot);
    let sender: Value = serde_json::from_str(&vm.account(SENDER).unwrap()).unwrap();
    assert_eq!(sender["nonce"], 2);
    assert_eq!(sender["balance"], format!("0x{:x}", 1_000_000_000_000_000_000_000u128 - 2_000_000));
}

#[test]
fn extension_trace_has_real_pc_stack_and_gas() {
    let mut vm = engine("monad-next");
    let contract = deploy(&mut vm, "ae005f5260205ff3");
    let result = execute(&mut vm, Some(&contract), "0x", false);
    assert_eq!(result["success"], true);
    assert_eq!(result["returnValue"], format!("0x{:064x}", 0));
    let trace = &result["structLogs"];
    assert_eq!(trace[0]["pc"], 0);
    assert_eq!(trace[0]["op"], "CALLSTACKDEPTH");
    assert_eq!(trace[0]["gasCost"], 2);
    assert_eq!(trace[0]["depth"], 1);
    assert_eq!(trace[1]["pc"], 2);
    assert_eq!(trace[1]["stack"][0], format!("{:064x}", 0));
}

#[test]
fn extension_is_invalid_before_monad_next() {
    let mut vm = engine("monad-ten");
    let contract = deploy(&mut vm, "ae005f5260205ff3");
    let result = execute(&mut vm, Some(&contract), "0x", true);
    assert_eq!(result["success"], false);
    assert_eq!(result["structLogs"][0]["op"], "INVALID");
    assert_eq!(result["gasUsed"], 1_000_000);
}

#[test]
fn caller_introspection_and_reverted_storage() {
    let mut vm = engine("monad-next");
    let caller = deploy(&mut vm, "5fae015f5260205ff3");
    let result = execute(&mut vm, Some(&caller), "0x", false);
    assert_eq!(result["returnValue"], format!("0x{:0>64}", &SENDER[2..]));
    let reverter = deploy(&mut vm, "602a5f555f5ffd");
    let result = execute(&mut vm, Some(&reverter), "0x", true);
    assert_eq!(result["error"], "revert");
    let account: Value = serde_json::from_str(&vm.account(&reverter).unwrap()).unwrap();
    assert_eq!(account["storage"], json!({}));
}

#[test]
fn malformed_restore_is_atomic_and_spec_specific() {
    let mut vm = engine("monad-next");
    let before = vm.snapshot().unwrap();
    assert!(vm.restore("{}").is_err());
    assert_eq!(vm.snapshot().unwrap(), before);
    assert!(engine("monad-ten").restore(&before).is_err());
}

#[test]
fn sha256_precompile_uses_browser_compatible_backend() {
    let mut vm = engine("monad-next");
    let result = execute(&mut vm, Some("0x0000000000000000000000000000000000000002"), "0x", false);
    assert_eq!(
        result["returnValue"],
        "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}
