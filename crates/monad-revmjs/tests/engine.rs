//! Native execution and state regressions for the browser API.

use monad_revmjs::Engine;
use revm::primitives::{hex, keccak256};
use serde_json::{json, Value};

const SENDER: &str = "0x1111111111111111111111111111111111111111";

fn engine(spec: &str) -> Engine {
    let mut engine = Engine::new(spec, 1337).unwrap();
    engine.fund(SENDER, "0x3635c9adc5dea00000").unwrap();
    engine
}

fn execute(engine: &mut Engine, to: Option<&str>, data: &str, commit: bool) -> Value {
    execute_with_options(engine, to, data, json!({ "commit": commit }))
}

fn execute_with_options(
    engine: &mut Engine,
    to: Option<&str>,
    data: &str,
    options: Value,
) -> Value {
    let sender: Value = serde_json::from_str(&engine.account(SENDER).unwrap()).unwrap();
    let mut request = json!({
        "from": SENDER, "to": to, "data": data, "value": "0x0", "gas": 1_000_000,
        "gasPrice": "0x1", "nonce": sender["nonce"], "blockNumber": 1,
        "timestamp": 1_800_000_000, "commit": false, "trace": true,
        "blockGasLimit": 30_000_000, "baseFee": "0x0",
        "beneficiary": "0x0000000000000000000000000000000000000000",
        "prevrandao": format!("0x{:064x}", 0)
    });
    request.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
    serde_json::from_str(&engine.execute(&request.to_string()).unwrap()).unwrap()
}

fn deploy(engine: &mut Engine, runtime: &str) -> String {
    let len = runtime.len() / 2;
    assert!(len < 256);
    let init = format!("0x60{len:02x}600c60003960{len:02x}6000f3{runtime}");
    let result = execute(engine, None, &init, true);
    assert_eq!(result["success"], true, "{result}");
    result["createdAddress"].as_str().unwrap().to_owned()
}

const CONSOLE_CALL: &str = "365f5f375f5f365f73000000000000000000636f6e736f6c652e6c6f675afa50";

#[test]
fn console_calls_survive_reverts_and_do_not_change_execution() {
    for spec in ["monad-ten", "monad-next"] {
        let mut vm = engine(spec);
        for (suffix, success) in [("60ff5f5300", true), ("60ff5f535f5ffd", false)] {
            let contract = deploy(&mut vm, &format!("{CONSOLE_CALL}{suffix}"));
            let before = vm.snapshot().unwrap();
            let traced = execute(&mut vm, Some(&contract), "0x11223344", false);
            let plain = execute_with_options(
                &mut vm,
                Some(&contract),
                "0x11223344",
                json!({ "trace": false }),
            );
            assert_eq!(traced["success"], success);
            assert_eq!(traced["consoleCalls"], json!(["0x11223344"]));
            assert_eq!(plain["consoleCalls"], json!([]));
            for field in ["success", "returnValue", "gasUsed", "gasSpent", "logs"] {
                assert_eq!(traced[field], plain[field], "{field}");
            }
            assert_eq!(vm.snapshot().unwrap(), before);
            let committed = execute(&mut vm, Some(&contract), "0x11223344", true);
            assert_eq!(committed["consoleCalls"], traced["consoleCalls"]);
        }
    }
}

#[test]
fn console_calls_continue_after_instruction_trace_truncation() {
    let mut vm = engine("monad-next");
    // A finite loop exceeds the step budget before the console call executes.
    let contract = deploy(&mut vm, &format!("614e205b600190038060035750{CONSOLE_CALL}00"));
    let result = execute(&mut vm, Some(&contract), "0xaabbccdd", false);
    assert_eq!(result["success"], true);
    assert_eq!(result["traceTruncated"], true);
    assert_eq!(result["consoleCalls"], json!(["0xaabbccdd"]));
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
fn block_hashes_require_supplied_history_before_and_after_restore() {
    for spec in ["monad-ten", "monad-next"] {
        let mut vm = engine(spec);
        let contract = deploy(&mut vm, "6001405f5260205ff3");
        let mut restored = engine(spec);
        let supplied = format!("0x{}", "ab".repeat(32));
        for expected in [format!("0x{:064x}", 0), supplied.clone()] {
            if expected == supplied {
                vm.set_block_hash(1, &supplied).unwrap();
            }
            restored.restore(&vm.snapshot().unwrap()).unwrap();
            for vm in [&mut vm, &mut restored] {
                for trace in [false, true] {
                    let result = execute_with_options(
                        vm,
                        Some(&contract),
                        "0x",
                        json!({ "blockNumber": 2, "trace": trace }),
                    );
                    assert_eq!(result["success"], true);
                    assert_eq!(result["returnValue"], expected);
                }
                for number in [1, 258] {
                    let result = execute_with_options(
                        vm,
                        Some(&contract),
                        "0x",
                        json!({ "blockNumber": number }),
                    );
                    assert_eq!(result["returnValue"], format!("0x{:064x}", 0));
                }
            }
        }
    }
}

#[test]
fn malformed_snapshot_bytecode_returns_error_and_preserves_usable_state() {
    let mut vm = engine("monad-next");
    let contract = deploy(&mut vm, "602a5f555f545f5260205ff3");
    execute(&mut vm, Some(&contract), "0x", true);
    vm.set_block_hash(1, &format!("0x{}", "ab".repeat(32))).unwrap();
    let before = vm.snapshot().unwrap();
    for code in ["0xef01".to_owned(), format!("0xef0101{}", "00".repeat(20))] {
        let mut invalid: Value = serde_json::from_str(&before).unwrap();
        invalid["accounts"][SENDER]["balance"] = json!("0x2");
        invalid["accounts"][&contract]["code"] = json!(code);
        assert!(vm.restore(&invalid.to_string()).unwrap_err().contains("Invalid bytecode"));
        assert_eq!(vm.snapshot().unwrap(), before);
        let result = execute(&mut vm, Some(&contract), "0x", false);
        assert_eq!(result["success"], true);
        assert_eq!(result["returnValue"], format!("0x{:064x}", 42));
        assert_eq!(vm.snapshot().unwrap(), before);
    }
}

#[test]
fn hash_preimages_include_expanded_memory_and_empty_inputs() {
    let cases = [
        ("60205f20".to_owned(), vec![0; 32], String::new()),
        (
            "60115f5360405f20".to_owned(),
            [vec![0x11], vec![0; 63]].concat(),
            format!("11{}", "00".repeat(31)),
        ),
        ("5f60ff20".to_owned(), Vec::new(), String::new()),
        (format!("5f7f{}20", "ff".repeat(32)), Vec::new(), String::new()),
    ];
    let mut vm = engine("monad-next");
    for (runtime, input, memory_before) in cases {
        let contract = deploy(&mut vm, &format!("{runtime}5f5260205ff3"));
        let result = execute(&mut vm, Some(&contract), "0x", false);
        let hash = hex::encode(keccak256(&input));
        assert_eq!(result["success"], true);
        assert_eq!(result["returnValue"], format!("0x{hash}"));
        assert_eq!(result["preimages"], json!({ hash: hex::encode(input) }));
        assert_eq!(result["traceTruncated"], false);
        let step = result["structLogs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["op"] == "KECCAK256")
            .unwrap();
        assert_eq!(step["memory"], memory_before);
        assert_eq!(step["stack"].as_array().unwrap().len(), 2);
    }
}

#[test]
fn failed_hash_does_not_record_a_preimage() {
    let mut vm = engine("monad-next");
    let contract = deploy(&mut vm, "60205f5260205f2000");
    let result = execute_with_options(&mut vm, Some(&contract), "0x", json!({ "gas": 21_020 }));
    assert_eq!(result["success"], false);
    assert_eq!(result["preimages"], json!({}));
    let step = result["structLogs"].as_array().unwrap().last().unwrap();
    assert_eq!(step["op"], "KECCAK256");
    assert!(step["error"].as_str().unwrap().contains("OutOfGas"));
    assert_eq!(result["traceTruncated"], false);
}

#[test]
fn preimages_count_toward_trace_budget_without_changing_execution() {
    let mut vm = engine("monad-next");
    let cases = [
        // The preimage fits, but leaves too little budget for the next step.
        ("628000005f205f00".to_owned(), 3, 1),
        // Earlier steps consume enough budget to exclude the preimage itself.
        (format!("5f620fffff53{}628000005f205f00", "5b".repeat(8)), 14, 0),
    ];
    for (runtime, step_count, preimage_count) in cases {
        let contract = deploy(&mut vm, &runtime);
        let traced =
            execute_with_options(&mut vm, Some(&contract), "0x", json!({ "gas": 3_000_000 }));
        let untraced = execute_with_options(
            &mut vm,
            Some(&contract),
            "0x",
            json!({ "gas": 3_000_000, "trace": false }),
        );
        assert_eq!(traced["success"], true);
        assert_eq!(traced["gasSpent"], untraced["gasSpent"]);
        assert_eq!(traced["returnValue"], untraced["returnValue"]);
        assert_eq!(traced["traceTruncated"], true);
        let steps = traced["structLogs"].as_array().unwrap();
        assert_eq!(steps.len(), step_count);
        assert_eq!(steps.last().unwrap()["op"], "KECCAK256");
        let preimages = traced["preimages"].as_object().unwrap();
        assert_eq!(preimages.len(), preimage_count);
        for input in preimages.values() {
            assert_eq!(input.as_str().unwrap().len(), 16 * 1024 * 1024);
        }
        assert!(traced.to_string().len() < 32 * 1024 * 1024);
    }
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
