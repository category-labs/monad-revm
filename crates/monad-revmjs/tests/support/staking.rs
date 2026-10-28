use monad_revmjs::Engine;
use serde_json::{json, Value};

fn subset(actual: &Value, expected: &Value, label: &str) {
    if let Some(object) = expected.as_object() {
        for (key, value) in object {
            subset(&actual[key], value, &format!("{label}/{key}"));
        }
    } else {
        assert_eq!(actual, expected, "{label}");
    }
}

pub(crate) fn report() -> Value {
    let fixtures: Value = serde_json::from_str(include_str!("../fixtures/staking.json")).unwrap();
    let mut report = serde_json::Map::new();
    for fork in ["monad-ten", "monad-next"] {
        for case in fixtures["cases"].as_array().unwrap() {
            let label = format!("{fork}/{}", case["name"].as_str().unwrap());
            let mut vm = Engine::new(fork, 1337).unwrap();
            let mut initial: Value = serde_json::from_str(&vm.snapshot().unwrap()).unwrap();
            initial["accounts"] = fixtures["accounts"].clone();
            if let Some(accounts) = case["accounts"].as_object() {
                initial["accounts"].as_object_mut().unwrap().extend(accounts.clone());
            }
            vm.restore(&initial.to_string()).unwrap();
            let mut results = Vec::new();
            for step in case["steps"].as_array().unwrap() {
                let label = format!("{label}/{}", step["name"].as_str().unwrap());
                let system = step["request"].get("action").is_some();
                let mut request = if system {
                    json!({ "blockNumber": 1, "timestamp": 1_800_000_000, "commit": false })
                } else {
                    json!({
                        "from": fixtures["payer"], "to": fixtures["staking"], "data": "0x", "value": "0x0",
                        "gas": 3_000_000, "gasPrice": "0x0", "blockNumber": 1, "timestamp": 1_800_000_000,
                        "blockGasLimit": 30_000_000, "baseFee": "0x0", "beneficiary": format!("0x{:040x}", 0),
                        "prevrandao": format!("0x{:064x}", 0), "commit": false, "trace": true
                    })
                };
                request
                    .as_object_mut()
                    .unwrap()
                    .extend(step["request"].as_object().unwrap().clone());
                if !system {
                    let sender: Value = serde_json::from_str(
                        &vm.account(request["from"].as_str().unwrap()).unwrap(),
                    )
                    .unwrap();
                    request["nonce"] = sender["nonce"].clone();
                }
                let before = vm.snapshot().unwrap();
                let run = |vm: &mut Engine, request: &Value| {
                    if system {
                        vm.staking_transition(&request.to_string())
                    } else {
                        vm.execute(&request.to_string())
                    }
                };
                let output = run(&mut vm, &request);
                if let Some(error) = step["error"].as_str() {
                    let message = output.expect_err(&label);
                    assert!(message.contains(error), "{label}: {message}, expected {error}");
                    request["commit"] = json!(true);
                    assert_eq!(run(&mut vm, &request).unwrap_err(), message, "{label}");
                    assert_eq!(
                        vm.snapshot().unwrap(),
                        before,
                        "{label}: invalid request mutated state"
                    );
                    results.push(json!({ "error": message }));
                    continue;
                }
                let result: Value =
                    serde_json::from_str(&output.unwrap_or_else(|e| panic!("{label}: {e}")))
                        .unwrap();
                subset(&result, &step["result"], &label);
                assert_eq!(
                    result["logs"].as_array().unwrap().len() as u64,
                    step["logCount"].as_u64().unwrap(),
                    "{label}: logs"
                );
                if !system {
                    request["trace"] = json!(false);
                    let plain: Value =
                        serde_json::from_str(&run(&mut vm, &request).unwrap()).unwrap();
                    for field in ["success", "error", "returnValue", "gasUsed", "gasSpent", "logs"]
                    {
                        assert_eq!(
                            result[field], plain[field],
                            "{label}: inspection changed {field}"
                        );
                    }
                    request["trace"] = json!(true);
                }
                assert_eq!(vm.snapshot().unwrap(), before, "{label}: simulation mutated state");
                request["commit"] = json!(true);
                let committed: Value =
                    serde_json::from_str(&run(&mut vm, &request).unwrap()).unwrap();
                assert_eq!(result, committed, "{label}: commit changed result");
                if system && result["success"] == false {
                    assert_eq!(
                        vm.snapshot().unwrap(),
                        before,
                        "{label}: failed transition mutated state"
                    );
                }
                if step["stakingUnchanged"] == true {
                    let before: Value = serde_json::from_str(&before).unwrap();
                    let actual: Value = serde_json::from_str(
                        &vm.account(fixtures["staking"].as_str().unwrap()).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        actual,
                        before["accounts"][fixtures["staking"].as_str().unwrap()],
                        "{label}: staking rollback"
                    );
                }
                if let Some(accounts) = step["accounts"].as_object() {
                    for (address, expected) in accounts {
                        let actual: Value =
                            serde_json::from_str(&vm.account(address).unwrap()).unwrap();
                        subset(&actual, expected, &label);
                    }
                }
                let after = vm.snapshot().unwrap();
                // Continue the lifecycle from restored state after every operation.
                vm = Engine::new(fork, 1337).unwrap();
                vm.restore(&after).unwrap();
                assert_eq!(vm.snapshot().unwrap(), after, "{label}: restore");
                results.push(json!({ "result": result, "state": serde_json::from_str::<Value>(&after).unwrap() }));
            }
            report.insert(label, json!(results));
        }
    }
    Value::Object(report)
}
