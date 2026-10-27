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
    let fixtures: Value =
        serde_json::from_str(include_str!("../fixtures/transactions.json")).unwrap();
    let mut report = serde_json::Map::new();
    for fork in ["monad-ten", "monad-next"] {
        for case in fixtures["cases"].as_array().unwrap() {
            let label = format!("{fork}/{}", case["name"].as_str().unwrap());
            let mut vm = Engine::new(fork, 1337).unwrap();
            let mut snapshot: Value = serde_json::from_str(&vm.snapshot().unwrap()).unwrap();
            for accounts in [&fixtures["accounts"], &case["accounts"]] {
                for (address, fields) in accounts.as_object().unwrap() {
                    let account = snapshot["accounts"]
                        .as_object_mut()
                        .unwrap()
                        .entry(address.clone())
                        .or_insert_with(
                            || json!({ "balance": "0x0", "nonce": 0, "code": "0x", "storage": {} }),
                        );
                    account.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
                }
            }
            vm.restore(&snapshot.to_string()).unwrap();
            let mut results = Vec::new();
            for step in case["steps"].as_array().unwrap() {
                let mut request = json!({
                    "from": fixtures["payer"], "to": fixtures["authority"], "data": "0x", "value": "0x0",
                    "gas": 100_000, "gasPrice": "0x0", "blockNumber": 1, "timestamp": 1_800_000_000,
                    "blockGasLimit": 30_000_000, "baseFee": "0x0", "beneficiary": format!("0x{:040x}", 0),
                    "prevrandao": format!("0x{:064x}", 0), "commit": false, "trace": true
                });
                request
                    .as_object_mut()
                    .unwrap()
                    .extend(step["request"].as_object().unwrap().clone());
                let sender: Value =
                    serde_json::from_str(&vm.account(request["from"].as_str().unwrap()).unwrap())
                        .unwrap();
                request["nonce"] = sender["nonce"].clone();
                let before = vm.snapshot().unwrap();
                let output = vm.execute(&request.to_string());
                if let Some(error) = step["error"].as_str() {
                    let message = output.unwrap_err();
                    assert!(message.contains(error), "{label}: {message}, expected {error}");
                    assert_eq!(
                        before,
                        vm.snapshot().unwrap(),
                        "{label}: invalid request mutated state"
                    );
                    results.push(json!({ "error": message }));
                    continue;
                }
                let result: Value =
                    serde_json::from_str(&output.unwrap_or_else(|e| panic!("{label}: {e}")))
                        .unwrap();
                subset(&result, &step["result"], &label);
                assert_eq!(result["gasUsed"], request["gas"], "{label}: full-limit charge");
                if let Some(costs) = step["opcodeCosts"].as_object() {
                    for (opcode, cost) in costs {
                        let instruction = result["structLogs"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|entry| entry["op"] == *opcode)
                            .unwrap();
                        assert_eq!(&instruction["gasCost"], cost, "{label}: {opcode}");
                    }
                }
                request["trace"] = json!(false);
                let plain: Value =
                    serde_json::from_str(&vm.execute(&request.to_string()).unwrap()).unwrap();
                for field in [
                    "success",
                    "error",
                    "returnValue",
                    "gasUsed",
                    "gasSpent",
                    "logs",
                    "authorizationAuthorities",
                ] {
                    assert_eq!(result[field], plain[field], "{label}: traced/untraced {field}");
                }
                assert_eq!(before, vm.snapshot().unwrap(), "{label}: simulation mutated state");
                request["trace"] = json!(true);
                request["commit"] = json!(true);
                let committed: Value =
                    serde_json::from_str(&vm.execute(&request.to_string()).unwrap()).unwrap();
                assert_eq!(result, committed, "{label}: commit changed execution");
                for (address, expected) in step["accounts"].as_object().unwrap() {
                    let mut actual: Value =
                        serde_json::from_str(&vm.account(address).unwrap()).unwrap();
                    if let Some(storage) = expected["storage"].as_object() {
                        for key in storage.keys() {
                            actual["storage"]
                                .as_object_mut()
                                .unwrap()
                                .entry(key.clone())
                                .or_insert_with(|| json!(format!("0x{:064x}", 0)));
                        }
                    }
                    subset(&actual, expected, &format!("{label}/{address}"));
                }
                let after = vm.snapshot().unwrap();
                let mut restored = Engine::new(fork, 1337).unwrap();
                restored.restore(&after).unwrap();
                assert_eq!(after, restored.snapshot().unwrap(), "{label}: restore");
                results.push(json!({ "result": result, "state": serde_json::from_str::<Value>(&after).unwrap() }));
            }
            report.insert(label, json!(results));
        }
    }
    Value::Object(report)
}
