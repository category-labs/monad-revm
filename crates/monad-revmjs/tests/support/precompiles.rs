use monad_revmjs::Engine;
use serde::Deserialize;
use serde_json::{json, Value};

const SENDER: &str = "0x1111111111111111111111111111111111111111";
const TX_GAS: u64 = 3_000_000;

#[derive(Debug, Deserialize)]
struct Vector {
    name: String,
    address: u16,
    input: String,
    output: String,
    gas: u64,
    success: bool,
}

fn execute(vm: &mut Engine, to: Option<&str>, data: &str, commit: bool, trace: bool) -> Value {
    let account: Value = serde_json::from_str(&vm.account(SENDER).unwrap()).unwrap();
    let request = json!({
        "from": SENDER, "to": to, "data": data, "value": "0x0", "gas": TX_GAS,
        "gasPrice": "0x1", "nonce": account["nonce"], "blockNumber": 1,
        "timestamp": 1_800_000_000, "commit": commit, "trace": trace,
        "blockGasLimit": 30_000_000, "baseFee": "0x0",
        "beneficiary": "0x0000000000000000000000000000000000000000",
        "prevrandao": format!("0x{:064x}", 0)
    });
    serde_json::from_str(&vm.execute(&request.to_string()).unwrap()).unwrap()
}

pub(crate) fn report() -> Value {
    let vectors: Vec<Vector> =
        serde_json::from_str(include_str!("../fixtures/precompiles.json")).unwrap();
    let mut report = serde_json::Map::new();
    for fork in ["monad-ten", "monad-next"] {
        for vector in &vectors {
            let mut variants = vec![("exact", vector.gas, vector.success, vector.output.as_str())];
            if vector.success {
                variants.push(("surplus", vector.gas + 123, true, vector.output.as_str()));
                variants.push(("out-of-gas", vector.gas - 1, false, "0x"));
            }
            for (variant, gas_limit, success, output) in variants {
                let label = format!("{fork}/{}/{variant}", vector.name);
                let mut vm = Engine::new(fork, 1337).unwrap();
                vm.fund(SENDER, "0x3635c9adc5dea00000").unwrap();
                // Copy calldata, STATICCALL with an exact gas allowance, then return
                // a success word followed by the complete precompile return data.
                let runtime = format!(
                    "365f5f375f5f365f61{:04x}63{gas_limit:08x}fa5f523d5f60203e3d6020015ff3",
                    vector.address
                );
                let len = runtime.len() / 2;
                let init = format!("0x60{len:02x}600c60003960{len:02x}6000f3{runtime}");
                let deployment = execute(&mut vm, None, &init, true, false);
                assert_eq!(deployment["success"], true, "{label}: deploy");
                let address = deployment["createdAddress"].as_str().unwrap();
                let before = vm.snapshot().unwrap();
                let result = execute(&mut vm, Some(address), &vector.input, false, true);
                assert_eq!(result["success"], true, "{label}: caller survives precompile failure");
                assert_eq!(
                    result["returnValue"],
                    format!("0x{:064x}{}", u8::from(success), &output[2..]),
                    "{label}: return data"
                );
                assert_eq!(result["gasUsed"], TX_GAS, "{label}: Monad full-limit charge");
                assert_eq!(result["traceTruncated"], false, "{label}");
                let trace = result["structLogs"].as_array().unwrap();
                assert!(trace.iter().all(|step| step["depth"] == 1), "{label}: no bytecode child");
                let call = trace.iter().position(|step| step["op"] == "STATICCALL").unwrap();
                let consumed =
                    trace[call]["gas"].as_u64().unwrap() - trace[call + 1]["gas"].as_u64().unwrap();
                // Input memory is already expanded; precompiles are warm. Measure
                // across the call so refunded, unused forwarded gas is included.
                assert_eq!(
                    consumed,
                    100 + if success { vector.gas } else { gas_limit },
                    "{label}: gas"
                );
                let plain = execute(&mut vm, Some(address), &vector.input, false, false);
                for field in ["success", "error", "returnValue", "gasUsed", "gasSpent", "logs"] {
                    assert_eq!(result[field], plain[field], "{label}: tracing changed {field}");
                }
                assert_eq!(vm.snapshot().unwrap(), before, "{label}: simulation changed state");
                let committed = execute(&mut vm, Some(address), &vector.input, true, true);
                assert_eq!(committed, result, "{label}: commit changed execution");
                let account: Value = serde_json::from_str(&vm.account(SENDER).unwrap()).unwrap();
                assert_eq!(account["nonce"], 2, "{label}");
                report.insert(label, json!({ "execution": result, "sender": account }));
            }
        }
    }
    Value::Object(report)
}
