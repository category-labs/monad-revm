//! End-to-end fixtures with independently calculated transaction effects.

use alloy_consensus::{
    crypto::secp256k1::sign_message,
    proofs::{calculate_receipt_root, calculate_transaction_root},
    transaction::SignerRecoverable,
    Block, BlockBody, Header, Receipt, ReceiptEnvelope, SignableTransaction, Transaction,
    TxEip1559, TxEip2930, TxEip7702, TxEnvelope, TxLegacy, TxType,
};
use alloy_eips::{
    eip2930::{AccessList, AccessListItem},
    eip4895::Withdrawal,
    eip7702::Authorization,
};
use alloy_primitives::{address, b256, hex, keccak256, Address, Bytes, Log, TxKind, B256, U256};
use alloy_trie::{
    root::{state_root_unhashed, storage_root_unhashed},
    TrieAccount, EMPTY_ROOT_HASH,
};
use monad_fixtures::{run, Report, Status};
use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

const RECIPIENT: Address = address!("2222222222222222222222222222222222222222");
const BENEFICIARY: Address = address!("3333333333333333333333333333333333333333");
const INITIAL_BALANCE: u128 = 20_000_000_000_000_000_000;
const SECRET: B256 = b256!("0101010101010101010101010101010101010101010101010101010101010101");

fn signed_transfer(nonce: u64) -> TxEnvelope {
    signed_transfer_with_price(nonce, 3)
}

fn signed_transfer_with_price(nonce: u64, gas_price: u128) -> TxEnvelope {
    let tx = TxLegacy {
        chain_id: Some(143),
        nonce,
        gas_price,
        gas_limit: 50_000,
        to: TxKind::Call(RECIPIENT),
        value: U256::from(7),
        input: Bytes::new(),
    };
    let signature = sign_message(SECRET, tx.signature_hash()).unwrap();
    tx.into_signed(signature).into()
}

fn account(nonce: u64, balance: u128, code: &str) -> Value {
    json!({"nonce": nonce, "balance": balance.to_string(), "code": code, "storage": {}})
}

fn root(alloc: &Value) -> B256 {
    state_root_unhashed(alloc.as_object().unwrap().iter().map(|(address, account)| {
        let storage_root =
            storage_root_unhashed(account["storage"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.parse::<B256>().unwrap(),
                    v.as_str().map_or_else(
                        || U256::from(v.as_u64().unwrap()),
                        |text| text.parse().unwrap(),
                    ),
                )
            }));
        (
            address.parse::<Address>().unwrap(),
            TrieAccount {
                nonce: account["nonce"].as_u64().unwrap(),
                balance: account["balance"].as_str().unwrap().parse().unwrap(),
                code_hash: keccak256(hex::decode(account["code"].as_str().unwrap()).unwrap()),
                storage_root,
            },
        )
    }))
}

fn header_json(header: &Header) -> Value {
    json!({
        "parentHash": header.parent_hash, "uncleHash": header.ommers_hash,
        "coinbase": header.beneficiary, "stateRoot": header.state_root,
        "transactionsTrie": header.transactions_root, "receiptTrie": header.receipts_root,
        "bloom": header.logs_bloom, "difficulty": "0x0", "number": header.number,
        "gasLimit": header.gas_limit, "gasUsed": header.gas_used, "timestamp": header.timestamp,
        "extraData": "0x", "mixHash": header.mix_hash, "nonce": "0x0",
        "baseFeePerGas": header.base_fee_per_gas, "withdrawalsRoot": header.withdrawals_root,
        "blobGasUsed": header.blob_gas_used, "excessBlobGas": header.excess_blob_gas,
        "parentBeaconBlockRoot": header.parent_beacon_block_root, "requestsHash": header.requests_hash,
        "hash": header.hash_slow(),
    })
}

fn block_json(block: &Block<TxEnvelope>) -> Value {
    json!({"blockHeader": header_json(&block.header), "rlp": hex::encode_prefixed(alloy_rlp::encode(block))})
}

// Fee, nonce, transfer, storage and log expectations are calculated without executing an EVM.
fn fixture(
    network: &str,
    code: &str,
    succeeds: bool,
    block_count: u64,
    store: bool,
    log: bool,
) -> Value {
    let sender = signed_transfer(0).recover_signer().unwrap();
    let pre = json!({
        sender.to_string(): account(0, INITIAL_BALANCE, "0x"),
        RECIPIENT.to_string(): account(1, 0, code),
    });
    let genesis = Header {
        beneficiary: BENEFICIARY,
        state_root: root(&pre),
        gas_limit: 30_000_000,
        base_fee_per_gas: Some(1),
        withdrawals_root: Some(EMPTY_ROOT_HASH),
        blob_gas_used: Some(0),
        excess_blob_gas: Some(0),
        parent_beacon_block_root: Some(B256::ZERO),
        requests_hash: Some(b256!(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )),
        ..Default::default()
    };
    let mut parent = genesis.clone();
    let mut post = pre.clone();
    let mut blocks = Vec::new();
    for nonce in 0..block_count {
        let tx = signed_transfer(nonce);
        let count = u128::from(nonce + 1);
        let value = if succeeds { 7 } else { 0 };
        post[&sender.to_string()] =
            account(nonce + 1, INITIAL_BALANCE - count * (150_000 + value), "0x");
        post[&RECIPIENT.to_string()] = account(1, count * value, code);
        post[&BENEFICIARY.to_string()] = account(0, count * 100_000, "0x");
        if store && succeeds {
            post[&RECIPIENT.to_string()]["storage"][B256::ZERO.to_string()] = json!(42);
        }
        let logs = if log && succeeds {
            vec![Log::new_unchecked(RECIPIENT, vec![], Bytes::new())]
        } else {
            vec![]
        };
        let receipt =
            Receipt { status: succeeds.into(), cumulative_gas_used: tx.gas_limit(), logs }
                .with_bloom();
        let header = Header {
            parent_hash: parent.hash_slow(),
            number: nonce + 1,
            timestamp: (nonce + 1) * 15,
            state_root: root(&post),
            transactions_root: calculate_transaction_root(std::slice::from_ref(&tx)),
            receipts_root: calculate_receipt_root(&[ReceiptEnvelope::from_typed(
                TxType::Legacy,
                receipt.clone(),
            )]),
            logs_bloom: receipt.logs_bloom,
            gas_used: tx.gas_limit(),
            ..genesis.clone()
        };
        let block = Block::new(
            header.clone(),
            BlockBody {
                transactions: vec![tx],
                ommers: vec![],
                withdrawals: Some(Default::default()),
            },
        );
        blocks.push(block_json(&block));
        parent = header;
    }
    json!({
        "_info": {"fixture-format": "blockchain_test"}, "network": network,
        "config": {"network": network, "chainid": "0x8f", "blobSchedule": {
            "MONAD_EIGHT": {"target": "0x6", "max": "0x9", "baseFeeUpdateFraction": "0x4c6964"},
            "MONAD_NINE": {"target": "0x6", "max": "0x9", "baseFeeUpdateFraction": "0x4c6964"},
        }},
        "genesisBlockHeader": header_json(&genesis), "genesisRLP": hex::encode_prefixed(alloy_rlp::encode(&genesis)),
        "pre": pre, "blocks": blocks, "postState": post, "lastblockhash": parent.hash_slow(),
    })
}

fn transfer() -> Value {
    fixture("MONAD_EIGHT", "0x00", true, 1, false, false)
}

fn reserve_fixture(
    network: &str,
    initial_balance: u128,
    gas_price: u128,
    outcomes: &[&[bool]],
) -> Value {
    let mut case = fixture(network, "0x00", true, 0, false, false);
    let sender = signed_transfer(0).recover_signer().unwrap().to_string();
    case["pre"][&sender]["balance"] = json!(initial_balance.to_string());
    let mut genesis = alloy_rlp::decode_exact::<Header>(
        &hex::decode(case["genesisRLP"].as_str().unwrap()).unwrap(),
    )
    .unwrap();
    genesis.state_root = root(&case["pre"]);
    case["genesisBlockHeader"] = header_json(&genesis);
    case["genesisRLP"] = json!(hex::encode_prefixed(alloy_rlp::encode(&genesis)));
    let mut parent = genesis.clone();
    let mut post = case["pre"].clone();
    let mut nonce = 0;
    let mut sent = 0;
    for (index, expected) in outcomes.iter().enumerate() {
        let mut transactions = Vec::new();
        let mut receipts = Vec::new();
        let mut gas_used = 0;
        for succeeds in *expected {
            let tx = signed_transfer_with_price(nonce, gas_price);
            gas_used += tx.gas_limit();
            nonce += 1;
            if *succeeds {
                sent += 7;
            }
            // Reserve rejection preserves full fees and nonce, but rolls back the value transfer.
            post[&sender] = account(
                nonce,
                initial_balance - u128::from(nonce) * 50_000 * gas_price - sent,
                "0x",
            );
            post[RECIPIENT.to_string()] = account(1, sent, "0x00");
            post[BENEFICIARY.to_string()] =
                account(0, u128::from(nonce) * 50_000 * (gas_price - 1), "0x");
            let receipt =
                Receipt { status: (*succeeds).into(), cumulative_gas_used: gas_used, logs: vec![] }
                    .with_bloom();
            receipts.push(ReceiptEnvelope::from_typed(TxType::Legacy, receipt));
            transactions.push(tx);
        }
        let header = Header {
            parent_hash: parent.hash_slow(),
            number: index as u64 + 1,
            timestamp: (index as u64 + 1) * 15,
            gas_used,
            state_root: root(&post),
            transactions_root: calculate_transaction_root(&transactions),
            receipts_root: calculate_receipt_root(&receipts),
            ..genesis.clone()
        };
        let block = Block::new(
            header.clone(),
            BlockBody { transactions, ommers: vec![], withdrawals: Some(Default::default()) },
        );
        case["blocks"].as_array_mut().unwrap().push(block_json(&block));
        parent = header;
    }
    case["lastblockhash"] = json!(parent.hash_slow());
    case["postState"] = post;
    case
}

fn write_cases(path: &Path, cases: Value) {
    fs::write(path, serde_json::to_vec(&cases).unwrap()).unwrap();
}

fn check(case: Value) -> Report {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.json");
    write_cases(&path, json!({"test": case}));
    run(&[path], None).unwrap()
}

fn rewrite_block(case: &mut Value, update: impl FnOnce(&mut Block<TxEnvelope>)) {
    let rlp = hex::decode(case["blocks"][0]["rlp"].as_str().unwrap()).unwrap();
    let mut block = alloy_rlp::decode_exact::<Block<TxEnvelope>>(&rlp).unwrap();
    update(&mut block);
    case["blocks"][0] = block_json(&block);
    case["lastblockhash"] = json!(block.header.hash_slow());
}

#[test]
fn executes_transactions_storage_logs_and_reverts_on_both_forks() {
    for network in ["MONAD_EIGHT", "MONAD_NINE", "MONAD_EIGHTToMONAD_NINEAtTime30"] {
        for (code, succeeds, store, log) in [
            ("0x00", true, false, false),
            ("0x602a60005500", true, true, false),
            ("0x60006000a000", true, false, true),
            ("0x602a60005560006000fd", false, true, false),
            ("0xfe", false, false, false),
        ] {
            let report = check(fixture(network, code, succeeds, 2, store, log));
            assert_eq!(report.cases[0].status, Status::Passed, "{network} {code}: {report:?}");
            assert_eq!(report.exit_code(true), 0);
        }
    }
}

#[test]
fn replays_typed_transactions_and_authorizations() {
    let access_list =
        AccessList(vec![AccessListItem { address: RECIPIENT, storage_keys: vec![B256::ZERO] }]);
    let tx = TxEip2930 {
        chain_id: 143,
        gas_price: 3,
        gas_limit: 50_000,
        to: TxKind::Call(RECIPIENT),
        value: U256::from(7),
        access_list: access_list.clone(),
        ..Default::default()
    };
    let signature = sign_message(SECRET, tx.signature_hash()).unwrap();
    let access = TxEnvelope::from(tx.into_signed(signature));
    let tx = TxEip1559 {
        chain_id: 143,
        max_fee_per_gas: 5,
        max_priority_fee_per_gas: 2,
        gas_limit: 50_000,
        to: TxKind::Call(RECIPIENT),
        value: U256::from(7),
        access_list,
        ..Default::default()
    };
    let signature = sign_message(SECRET, tx.signature_hash()).unwrap();
    let dynamic = TxEnvelope::from(tx.into_signed(signature));
    let auth = Authorization { chain_id: U256::from(143), address: RECIPIENT, nonce: 1 };
    let signature = sign_message(SECRET, auth.signature_hash()).unwrap();
    let tx = TxEip7702 {
        chain_id: 143,
        max_fee_per_gas: 5,
        max_priority_fee_per_gas: 2,
        gas_limit: 50_000,
        to: RECIPIENT,
        value: U256::from(7),
        authorization_list: vec![auth.into_signed(signature)],
        ..Default::default()
    };
    let signature = sign_message(SECRET, tx.signature_hash()).unwrap();
    let delegated = TxEnvelope::from(tx.into_signed(signature));
    for tx in [access, dynamic, delegated] {
        let mut case = transfer();
        if tx.tx_type() == TxType::Eip7702 {
            let sender = tx.recover_signer().unwrap().to_string();
            case["postState"][&sender]["nonce"] = json!(2);
            case["postState"][&sender]["code"] =
                json!(format!("0xef0100{}", hex::encode(RECIPIENT)));
        }
        let expected_root = root(&case["postState"]);
        rewrite_block(&mut case, |block| {
            let receipt =
                Receipt { status: true.into(), cumulative_gas_used: 50_000, logs: vec![] }
                    .with_bloom();
            block.header.receipts_root =
                calculate_receipt_root(&[ReceiptEnvelope::from_typed(tx.tx_type(), receipt)]);
            block.body.transactions = vec![tx];
            block.header.transactions_root = calculate_transaction_root(&block.body.transactions);
            block.header.state_root = expected_root;
        });
        let report = check(case);
        assert_eq!(report.cases[0].status, Status::Passed, "{report:?}");
    }
}

#[test]
fn enforces_reserves_with_current_and_historical_senders() {
    let histories: &[&[&[bool]]] = &[
        &[&[true, false]],
        &[&[true], &[false]],
        &[&[true], &[], &[false]],
        &[&[true], &[], &[], &[true]],
    ];
    for network in ["MONAD_EIGHT", "MONAD_NINE"] {
        for history in histories {
            let report = check(reserve_fixture(network, 1_000_000, 3, history));
            assert_eq!(report.cases[0].status, Status::Passed, "{network} {history:?}: {report:?}");
        }
    }
}

#[test]
fn reports_high_fee_spec_divergence_as_a_failure() {
    for network in ["MONAD_EIGHT", "MONAD_NINE"] {
        // Each fee is 10.2 MON: the first transaction may empty reserves, the second may not.
        let report = check(reserve_fixture(
            network,
            100_000_000_000_000_000_000,
            204_000_000_000_000,
            &[&[true], &[false]],
        ));
        assert_eq!(report.cases[0].status, Status::Passed, "{network}: {report:?}");
        // The OCaml specification instead permits the second transaction even above the reserve.
        let report = check(reserve_fixture(
            network,
            100_000_000_000_000_000_000,
            204_000_000_000_000,
            &[&[true], &[true]],
        ));
        assert_eq!(report.cases[0].status, Status::Failed, "{network}: {report:?}");
        assert!(report.cases[0].message.contains("receipts root mismatch"), "{report:?}");
        assert!(report.cases[0].message.contains("ReserveBalanceViolation"), "{report:?}");
    }
}

#[test]
fn applies_withdrawals_after_transactions() {
    let mut case = transfer();
    case["postState"][RECIPIENT.to_string()]["balance"] = json!("2000000007");
    let expected_root = root(&case["postState"]);
    rewrite_block(&mut case, |block| {
        let withdrawals =
            vec![Withdrawal { index: 0, validator_index: 0, address: RECIPIENT, amount: 2 }];
        block.header.withdrawals_root =
            Some(alloy_consensus::proofs::calculate_withdrawals_root(&withdrawals));
        block.header.state_root = expected_root;
        block.body.withdrawals = Some(withdrawals.into());
    });
    let report = check(case);
    assert_eq!(report.cases[0].status, Status::Passed, "{report:?}");
}

#[test]
fn applies_system_calls_in_empty_blocks() {
    let beacon = address!("000f3df6d732807ef1319fb7b8bb8522d0beac02");
    let history = address!("0000f90827f1c53a10cb7a02335b175320002935");
    let mut case = fixture("MONAD_EIGHT", "0x00", true, 0, false, false);
    for address in [beacon, history] {
        // Save the system-call input in slot zero.
        case["pre"][address.to_string()] = account(1, 0, "0x60003560005500");
    }
    let mut genesis = alloy_rlp::decode_exact::<Header>(
        &hex::decode(case["genesisRLP"].as_str().unwrap()).unwrap(),
    )
    .unwrap();
    genesis.state_root = root(&case["pre"]);
    case["genesisBlockHeader"] = header_json(&genesis);
    case["genesisRLP"] = json!(hex::encode_prefixed(alloy_rlp::encode(&genesis)));
    let mut post = case["pre"].clone();
    let mut parent = genesis.clone();
    for number in 1..=2 {
        let beacon_root = B256::repeat_byte(number as u8);
        post[beacon.to_string()]["storage"][B256::ZERO.to_string()] = json!(beacon_root);
        post[history.to_string()]["storage"][B256::ZERO.to_string()] = json!(parent.hash_slow());
        let header = Header {
            number,
            timestamp: number * 15,
            parent_hash: parent.hash_slow(),
            parent_beacon_block_root: Some(beacon_root),
            state_root: root(&post),
            ..genesis.clone()
        };
        let block = Block::<TxEnvelope>::new(
            header.clone(),
            BlockBody { withdrawals: Some(Default::default()), ..Default::default() },
        );
        case["blocks"].as_array_mut().unwrap().push(block_json(&block));
        parent = header;
    }
    case["lastblockhash"] = json!(parent.hash_slow());
    case["postState"] = post;
    let report = check(case);
    assert_eq!(report.cases[0].status, Status::Passed, "{report:?}");
}

#[test]
fn checks_execution_commitments_and_final_accounts() {
    let mut cases = Vec::new();
    let mut bad = transfer();
    rewrite_block(&mut bad, |block| block.header.state_root = B256::ZERO);
    cases.push((bad, "state root mismatch"));
    let mut bad = transfer();
    rewrite_block(&mut bad, |block| block.header.receipts_root = B256::ZERO);
    cases.push((bad, "receipts root mismatch"));
    let mut bad = transfer();
    rewrite_block(&mut bad, |block| block.header.gas_used = 21_000);
    cases.push((bad, "gas used mismatch"));
    let mut bad = transfer();
    rewrite_block(&mut bad, |block| block.header.transactions_root = B256::ZERO);
    cases.push((bad, "transactions root mismatch"));
    let mut bad = transfer();
    rewrite_block(&mut bad, |block| block.header.logs_bloom.0[0] = 1);
    cases.push((bad, "logs bloom mismatch"));
    let mut bad = transfer();
    bad["postState"][RECIPIENT.to_string()]["balance"] = json!("8");
    cases.push((bad, "balance expected 8, got 7"));
    let mut bad = transfer();
    bad["lastblockhash"] = json!(B256::ZERO);
    cases.push((bad, "lastblockhash mismatch"));
    let mut bad = transfer();
    bad["postStateHash"] = json!(B256::ZERO);
    cases.push((bad, "postStateHash mismatch"));
    for (case, reason) in cases {
        let report = check(case);
        assert_eq!(report.cases[0].status, Status::Failed, "{report:?}");
        assert!(report.cases[0].message.contains(reason), "{reason}: {report:?}");
        assert_eq!(report.exit_code(false), 1);
    }
}

#[test]
fn supports_post_hash_and_genesis_block_encoding() {
    let mut case = transfer();
    case["postStateHash"] = case["blocks"][0]["blockHeader"]["stateRoot"].clone();
    case.as_object_mut().unwrap().remove("postState");
    let genesis = alloy_rlp::decode_exact::<Header>(
        &hex::decode(case["genesisRLP"].as_str().unwrap()).unwrap(),
    )
    .unwrap();
    let genesis_block = Block::<TxEnvelope>::new(genesis, BlockBody::default());
    case["genesisRLP"] = json!(hex::encode_prefixed(alloy_rlp::encode(genesis_block)));
    assert_eq!(check(case).cases[0].status, Status::Passed);
}

#[test]
fn unsupported_cases_are_not_execution_failures_or_passes() {
    let mut future = transfer();
    future["network"] = json!("MONAD_TEN");
    future["config"]["network"] = json!("MONAD_TEN");
    let mut invalid = transfer();
    invalid["blocks"][0]["expectException"] = json!("TR_IntrinsicGas");
    let mut extension = transfer();
    extension["config"]["epoch"] = json!(9);
    let mut blob_environment = transfer();
    rewrite_block(&mut blob_environment, |block| block.header.excess_blob_gas = Some(1));
    for case in
        [future, invalid, extension, blob_environment, json!({"env": {}, "transaction": {}})]
    {
        let report = check(case);
        assert_eq!(report.cases[0].status, Status::Unsupported, "{report:?}");
        assert_eq!(report.exit_code(false), 2);
        assert_eq!(report.exit_code(true), 2);
    }
}

#[test]
fn malformed_rlp_and_header_disagreements_fail() {
    let mut broken_rlp = transfer();
    broken_rlp["blocks"][0]["rlp"] = json!("0xf8");
    let mut broken_header = transfer();
    broken_header["blocks"][0]["blockHeader"]["gasUsed"] = json!(1);
    let mut broken_code = transfer();
    broken_code["pre"][RECIPIENT.to_string()]["code"] = json!("0xef01");
    for case in [broken_rlp, broken_header, broken_code] {
        assert_eq!(check(case).cases[0].status, Status::Failed);
    }
}

#[test]
fn discovers_deduplicates_filters_and_continues_after_bad_files() {
    let dir = tempdir().unwrap();
    let nested = dir.path().join("nested");
    fs::create_dir(&nested).unwrap();
    let fixture_file = nested.join("cases.json");
    let mut unsupported = transfer();
    unsupported["blocks"][0]["expectException"] = json!("InvalidBlock");
    write_cases(&fixture_file, json!({"pass": transfer(), "unsupported": unsupported}));
    fs::write(dir.path().join("bad.json"), "{").unwrap();
    fs::write(dir.path().join("index.json"), "not a fixture").unwrap();
    fs::write(dir.path().join("ignored.txt"), "not a fixture").unwrap();
    let report = run(&[dir.path().to_owned(), fixture_file], None).unwrap();
    assert_eq!(
        serde_json::to_value(&report.summary).unwrap(),
        json!({"total": 3, "passed": 1, "failed": 1, "unsupported": 1})
    );
    assert_eq!(report.cases[0].name, "<file>");
    assert_eq!(report.cases[1].name, "pass");
    assert_eq!(report.cases[2].name, "unsupported");
    assert_eq!(report.exit_code(false), 1);
    let filtered = run(std::slice::from_ref(&nested), Some("pass")).unwrap();
    assert_eq!(filtered.summary.total, 1);
    assert_eq!(filtered.exit_code(true), 0);
    assert_eq!(run(&[nested], Some("no-match")).unwrap().exit_code(false), 2);
}

#[test]
fn cli_json_schema_and_exit_codes() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("cases.json");
    let mut unsupported = transfer();
    unsupported["blocks"][0]["expectException"] = json!("InvalidBlock");
    write_cases(&file, json!({"pass": transfer(), "unsupported": unsupported}));
    for (args, expected_code) in [(vec![], 0), (vec!["--fail-on-unsupported"], 1)] {
        let output = Command::new(env!("CARGO_BIN_EXE_monad-fixtures"))
            .arg(&file)
            .arg("--json")
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(expected_code));
        assert!(output.stderr.is_empty());
        let actual = serde_json::from_slice::<Value>(&output.stdout).unwrap();
        assert_eq!(
            actual,
            json!({
                "schema_version": 1, "verification": "valid_block_execution",
                "summary": {"total": 2, "passed": 1, "failed": 0, "unsupported": 1},
                "cases": [
                    {"file": file.canonicalize().unwrap(), "name": "pass", "network": "MONAD_EIGHT", "status": "passed", "message": "1 blocks, 1 transactions; execution commitments matched"},
                    {"file": file.canonicalize().unwrap(), "name": "unsupported", "network": "MONAD_EIGHT", "status": "unsupported", "message": "invalid-block expectations require consensus validation and exact error matching"},
                ],
            })
        );
    }
    for filter in ["no-match", "unsupported"] {
        let output = Command::new(env!("CARGO_BIN_EXE_monad-fixtures"))
            .arg(&file)
            .args(["--json", "--filter", filter])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stderr.is_empty());
    }
}
