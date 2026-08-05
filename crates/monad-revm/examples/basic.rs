//! Executes a minimal Monad transaction against an in-memory database.

use monad_revm::{monad_context_with_db, MonadBuilder};
use revm::{
    context::TxEnv,
    database::InMemoryDB,
    primitives::{Address, TxKind, U256},
    state::AccountInfo,
    ExecuteEvm,
};

fn main() {
    let caller = Address::from([0x11; 20]);
    let recipient = Address::from([0x22; 20]);

    let mut db = InMemoryDB::default();
    db.insert_account_info(
        caller,
        AccountInfo { balance: U256::from(1_000_000), ..Default::default() },
    );

    let context = monad_context_with_db(db);
    let mut evm = context.build_monad();

    let tx = TxEnv::builder()
        .caller(caller)
        .kind(TxKind::Call(recipient))
        .gas_limit(21_000)
        .gas_price(0)
        .build_fill();

    let _result = evm.transact(tx).expect("transaction should execute");
}
