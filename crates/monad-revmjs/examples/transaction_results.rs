//! Emit native typed-transaction results for comparison with the installed WASM package.

#[path = "../tests/support/transactions.rs"]
mod transactions;

fn main() {
    println!("{}", transactions::report());
}
