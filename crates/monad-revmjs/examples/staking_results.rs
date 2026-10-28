//! Emit native staking results for comparison with the installed WASM package.

#[path = "../tests/support/staking.rs"]
mod staking;

fn main() {
    println!("{}", staking::report());
}
