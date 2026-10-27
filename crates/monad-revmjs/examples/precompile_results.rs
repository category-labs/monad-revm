//! Emit native execution results for comparison with the installed WASM package.

#[path = "../tests/support/precompiles.rs"]
mod precompiles;

fn main() {
    println!("{}", precompiles::report());
}
