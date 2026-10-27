//! Shared known-answer precompile vectors for both supported Monad forks.

#[path = "support/precompiles.rs"]
mod precompiles;

#[test]
fn precompile_outputs_gas_and_failure_semantics() {
    precompiles::report();
}
