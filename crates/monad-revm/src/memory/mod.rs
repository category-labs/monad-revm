//! MIP-3: Linear memory cost model with 8 MB pooled transaction limit.
//!
//! Replaces the Ethereum quadratic memory cost formula (`3·n + n²/512`) with
//! a linear formula (`n / 2`, where n = word count) and enforces an 8 MB
//! global memory cap shared across the call stack.
//!
//! The 8 MB limit and memory pooling are handled by REVM's existing
//! [`SharedMemory`] infrastructure — see [`crate::cfg`] for the limit value.
//!
//! This module provides a custom `monad_resize_memory` function and
//! replacement opcode handlers that use the linear cost model.

pub mod opcodes;

use revm::interpreter::{
    interpreter::num_words, interpreter_types::MemoryTr, Gas, InstructionResult,
};

/// MIP-3 memory expansion cost: `num_words / 2`.
///
/// For 8 MB (262 144 words) the total cost is 131 072 gas.
#[inline]
pub const fn monad_memory_cost(num_words: usize) -> u64 {
    (num_words as u64) >> 1
}

/// Monad-specific memory resize using the MIP-3 linear cost model.
///
/// Drop-in replacement for [`revm::interpreter::interpreter::resize_memory`]
/// that charges `new_words >> 1` instead of `3·n + n²/512`.
///
/// The 8 MB pooled limit is still enforced by the `memory_limit` feature
/// through [`MemoryTr::limit_reached`].
#[inline]
pub fn monad_resize_memory<Memory: MemoryTr>(
    gas: &mut Gas,
    memory: &mut Memory,
    offset: usize,
    len: usize,
) -> Result<(), InstructionResult> {
    #[cfg(feature = "memory_limit")]
    if memory.limit_reached(offset, len) {
        return Err(InstructionResult::MemoryLimitOOG);
    }

    let new_num_words = num_words(offset.saturating_add(len));
    if new_num_words > gas.memory().words_num {
        return monad_resize_memory_cold(gas, memory, new_num_words);
    }

    Ok(())
}

#[cold]
#[inline(never)]
fn monad_resize_memory_cold<Memory: MemoryTr>(
    gas: &mut Gas,
    memory: &mut Memory,
    new_num_words: usize,
) -> Result<(), InstructionResult> {
    let total_cost = monad_memory_cost(new_num_words);
    // SAFETY: `new_num_words > words_num` guarantees `total_cost >= expansion_cost`,
    // so `set_words_num` always returns `Some`.
    let delta =
        unsafe { gas.memory_mut().set_words_num(new_num_words, total_cost).unwrap_unchecked() };

    if !gas.record_cost(delta) {
        return Err(InstructionResult::MemoryOOG);
    }
    memory.resize(new_num_words * 32);
    Ok(())
}

/// Resizes interpreter memory using the MIP-3 linear cost model.
///
/// Halts the interpreter and returns `$ret` on failure.
macro_rules! resize_memory_mip3 {
    ($interpreter:expr, $offset:expr, $len:expr) => {
        resize_memory_mip3!($interpreter, $offset, $len, ())
    };
    ($interpreter:expr, $offset:expr, $len:expr, $ret:expr) => {
        if let Err(result) = $crate::memory::monad_resize_memory(
            &mut $interpreter.gas,
            &mut $interpreter.memory,
            $offset,
            $len,
        ) {
            $interpreter.halt(result);
            return $ret;
        }
    };
}

pub(crate) use resize_memory_mip3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_memory_cost() {
        // 0 words → 0 gas
        assert_eq!(monad_memory_cost(0), 0);
        // 1 word (32 bytes) → 0 gas (1 >> 1 = 0)
        assert_eq!(monad_memory_cost(1), 0);
        // 2 words (64 bytes) → 1 gas
        assert_eq!(monad_memory_cost(2), 1);
        // 10 words → 5 gas
        assert_eq!(monad_memory_cost(10), 5);
        // 8 MB = 262_144 words → 131_072 gas
        assert_eq!(monad_memory_cost(262_144), 131_072);
    }

    #[test]
    fn test_monad_memory_cost_is_linear() {
        let cost_100 = monad_memory_cost(100);
        let cost_200 = monad_memory_cost(200);
        let cost_400 = monad_memory_cost(400);
        // Linear: doubling words doubles cost
        assert_eq!(cost_200, cost_100 * 2);
        assert_eq!(cost_400, cost_200 * 2);
    }

    #[test]
    fn test_monad_vs_eth_memory_cost() {
        // Ethereum cost at 8 MB (262_144 words):
        // 3 * 262_144 + 262_144² / 512 = 786_432 + 134_217_728 = 135_004_160
        let eth_cost = 3u64 * 262_144 + (262_144u64 * 262_144) / 512;
        let monad_cost = monad_memory_cost(262_144);
        // MIP-3 is drastically cheaper
        assert_eq!(monad_cost, 131_072);
        assert!(monad_cost < eth_cost);
    }
}
