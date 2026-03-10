//! Monad reserve-balance precompile Solidity interface.

alloy_sol_types::sol! {
    /// Monad reserve-balance precompile interface at address 0x1001.
    interface IReserveBalance {
        function dippedIntoReserve() external returns (bool);
    }
}
