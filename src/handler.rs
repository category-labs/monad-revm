//! Monad handler implementation.
//!
//! Key differences from Ethereum:
//! - Gas is charged based on gas_limit, not gas_used (no refunds)
//! - Blob transactions (EIP-4844) are not supported
//! - No header validation for prevrandao or excess_blob_gas (Monad doesn't use these)
use revm::{
    context_interface::{
        result::{HaltReason, InvalidTransaction},
        transaction::TransactionType,
        Block, Cfg, ContextTr, JournalTr, Transaction,
    },
    handler::{
        evm::FrameTr, handler::EvmTrError, validation, EthFrame, EvmTr, FrameResult, Handler,
        MainnetHandler,
    },
    inspector::{Inspector, InspectorEvmTr, InspectorHandler},
    interpreter::{interpreter::EthInterpreter, interpreter_action::FrameInit},
    primitives::{hardfork::SpecId, U256},
    state::EvmState,
};

use crate::api::exec::MonadContextTr;

/// Monad handler extends [`Handler`] with Monad-specific gas handling.
///
/// Key difference: Gas is charged based on gas_limit rather than gas_used.
/// This is a DOS-prevention measure for Monad's asynchronous execution.

#[derive(Debug, Clone)]
pub struct MonadHandler<EVM, ERROR, FRAME> {
    /// Mainnet handler allows us to use functions from the mainnet handler inside monad handler.
    /// So we dont duplicate the logic
    pub mainnet: MainnetHandler<EVM, ERROR, FRAME>,
}

impl<EVM, ERROR, FRAME> MonadHandler<EVM, ERROR, FRAME> {
    /// Create a new Monad handler.
    pub fn new() -> Self {
        Self {
            mainnet: MainnetHandler::default(),
        }
    }
}

impl<EVM, ERROR, FRAME> Default for MonadHandler<EVM, ERROR, FRAME> {
    fn default() -> Self {
        Self::new()
    }
}

impl<EVM, ERROR, FRAME> Handler for MonadHandler<EVM, ERROR, FRAME>
where
    EVM: EvmTr<Context: ContextTr<Journal: JournalTr<State = EvmState>>, Frame = FRAME>,
    ERROR: EvmTrError<EVM>,
    FRAME: FrameTr<FrameResult = FrameResult, FrameInit = FrameInit>,
{
    type Evm = EVM;
    type Error = ERROR;
    type HaltReason = HaltReason;

    /// Validates transaction and configuration fields.
    ///
    /// Monad-specific validation:
    /// - Blob transactions (EIP-4844) are not supported
    /// - Skips header validation (prevrandao, excess_blob_gas) since Monad doesn't use these
    fn validate_env(&self, evm: &mut Self::Evm) -> Result<(), Self::Error> {
        // Reject blob transactions (EIP-4844) - Monad does not support them
        let tx_type = TransactionType::from(evm.ctx().tx().tx_type());
        if tx_type == TransactionType::Eip4844 {
            return Err(InvalidTransaction::Eip4844NotSupported.into());
        }

        // Validate transaction fields only (skip header checks for prevrandao/excess_blob_gas)
        // Monad doesn't use prevrandao or blob gas, so we call validate_tx_env directly
        // instead of validate_env which includes header checks
        let spec = evm.ctx().cfg().spec().into();
        validation::validate_tx_env(evm.ctx(), spec).map_err(Into::into)
    }

    // Disable gas refunds
    fn refund(
        &self,
        _evm: &mut Self::Evm,
        exec_result: &mut <<Self::Evm as EvmTr>::Frame as FrameTr>::FrameResult,
        _eip7702_refund: i64,
    ) {
        exec_result.gas_mut().set_refund(0);
    }

    // Don't reimburse caller
    fn reimburse_caller(
        &self,
        _evm: &mut Self::Evm,
        _exec_result: &mut <<Self::Evm as EvmTr>::Frame as FrameTr>::FrameResult,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    // Pay full gas_limit to beneficiary
    fn reward_beneficiary(
        &self,
        evm: &mut Self::Evm,
        _exec_result: &mut <<Self::Evm as EvmTr>::Frame as FrameTr>::FrameResult,
    ) -> Result<(), Self::Error> {
        // a modified version of post_execution::reward_beneficiary() to charge based on gas_limit() not gas.used()
        let ctx = evm.ctx();

        let gas_limit = ctx.tx().gas_limit();
        let basefee = ctx.block().basefee() as u128;
        let effective_gas_price = ctx.tx().effective_gas_price(basefee);

        let coinbase_gas_price = if ctx.cfg().spec().into().is_enabled_in(SpecId::LONDON) {
            effective_gas_price.saturating_sub(basefee)
        } else {
            effective_gas_price
        };

        let reward = coinbase_gas_price * gas_limit as u128;
        let beneficiary = ctx.block().beneficiary();

        ctx.journal_mut()
            .balance_incr(beneficiary, U256::from(reward))?;

        Ok(())
    }
}

impl<EVM, ERROR> InspectorHandler for MonadHandler<EVM, ERROR, EthFrame<EthInterpreter>>
where
    EVM: InspectorEvmTr<
        Context: MonadContextTr,
        Frame = EthFrame<EthInterpreter>,
        Inspector: Inspector<<<Self as Handler>::Evm as EvmTr>::Context, EthInterpreter>,
    >,
    ERROR: EvmTrError<EVM>,
{
    type IT = EthInterpreter;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{api::builder::MonadBuilder, api::default_ctx::DefaultMonad};
    use revm::{
        context::{result::EVMError, Context, TxEnv},
        database::InMemoryDB,
        inspector::NoOpInspector,
        primitives::{Address, B256},
        ExecuteEvm,
    };

    #[test]
    fn test_blob_transaction_rejected() {
        let ctx = Context::monad();
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});

        // Create a blob transaction (EIP-4844)
        let tx = TxEnv::builder()
            .tx_type(Some(3)) // EIP-4844 blob transaction type
            .gas_priority_fee(Some(10))
            .blob_hashes(vec![B256::from([5u8; 32])])
            .build_fill();

        let result = evm.transact(tx);

        // Verify that blob transactions are rejected
        assert!(matches!(
            result,
            Err(EVMError::Transaction(
                InvalidTransaction::Eip4844NotSupported
            ))
        ));
    }

    #[test]
    fn test_reward_beneficiary_charges_full_gas_limit() {
        // Setup: Create EVM with a specific coinbase and caller
        let caller = Address::from([1u8; 20]);
        let coinbase = Address::from([2u8; 20]);
        let gas_limit = 100_000u64;
        let gas_price = 1_000_000_000u128; // 1 gwei

        let mut db = InMemoryDB::default();
        // Give caller enough balance for gas
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: U256::from(gas_limit as u128 * gas_price * 2),
                ..Default::default()
            },
        );
        // Coinbase starts with 0 balance
        db.insert_account_info(coinbase, revm::state::AccountInfo::default());

        let ctx = Context::monad().with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});

        // Set block beneficiary
        evm.ctx().block.beneficiary = coinbase;
        evm.ctx().block.basefee = 0;

        // Simple transfer transaction - uses ~21000 gas
        let tx = TxEnv::builder()
            .caller(caller)
            .to(Address::from([3u8; 20]))
            .value(U256::from(1))
            .gas_limit(gas_limit)
            .gas_price(gas_price)
            .build_fill();

        let result = evm.transact(tx).expect("Transaction should succeed");

        // Verify coinbase received gas_limit * gas_price, NOT gas_used * gas_price
        let coinbase_balance = result
            .state
            .get(&coinbase)
            .map(|a| a.info.balance)
            .unwrap_or_default();

        let expected_reward = U256::from(gas_limit as u128 * gas_price);
        assert_eq!(
            coinbase_balance, expected_reward,
            "Coinbase should receive gas_limit * gas_price = {}, got {}",
            expected_reward, coinbase_balance
        );
    }

    #[test]
    fn test_no_gas_refund_for_unused_gas() {
        // Setup: Execute a transaction that uses less gas than gas_limit
        let caller = Address::from([1u8; 20]);
        let gas_limit = 100_000u64;
        let gas_price = 1_000_000_000u128; // 1 gwei
        let initial_balance = U256::from(1_000_000_000_000_000_000u128); // 1 ETH

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: initial_balance,
                ..Default::default()
            },
        );

        let ctx = Context::monad().with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        evm.ctx().block.basefee = 0;

        // Simple transfer - uses ~21000 gas, but we set gas_limit to 100000
        let tx = TxEnv::builder()
            .caller(caller)
            .to(Address::from([3u8; 20]))
            .value(U256::from(1000))
            .gas_limit(gas_limit)
            .gas_price(gas_price)
            .build_fill();

        let result = evm.transact(tx).expect("Transaction should succeed");

        // On Monad, caller should NOT be reimbursed for unused gas
        // Final balance = initial - (gas_limit * gas_price) - value_sent
        let caller_balance = result
            .state
            .get(&caller)
            .map(|a| a.info.balance)
            .unwrap_or_default();

        let gas_cost = U256::from(gas_limit as u128 * gas_price);
        let value_sent = U256::from(1000);
        let expected_balance = initial_balance - gas_cost - value_sent;

        assert_eq!(
            caller_balance,
            expected_balance,
            "Caller should be charged full gas_limit, not gas_used. \
             Expected {}, got {}. Gas used was {}",
            expected_balance,
            caller_balance,
            result.result.gas_used()
        );

        // Verify gas_used < gas_limit (to confirm unused gas wasn't refunded)
        assert!(
            result.result.gas_used() < gas_limit,
            "Gas used ({}) should be less than gas_limit ({})",
            result.result.gas_used(),
            gas_limit
        );
    }

    #[test]
    fn test_refund_counter_is_zero() {
        use revm::context_interface::result::ExecutionResult;

        // Test that the refund counter is always set to 0
        let caller = Address::from([1u8; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u128),
                ..Default::default()
            },
        );

        let ctx = Context::monad().with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        evm.ctx().block.basefee = 0;

        let tx = TxEnv::builder()
            .caller(caller)
            .to(Address::from([3u8; 20]))
            .value(U256::from(1))
            .gas_limit(50_000)
            .gas_price(1_000_000_000u128)
            .build_fill();

        let result = evm.transact(tx).expect("Transaction should succeed");

        // Verify refund is 0 (Monad disables refunds)
        match result.result {
            ExecutionResult::Success { gas_refunded, .. } => {
                assert_eq!(
                    gas_refunded, 0,
                    "Refund should be 0 on Monad, got {}",
                    gas_refunded
                );
            }
            _ => panic!("Expected successful transaction"),
        }
    }
}
