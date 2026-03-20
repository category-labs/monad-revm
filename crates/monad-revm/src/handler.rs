//! Monad handler implementation.
//!
//! Key differences from Ethereum:
//! - Gas is charged based on gas_limit, not gas_used (no refunds)
//! - Blob transactions (EIP-4844) are not supported
//! - No header validation for prevrandao or excess_blob_gas (Monad doesn't use these)
use revm::{
    context_interface::{
        result::{HaltReason, InvalidTransaction},
        transaction::{AuthorizationTr, TransactionType},
        Block, Cfg, ContextTr, JournalTr, Transaction,
    },
    handler::{
        evm::FrameTr, handler::EvmTrError, validation, EthFrame, EvmTr, FrameResult, Handler,
        MainnetHandler,
    },
    inspector::{Inspector, InspectorEvmTr, InspectorHandler},
    interpreter::{interpreter::EthInterpreter, interpreter_action::FrameInit},
    primitives::{hardfork::SpecId, U256},
};

use crate::chain::MonadChainContext;
use crate::journal::MonadJournalTr;
use crate::reserve_balance::tracker::ReserveBalanceInit;
use crate::staking::constants::SYSTEM_ADDRESS;

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
        Self { mainnet: MainnetHandler::default() }
    }
}

impl<EVM, ERROR, FRAME> Default for MonadHandler<EVM, ERROR, FRAME> {
    fn default() -> Self {
        Self::new()
    }
}

impl<EVM, ERROR, FRAME> Handler for MonadHandler<EVM, ERROR, FRAME>
where
    EVM: EvmTr<Context: MonadContextTr, Frame = FRAME>,
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
    /// - EIP-7702: system sender cannot appear as an authority in authorization list
    /// - Skips header validation (prevrandao, excess_blob_gas) since Monad doesn't use these
    fn validate_env(&self, evm: &mut Self::Evm) -> Result<(), Self::Error> {
        // Reject blob transactions (EIP-4844) - Monad does not support them
        let tx_type = TransactionType::from(evm.ctx().tx().tx_type());
        if tx_type == TransactionType::Eip4844 {
            return Err(InvalidTransaction::Eip4844NotSupported.into());
        }

        // EIP-7702: reject if SYSTEM_ADDRESS appears as an authority in the authorization list.
        // C++ parity: validate_monad_transaction.cpp checks
        //   `std::ranges::contains(authorities, SYSTEM_SENDER)`
        // This prevents EIP-7702 delegation from/to the system account, which could
        // compromise the integrity of system transactions (syscalls).
        if tx_type == TransactionType::Eip7702 {
            let has_system_authority = evm
                .ctx()
                .tx()
                .authorization_list()
                .any(|auth| auth.authority() == Some(SYSTEM_ADDRESS));
            if has_system_authority {
                return Err(InvalidTransaction::Str(
                    "system transaction sender is authority".into(),
                )
                .into());
            }
        }

        // Validate transaction fields only (skip header checks for prevrandao/excess_blob_gas)
        // Monad doesn't use prevrandao or blob gas, so we call validate_tx_env directly
        // instead of validate_env which includes header checks
        let spec = evm.ctx().cfg().spec().into();
        validation::validate_tx_env(evm.ctx(), spec).map_err(Into::into)
    }

    fn pre_execution(&self, evm: &mut Self::Evm) -> Result<u64, Self::Error> {
        let monad_spec = evm.ctx().cfg().spec();
        evm.ctx().journal_mut().set_monad_spec(monad_spec);
        self.validate_against_state_and_deduct_caller(evm)?;
        self.load_accounts(evm)?;
        let gas = self.apply_eip7702_auth_list(evm)?;

        let sender = evm.ctx().tx().caller();
        let basefee = evm.ctx().block().basefee() as u128;
        let effective_gas_price = evm.ctx().tx().effective_gas_price(basefee);
        let gas_limit = evm.ctx().tx().gas_limit();
        let spec = evm.ctx().cfg().spec();
        let chain = evm.ctx().chain().clone();
        let (sender_is_delegated, sender_account) = {
            let sender_account = evm.ctx().journal_mut().load_account_with_code(sender)?.data;
            (
                sender_account.info.code.as_ref().is_some_and(revm::bytecode::Bytecode::is_eip7702),
                sender_account.clone(),
            )
        };

        evm.ctx().journal_mut().reserve_balance_mut().init(ReserveBalanceInit {
            chain: &chain,
            spec,
            sender,
            effective_gas_price,
            gas_limit,
            sender_is_delegated,
            sender_account: Some(&sender_account),
        });
        Ok(gas)
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

        let eth_spec: SpecId = ctx.cfg().spec().into();
        let coinbase_gas_price = if eth_spec.is_enabled_in(SpecId::LONDON) {
            effective_gas_price.saturating_sub(basefee)
        } else {
            effective_gas_price
        };

        let reward = coinbase_gas_price * gas_limit as u128;
        let beneficiary = ctx.block().beneficiary();

        ctx.journal_mut().balance_incr(beneficiary, U256::from(reward))?;

        Ok(())
    }
}

impl<EVM, ERROR> InspectorHandler for MonadHandler<EVM, ERROR, EthFrame<EthInterpreter>>
where
    EVM: InspectorEvmTr<
        Context: MonadContextTr<Chain = MonadChainContext, Journal: MonadJournalTr>,
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
    use crate::{
        api::builder::MonadBuilder,
        api::default_ctx::{monad_context_with_db, DefaultMonad},
    };
    use revm::{
        context::{result::EVMError, Context, TxEnv},
        context_interface::{
            either::Either,
            transaction::{Authorization, RecoveredAuthority, RecoveredAuthorization},
        },
        database::InMemoryDB,
        inspector::NoOpInspector,
        primitives::{Address, TxKind, B256},
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
            Err(EVMError::Transaction(InvalidTransaction::Eip4844NotSupported))
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

        let ctx = monad_context_with_db(db);
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
        let coinbase_balance =
            result.state.get(&coinbase).map(|a| a.info.balance).unwrap_or_default();

        let expected_reward = U256::from(gas_limit as u128 * gas_price);
        assert_eq!(
            coinbase_balance, expected_reward,
            "Coinbase should receive gas_limit * gas_price = {expected_reward}, got {coinbase_balance}"
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
            revm::state::AccountInfo { balance: initial_balance, ..Default::default() },
        );

        let ctx = monad_context_with_db(db);
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
        let caller_balance = result.state.get(&caller).map(|a| a.info.balance).unwrap_or_default();

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

        let ctx = monad_context_with_db(db);
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
                assert_eq!(gas_refunded, 0, "Refund should be 0 on Monad, got {gas_refunded}");
            }
            _ => panic!("Expected successful transaction"),
        }
    }

    /// Helper to create a RecoveredAuthorization with a specific authority address.
    fn make_recovered_auth(authority: Address) -> RecoveredAuthorization {
        RecoveredAuthorization::new_unchecked(
            Authorization { chain_id: U256::from(1), address: Address::from([0xAA; 20]), nonce: 0 },
            RecoveredAuthority::Valid(authority),
        )
    }

    #[test]
    fn test_eip7702_system_sender_authority_rejected() {
        // C++ parity: validate_monad_transaction.cpp:50
        //   `std::ranges::contains(authorities, SYSTEM_SENDER)`
        //   → MonadTransactionError::SystemTransactionSenderIsAuthority
        let caller = Address::from([1u8; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u128),
                ..Default::default()
            },
        );

        let ctx = monad_context_with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        evm.ctx().block.basefee = 0;

        // Create an EIP-7702 transaction with SYSTEM_ADDRESS as the authority
        let auth = make_recovered_auth(SYSTEM_ADDRESS);
        let tx = TxEnv::builder()
            .caller(caller)
            .gas_limit(100_000)
            .gas_price(1_000_000_000u128)
            .gas_priority_fee(Some(1_000_000_000u128))
            .kind(TxKind::Call(Address::from([3u8; 20])))
            .authorization_list(vec![Either::Right(auth)])
            .build_fill();

        let result = evm.transact(tx);

        // Must be rejected with the system authority error
        match result {
            Err(EVMError::Transaction(InvalidTransaction::Str(msg))) => {
                assert_eq!(
                    msg.as_ref(),
                    "system transaction sender is authority",
                    "Expected system authority error, got: {msg}"
                );
            }
            Err(e) => panic!("Expected Str transaction error, got: {e:?}"),
            Ok(_) => panic!("Expected transaction to be rejected, but it succeeded"),
        }
    }

    #[test]
    fn test_eip7702_non_system_authority_accepted() {
        // EIP-7702 with a non-system authority should pass validation
        let caller = Address::from([1u8; 20]);
        let normal_authority = Address::from([0xBB; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u128),
                ..Default::default()
            },
        );

        let ctx = monad_context_with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        evm.ctx().block.basefee = 0;

        // EIP-7702 tx with a normal (non-system) authority
        let auth = make_recovered_auth(normal_authority);
        let tx = TxEnv::builder()
            .caller(caller)
            .gas_limit(100_000)
            .gas_price(1_000_000_000u128)
            .gas_priority_fee(Some(1_000_000_000u128))
            .kind(TxKind::Call(Address::from([3u8; 20])))
            .authorization_list(vec![Either::Right(auth)])
            .build_fill();

        // Should succeed (not rejected at validation)
        let result = evm.transact(tx);
        assert!(
            result.is_ok(),
            "EIP-7702 with normal authority should be accepted, got: {result:?}"
        );
    }

    #[test]
    fn test_eip7702_system_authority_among_multiple_rejected() {
        // If SYSTEM_ADDRESS is ANY authority in the list, reject the whole transaction
        let caller = Address::from([1u8; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            revm::state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u128),
                ..Default::default()
            },
        );

        let ctx = monad_context_with_db(db);
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        evm.ctx().block.basefee = 0;

        // Mix of normal + system authorities
        let auth_normal = make_recovered_auth(Address::from([0xCC; 20]));
        let auth_system = make_recovered_auth(SYSTEM_ADDRESS);
        let tx = TxEnv::builder()
            .caller(caller)
            .gas_limit(100_000)
            .gas_price(1_000_000_000u128)
            .gas_priority_fee(Some(1_000_000_000u128))
            .kind(TxKind::Call(Address::from([3u8; 20])))
            .authorization_list(vec![Either::Right(auth_normal), Either::Right(auth_system)])
            .build_fill();

        let result = evm.transact(tx);

        // Must be rejected
        assert!(
            matches!(
                result,
                Err(EVMError::Transaction(InvalidTransaction::Str(ref msg)))
                    if msg.as_ref() == "system transaction sender is authority"
            ),
            "Expected system authority error, got: {result:?}"
        );
    }
}
