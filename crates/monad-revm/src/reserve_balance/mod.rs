//! Reserve-balance precompile.

pub mod abi;
pub mod error;
pub mod interface;
pub mod tracker;

use crate::{api::exec::MonadContextTr, journal::MonadJournalTr, MonadSpecId};
use abi::{DIPPED_INTO_RESERVE_GAS, DIPPED_INTO_RESERVE_SELECTOR, RESERVE_BALANCE_ADDRESS};
use error::ReserveBalanceError;
use revm::{
    context::Cfg,
    context_interface::ContextTr,
    interpreter::{CallInputs, CallScheme, Gas, InstructionResult, InterpreterResult},
    primitives::Bytes,
};

/// Runs the reserve-balance precompile.
pub fn run_reserve_balance_precompile<CTX>(
    context: &mut CTX,
    inputs: &CallInputs,
) -> Result<Option<InterpreterResult>, String>
where
    CTX: MonadContextTr,
{
    if inputs.bytecode_address != RESERVE_BALANCE_ADDRESS
        || !MonadSpecId::MonadNine.is_enabled_in(context.cfg().spec())
    {
        return Ok(None);
    }

    if inputs.scheme != CallScheme::Call || inputs.is_static {
        return Ok(Some(revert_result(inputs.gas_limit, Bytes::new())));
    }

    if inputs.gas_limit < DIPPED_INTO_RESERVE_GAS {
        return Ok(Some(oog_result(inputs.gas_limit)));
    }

    let input = inputs.input.bytes(context);

    let selector = match input.get(..4) {
        Some(selector) if selector == DIPPED_INTO_RESERVE_SELECTOR => selector,
        _ => {
            return Ok(Some(revert_error_result(
                inputs.gas_limit,
                ReserveBalanceError::MethodNotSupported,
            )));
        }
    };

    let _ = selector;

    if !inputs.call_value().is_zero() {
        return Ok(Some(revert_error_result(inputs.gas_limit, ReserveBalanceError::ValueNonZero)));
    }

    if input.len() > 4 {
        return Ok(Some(revert_error_result(inputs.gas_limit, ReserveBalanceError::InvalidInput)));
    }

    let has_violation = context.journal().reserve_balance().has_violation();
    Ok(Some(success_result(inputs.gas_limit, DIPPED_INTO_RESERVE_GAS, has_violation)))
}

fn success_result(gas_limit: u64, gas_used: u64, value: bool) -> InterpreterResult {
    let mut gas = Gas::new(gas_limit);
    let _ = gas.record_cost(gas_used);
    let mut output = [0u8; 32];
    output[31] = u8::from(value);
    InterpreterResult {
        result: InstructionResult::Return,
        gas,
        output: Bytes::copy_from_slice(&output),
    }
}

fn revert_error_result(gas_limit: u64, error: ReserveBalanceError) -> InterpreterResult {
    revert_result(gas_limit, Bytes::copy_from_slice(error.to_string().as_bytes()))
}

fn revert_result(gas_limit: u64, output: Bytes) -> InterpreterResult {
    let mut gas = Gas::new(gas_limit);
    let _ = gas.record_cost(gas_limit);
    InterpreterResult { result: InstructionResult::Revert, gas, output }
}

fn oog_result(gas_limit: u64) -> InterpreterResult {
    let mut gas = Gas::new(gas_limit);
    let _ = gas.record_cost(gas_limit);
    InterpreterResult { result: InstructionResult::PrecompileOOG, gas, output: Bytes::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::default_ctx::{monad_context_with_db, DefaultMonad},
        journal::MonadJournalTr,
        reserve_balance::{
            abi::RESERVE_BALANCE_ADDRESS, interface::IReserveBalance::dippedIntoReserveCall,
            tracker::ReserveBalanceInit,
        },
        MonadCfgEnv, MonadSpecId,
    };
    use alloy_sol_types::SolCall;
    use revm::{
        context_interface::JournalTr,
        database::InMemoryDB,
        interpreter::{CallInput, CallValue},
        primitives::{Address, U256},
        state::AccountInfo,
    };

    fn reserve_balance_call_inputs_with_input(
        scheme: CallScheme,
        is_static: bool,
        value: CallValue,
        gas_limit: u64,
        input: Bytes,
    ) -> CallInputs {
        CallInputs {
            input: CallInput::Bytes(input),
            return_memory_offset: 0..0,
            gas_limit,
            bytecode_address: RESERVE_BALANCE_ADDRESS,
            known_bytecode: None,
            target_address: RESERVE_BALANCE_ADDRESS,
            caller: Address::ZERO,
            value,
            scheme,
            is_static,
        }
    }

    fn reserve_balance_call_inputs(
        scheme: CallScheme,
        is_static: bool,
        value: CallValue,
    ) -> CallInputs {
        reserve_balance_call_inputs_with_input(
            scheme,
            is_static,
            value,
            100_000,
            Bytes::from(dippedIntoReserveCall::SELECTOR.to_vec()),
        )
    }

    #[test]
    fn test_selector_matches_interface() {
        assert_eq!(DIPPED_INTO_RESERVE_SELECTOR, dippedIntoReserveCall::SELECTOR);
    }

    #[test]
    fn test_non_reserve_balance_address_returns_none() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let mut inputs =
            reserve_balance_call_inputs(CallScheme::Call, false, CallValue::Transfer(U256::ZERO));
        inputs.bytecode_address = Address::ZERO;

        let result = run_reserve_balance_precompile(&mut ctx, &inputs).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_monad_eight_returns_none() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadEight));
        let inputs =
            reserve_balance_call_inputs(CallScheme::Call, false, CallValue::Transfer(U256::ZERO));

        let result = run_reserve_balance_precompile(&mut ctx, &inputs).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delegatecall_rejected() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs = reserve_balance_call_inputs(
            CallScheme::DelegateCall,
            false,
            CallValue::Transfer(U256::ZERO),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert!(result.output.is_empty());
    }

    #[test]
    fn test_call_in_static_context_rejected() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs =
            reserve_balance_call_inputs(CallScheme::Call, true, CallValue::Transfer(U256::ZERO));

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert!(result.output.is_empty());
    }

    #[test]
    fn test_oog_before_selector_decode() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs = reserve_balance_call_inputs_with_input(
            CallScheme::Call,
            false,
            CallValue::Transfer(U256::ZERO),
            DIPPED_INTO_RESERVE_GAS - 1,
            Bytes::from(dippedIntoReserveCall::SELECTOR.to_vec()),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::PrecompileOOG);
    }

    #[test]
    fn test_short_input_hits_fallback() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs = reserve_balance_call_inputs_with_input(
            CallScheme::Call,
            false,
            CallValue::Transfer(U256::ZERO),
            100_000,
            Bytes::from(vec![0xde, 0xad, 0xbe]),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert_eq!(result.output, Bytes::from("method not supported"));
    }

    #[test]
    fn test_unknown_selector_hits_fallback() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs = reserve_balance_call_inputs_with_input(
            CallScheme::Call,
            false,
            CallValue::Transfer(U256::ZERO),
            100_000,
            Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert_eq!(result.output, Bytes::from("method not supported"));
    }

    #[test]
    fn test_nonzero_value_rejected() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs = reserve_balance_call_inputs(
            CallScheme::Call,
            false,
            CallValue::Transfer(U256::from(1)),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert_eq!(result.output, Bytes::from("value is nonzero"));
    }

    #[test]
    fn test_extra_input_rejected() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let mut input = dippedIntoReserveCall::SELECTOR.to_vec();
        input.extend_from_slice(&[0u8; 32]);
        let inputs = reserve_balance_call_inputs_with_input(
            CallScheme::Call,
            false,
            CallValue::Transfer(U256::ZERO),
            100_000,
            Bytes::from(input),
        );

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Revert);
        assert_eq!(result.output, Bytes::from("input is invalid"));
    }

    #[test]
    fn test_success_returns_false_when_no_violation_is_tracked() {
        let mut ctx = crate::api::default_ctx::MonadContext::monad()
            .with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let inputs =
            reserve_balance_call_inputs(CallScheme::Call, false, CallValue::Transfer(U256::ZERO));

        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Return);
        assert_eq!(result.output.len(), 32);
        assert_eq!(result.output[31], 0);
    }

    #[test]
    fn test_success_returns_true_when_tracker_has_violation() {
        let sender = Address::from([0x11; 20]);
        let tracked = Address::from([0x22; 20]);
        let recipient = Address::from([0x33; 20]);

        let mut db = InMemoryDB::default();
        db.insert_account_info(
            sender,
            AccountInfo { balance: U256::from(1_000_000u64), ..Default::default() },
        );
        db.insert_account_info(
            tracked,
            AccountInfo { balance: U256::from(1_000u64), ..Default::default() },
        );

        let mut ctx =
            monad_context_with_db(db).with_cfg(MonadCfgEnv::new_with_spec(MonadSpecId::MonadNine));
        let chain = ctx.chain().clone();
        ctx.journal_mut().reserve_balance_mut().init(ReserveBalanceInit {
            chain: &chain,
            spec: MonadSpecId::MonadNine,
            sender,
            effective_gas_price: 0,
            gas_limit: 0,
            sender_is_delegated: false,
            sender_account: None,
        });
        ctx.journal_mut()
            .transfer(tracked, recipient, U256::from(1u64))
            .expect("transfer should succeed");

        let inputs =
            reserve_balance_call_inputs(CallScheme::Call, false, CallValue::Transfer(U256::ZERO));
        let result = run_reserve_balance_precompile(&mut ctx, &inputs)
            .unwrap()
            .expect("should dispatch reserve-balance precompile");
        assert_eq!(result.result, InstructionResult::Return);
        assert_eq!(result.output.len(), 32);
        assert_eq!(result.output[31], 1);
    }
}
