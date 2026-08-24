//! MIP-8 storage opcode handlers.

use crate::{api::exec::MonadContextTr, page::StoragePageKey, MonadJournalTr};
use revm::interpreter::{
    interpreter_types::{InputsTr, InterpreterTypes, RuntimeFlag, StackTr},
    InstructionContext, InstructionExecResult as Result, InstructionResult,
};
use revm::{context_interface::Host, primitives::hardfork::SpecId::*};

/// Executes SSTORE with MIP-8 page-based gas accounting.
pub fn sstore<WIRE: InterpreterTypes, H: MonadContextTr + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) -> Result {
    revm_interpreter::require_non_staticcall!(context.interpreter);
    revm_interpreter::popn!([index, value], context.interpreter);

    let target = context.interpreter.input.target_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();

    if spec_id.is_enabled_in(ISTANBUL)
        && context.interpreter.gas.remaining() <= context.host.gas_params().call_stipend()
    {
        return Err(InstructionResult::ReentrancySentryOOG);
    }

    let state_load = if spec_id.is_enabled_in(BERLIN) {
        let skip_cold_load =
            context.interpreter.gas.remaining() < context.host.gas_params().cold_storage_cost();
        context.host.sstore_skip_cold_load(target, index, value, skip_cold_load)?
    } else {
        context.host.sstore(target, index, value).ok_or(InstructionResult::FatalExternalError)?
    };

    let key = StoragePageKey::from_slot(target, index);
    let mut gas = context.host.journal_mut().page_access_mut().sstore_gas(key, &state_load.data);
    if state_load.is_cold {
        gas += context.host.gas_params().cold_storage_additional_cost();
    }
    revm_interpreter::gas!(context.interpreter, gas);
    Ok(())
}
