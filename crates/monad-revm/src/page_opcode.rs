//! MIP-8 storage opcode handlers.

use crate::{api::exec::MonadContextTr, journal::MonadJournalTr, page::StoragePageKey};
use revm::{
    context_interface::host::LoadError,
    interpreter::{
        interpreter_types::{InputsTr, InterpreterTypes, RuntimeFlag, StackTr},
        InstructionContext, InstructionResult,
    },
    primitives::hardfork::SpecId::*,
};

/// MIP-8 SSTORE handler with page-based gas scheduling.
pub fn sstore<WIRE: InterpreterTypes, H: MonadContextTr + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::require_non_staticcall!(context.interpreter);
    revm_interpreter::popn!([index, value], context.interpreter);

    let target = context.interpreter.input.target_address();
    let slot = index;
    let spec_id = context.interpreter.runtime_flag.spec_id();

    // EIP-2200: fail if gasleft is at or below the stipend before charging SSTORE.
    if spec_id.is_enabled_in(ISTANBUL)
        && context.interpreter.gas.remaining() <= context.host.gas_params().call_stipend()
    {
        context.interpreter.halt(InstructionResult::ReentrancySentryOOG);
        return;
    }

    let cold_read_cost = context.host.gas_params().cold_storage_additional_cost();
    let skip_cold = context.interpreter.gas.remaining() < cold_read_cost;
    let state_load = match context.host.sstore_skip_cold_load(target, slot, value, skip_cold) {
        Ok(load) => load,
        Err(LoadError::ColdLoadSkipped) => return context.interpreter.halt_oog(),
        Err(LoadError::DBError) => return context.interpreter.halt_fatal(),
    };

    let mut gas_cost = 0;
    if state_load.is_cold {
        gas_cost += cold_read_cost;
    }

    let page_key = StoragePageKey::from_slot(target, slot);
    gas_cost += context.host.journal_mut().page_access_mut().sstore_gas(page_key, &state_load.data);

    revm_interpreter::gas!(context.interpreter, gas_cost);
}
