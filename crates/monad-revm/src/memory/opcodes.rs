//! MIP-3 opcode handlers that use linear memory expansion cost.
//!
//! Each function mirrors its REVM counterpart but calls
//! [`resize_memory_mip3!`] (cost = `words / 2`) instead of the standard
//! `resize_memory!` (cost = `3·words + words²/512`).

use super::resize_memory_mip3;
use core::cmp::max;
use revm::interpreter::{
    instructions::contract::load_acc_and_calc_gas,
    interpreter::Interpreter,
    interpreter_types::{
        InputsTr, InterpreterTypes, LegacyBytecode, LoopControl, MemoryTr, ReturnData, RuntimeFlag,
        StackTr,
    },
    CallInput, CallInputs, CallScheme, CallValue, CreateInputs, InstructionContext,
    InstructionResult, InterpreterAction,
};
use revm::{
    context_interface::{cfg::GasParams, host::LoadError, CreateScheme, Host},
    interpreter::interpreter_action::FrameInput,
    primitives::{self, hardfork::SpecId, Address, Bytes, B256, KECCAK_EMPTY, U256},
};
use std::boxed::Box;

// Re-export internal macros / traits needed by revm_interpreter macros.
use revm_interpreter::{_count, instructions::utility::IntoAddress};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy-cost + memory resize using MIP-3 linear cost.
///
/// Mirrors [`revm::interpreter::instructions::system::copy_cost_and_memory_resize`].
fn monad_copy_cost_and_memory_resize(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    gas_params: &GasParams,
    memory_offset: U256,
    len: usize,
) -> Option<usize> {
    revm_interpreter::gas!(interpreter, gas_params.copy_cost(len), None);
    if len == 0 {
        return None;
    }
    let memory_offset = revm_interpreter::as_usize_or_fail_ret!(interpreter, memory_offset, None);
    resize_memory_mip3!(interpreter, memory_offset, len, None);
    Some(memory_offset)
}

/// Return/revert helper using MIP-3 linear cost.
///
/// Mirrors [`revm::interpreter::instructions::control::return_inner`].
#[inline]
fn monad_return_inner(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    instruction_result: InstructionResult,
) {
    revm_interpreter::popn!([offset, len], interpreter);
    let len = revm_interpreter::as_usize_or_fail!(interpreter, len);
    let mut output = Bytes::default();
    if len != 0 {
        let offset = revm_interpreter::as_usize_or_fail!(interpreter, offset);
        if let Err(result) = crate::memory::monad_resize_memory(
            &mut interpreter.gas,
            &mut interpreter.memory,
            offset,
            len,
        ) {
            interpreter.halt(result);
            return;
        }
        output = interpreter.memory.slice_len(offset, len).to_vec().into();
    }

    interpreter.bytecode.set_action(InterpreterAction::new_return(
        instruction_result,
        output,
        interpreter.gas,
    ));
}

/// Memory input/output ranges for CALL instructions using MIP-3 linear cost.
///
/// Mirrors [`revm::interpreter::instructions::contract::get_memory_input_and_out_ranges`]
/// but uses MIP-3 for memory expansion.
#[inline]
fn monad_get_memory_input_and_out_ranges(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    _gas_params: &GasParams,
) -> Option<(core::ops::Range<usize>, core::ops::Range<usize>)> {
    revm_interpreter::popn!([in_offset, in_len, out_offset, out_len], interpreter, None);

    let mut in_range = monad_call_resize_memory(interpreter, in_offset, in_len)?;

    if !in_range.is_empty() {
        let offset = interpreter.memory.local_memory_offset();
        in_range = in_range.start.saturating_add(offset)..in_range.end.saturating_add(offset);
    }

    let ret_range = monad_call_resize_memory(interpreter, out_offset, out_len)?;
    Some((in_range, ret_range))
}

/// Resize memory for CALL arguments and return range using MIP-3 linear cost.
///
/// Mirrors [`revm::interpreter::instructions::contract::resize_memory`].
#[inline]
fn monad_call_resize_memory(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    offset: U256,
    len: U256,
) -> Option<core::ops::Range<usize>> {
    let len = revm_interpreter::as_usize_or_fail_ret!(interpreter, len, None);
    let offset = if len != 0 {
        let offset = revm_interpreter::as_usize_or_fail_ret!(interpreter, offset, None);
        resize_memory_mip3!(interpreter, offset, len, None);
        offset
    } else {
        usize::MAX
    };
    Some(offset..offset + len)
}

// ---------------------------------------------------------------------------
// Memory opcodes
// ---------------------------------------------------------------------------

/// MLOAD with MIP-3 linear memory cost.
pub fn mload<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    revm_interpreter::popn_top!([], top, context.interpreter);
    let offset = revm_interpreter::as_usize_or_fail!(context.interpreter, top);
    resize_memory_mip3!(context.interpreter, offset, 32);
    *top =
        U256::try_from_be_slice(context.interpreter.memory.slice_len(offset, 32).as_ref()).unwrap()
}

/// MSTORE with MIP-3 linear memory cost.
pub fn mstore<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    revm_interpreter::popn!([offset, value], context.interpreter);
    let offset = revm_interpreter::as_usize_or_fail!(context.interpreter, offset);
    resize_memory_mip3!(context.interpreter, offset, 32);
    context.interpreter.memory.set(offset, &value.to_be_bytes::<32>());
}

/// MSTORE8 with MIP-3 linear memory cost.
pub fn mstore8<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    revm_interpreter::popn!([offset, value], context.interpreter);
    let offset = revm_interpreter::as_usize_or_fail!(context.interpreter, offset);
    resize_memory_mip3!(context.interpreter, offset, 1);
    context.interpreter.memory.set(offset, &[value.byte(0)]);
}

/// MCOPY with MIP-3 linear memory cost.
pub fn mcopy<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    revm_interpreter::check!(context.interpreter, CANCUN);
    revm_interpreter::popn!([dst, src, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);
    revm_interpreter::gas!(context.interpreter, context.host.gas_params().mcopy_cost(len));
    if len == 0 {
        return;
    }
    let dst = revm_interpreter::as_usize_or_fail!(context.interpreter, dst);
    let src = revm_interpreter::as_usize_or_fail!(context.interpreter, src);
    resize_memory_mip3!(context.interpreter, max(dst, src), len);
    context.interpreter.memory.copy(dst, src, len);
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

/// KECCAK256 with MIP-3 linear memory cost.
pub fn keccak256<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn_top!([offset], top, context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, top);
    revm_interpreter::gas!(context.interpreter, context.host.gas_params().keccak256_cost(len));
    let hash = if len == 0 {
        KECCAK_EMPTY
    } else {
        let from = revm_interpreter::as_usize_or_fail!(context.interpreter, offset);
        resize_memory_mip3!(context.interpreter, from, len);
        revm::primitives::keccak256(
            context.interpreter.memory.slice_len(from, len).as_ref() as &[u8]
        )
    };
    *top = hash.into();
}

// ---------------------------------------------------------------------------
// Copy opcodes
// ---------------------------------------------------------------------------

/// CALLDATACOPY with MIP-3 linear memory cost.
pub fn calldatacopy<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn!([memory_offset, data_offset, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = monad_copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    ) else {
        return;
    };
    let data_offset = revm_interpreter::as_usize_saturated!(data_offset);
    match context.interpreter.input.input() {
        CallInput::Bytes(bytes) => {
            context.interpreter.memory.set_data(memory_offset, data_offset, len, bytes.as_ref());
        }
        CallInput::SharedBuffer(range) => {
            context.interpreter.memory.set_data_from_global(
                memory_offset,
                data_offset,
                len,
                range.clone(),
            );
        }
    }
}

/// CODECOPY with MIP-3 linear memory cost.
pub fn codecopy<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn!([memory_offset, code_offset, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = monad_copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    ) else {
        return;
    };
    let code_offset = revm_interpreter::as_usize_saturated!(code_offset);
    context.interpreter.memory.set_data(
        memory_offset,
        code_offset,
        len,
        context.interpreter.bytecode.bytecode_slice(),
    );
}

/// RETURNDATACOPY with MIP-3 linear memory cost.
pub fn returndatacopy<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::check!(context.interpreter, BYZANTIUM);
    revm_interpreter::popn!([memory_offset, offset, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);
    let data_offset = revm_interpreter::as_usize_saturated!(offset);

    let data_end = data_offset.saturating_add(len);
    if data_end > context.interpreter.return_data.buffer().len() {
        context.interpreter.halt(InstructionResult::OutOfOffset);
        return;
    }

    let Some(memory_offset) = monad_copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    ) else {
        return;
    };
    context.interpreter.memory.set_data(
        memory_offset,
        data_offset,
        len,
        context.interpreter.return_data.buffer(),
    );
}

/// EXTCODECOPY with MIP-3 linear memory cost.
pub fn extcodecopy<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn!([address, memory_offset, code_offset, len_u256], context.interpreter);
    let address = address.into_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len_u256);
    revm_interpreter::gas!(context.interpreter, context.host.gas_params().extcodecopy(len));

    let mut memory_offset_usize = 0;
    if len != 0 {
        memory_offset_usize =
            revm_interpreter::as_usize_or_fail!(context.interpreter, memory_offset);
        resize_memory_mip3!(context.interpreter, memory_offset_usize, len);
    }

    let code = if spec_id.is_enabled_in(SpecId::BERLIN) {
        let account = revm_interpreter::berlin_load_account!(context, address, true);
        account.code.as_ref().unwrap().original_bytes()
    } else {
        let Some(code) = context.host.load_account_code(address) else {
            return context.interpreter.halt_fatal();
        };
        code.data
    };

    let code_offset_usize =
        core::cmp::min(revm_interpreter::as_usize_saturated!(code_offset), code.len());
    context.interpreter.memory.set_data(memory_offset_usize, code_offset_usize, len, &code);
}

// ---------------------------------------------------------------------------
// LOG0–LOG4
// ---------------------------------------------------------------------------

/// LOG with MIP-3 linear memory cost.
pub fn log<const N: usize, H: Host + ?Sized>(
    context: InstructionContext<'_, H, impl InterpreterTypes>,
) {
    revm_interpreter::require_non_staticcall!(context.interpreter);
    revm_interpreter::popn!([offset, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);
    revm_interpreter::gas!(
        context.interpreter,
        context.host.gas_params().log_cost(N as u8, len as u64)
    );
    let data = if len == 0 {
        Bytes::new()
    } else {
        let offset = revm_interpreter::as_usize_or_fail!(context.interpreter, offset);
        resize_memory_mip3!(context.interpreter, offset, len);
        Bytes::copy_from_slice(context.interpreter.memory.slice_len(offset, len).as_ref())
    };
    let Some(topics) = context.interpreter.stack.popn::<N>() else {
        context.interpreter.halt_underflow();
        return;
    };

    let log = revm::primitives::Log {
        address: context.interpreter.input.target_address(),
        data: revm::primitives::LogData::new(topics.into_iter().map(B256::from).collect(), data)
            .expect("LogData should have <=4 topics"),
    };
    context.host.log(log);
}

// ---------------------------------------------------------------------------
// CREATE / CREATE2
// ---------------------------------------------------------------------------

/// CREATE/CREATE2 with MIP-3 linear memory cost.
pub fn create<WIRE: InterpreterTypes, const IS_CREATE2: bool, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::require_non_staticcall!(context.interpreter);
    if IS_CREATE2 {
        revm_interpreter::check!(context.interpreter, PETERSBURG);
    }

    revm_interpreter::popn!([value, code_offset, len], context.interpreter);
    let len = revm_interpreter::as_usize_or_fail!(context.interpreter, len);

    let mut code = Bytes::new();
    if len != 0 {
        if context.interpreter.runtime_flag.spec_id().is_enabled_in(SpecId::SHANGHAI) {
            if len > context.host.max_initcode_size() {
                context.interpreter.halt(InstructionResult::CreateInitCodeSizeLimit);
                return;
            }
            revm_interpreter::gas!(
                context.interpreter,
                context.host.gas_params().initcode_cost(len)
            );
        }

        let code_offset = revm_interpreter::as_usize_or_fail!(context.interpreter, code_offset);
        resize_memory_mip3!(context.interpreter, code_offset, len);

        code =
            Bytes::copy_from_slice(context.interpreter.memory.slice_len(code_offset, len).as_ref());
    }

    let scheme = if IS_CREATE2 {
        revm_interpreter::popn!([salt], context.interpreter);
        revm_interpreter::gas!(context.interpreter, context.host.gas_params().create2_cost(len));
        CreateScheme::Create2 { salt }
    } else {
        revm_interpreter::gas!(context.interpreter, context.host.gas_params().create_cost());
        CreateScheme::Create
    };

    let mut gas_limit = context.interpreter.gas.remaining();
    if context.interpreter.runtime_flag.spec_id().is_enabled_in(SpecId::TANGERINE) {
        gas_limit = context.host.gas_params().call_stipend_reduction(gas_limit);
    }
    revm_interpreter::gas!(context.interpreter, gas_limit);

    context.interpreter.bytecode.set_action(InterpreterAction::NewFrame(FrameInput::Create(
        Box::new(CreateInputs::new(
            context.interpreter.input.target_address(),
            scheme,
            value,
            code,
            gas_limit,
        )),
    )));
}

// ---------------------------------------------------------------------------
// CALL / CALLCODE / DELEGATECALL / STATICCALL
// ---------------------------------------------------------------------------

/// CALL with MIP-3 linear memory cost.
pub fn call<WIRE: InterpreterTypes, H: Host + ?Sized>(
    mut context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn!([local_gas_limit, to, value], context.interpreter);
    let to = to.into_address();
    let local_gas_limit = u64::try_from(local_gas_limit).unwrap_or(u64::MAX);
    let has_transfer = !value.is_zero();

    if context.interpreter.runtime_flag.is_static() && has_transfer {
        context.interpreter.halt(InstructionResult::CallNotAllowedInsideStatic);
        return;
    }

    let Some((input, return_memory_offset)) =
        monad_get_memory_input_and_out_ranges(context.interpreter, context.host.gas_params())
    else {
        return;
    };

    let Some((gas_limit, bytecode, bytecode_hash)) =
        load_acc_and_calc_gas(&mut context, to, has_transfer, true, local_gas_limit)
    else {
        return;
    };

    context.interpreter.bytecode.set_action(InterpreterAction::NewFrame(FrameInput::Call(
        Box::new(CallInputs {
            input: CallInput::SharedBuffer(input),
            gas_limit,
            target_address: to,
            caller: context.interpreter.input.target_address(),
            bytecode_address: to,
            known_bytecode: Some((bytecode_hash, bytecode)),
            value: CallValue::Transfer(value),
            scheme: CallScheme::Call,
            is_static: context.interpreter.runtime_flag.is_static(),
            return_memory_offset,
        }),
    )));
}

/// CALLCODE with MIP-3 linear memory cost.
pub fn call_code<WIRE: InterpreterTypes, H: Host + ?Sized>(
    mut context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::popn!([local_gas_limit, to, value], context.interpreter);
    let to = Address::from_word(B256::from(to));
    let local_gas_limit = u64::try_from(local_gas_limit).unwrap_or(u64::MAX);
    let has_transfer = !value.is_zero();

    let Some((input, return_memory_offset)) =
        monad_get_memory_input_and_out_ranges(context.interpreter, context.host.gas_params())
    else {
        return;
    };

    let Some((gas_limit, bytecode, bytecode_hash)) =
        load_acc_and_calc_gas(&mut context, to, has_transfer, false, local_gas_limit)
    else {
        return;
    };

    context.interpreter.bytecode.set_action(InterpreterAction::NewFrame(FrameInput::Call(
        Box::new(CallInputs {
            input: CallInput::SharedBuffer(input),
            gas_limit,
            target_address: context.interpreter.input.target_address(),
            caller: context.interpreter.input.target_address(),
            bytecode_address: to,
            known_bytecode: Some((bytecode_hash, bytecode)),
            value: CallValue::Transfer(value),
            scheme: CallScheme::CallCode,
            is_static: context.interpreter.runtime_flag.is_static(),
            return_memory_offset,
        }),
    )));
}

/// DELEGATECALL with MIP-3 linear memory cost.
pub fn delegate_call<WIRE: InterpreterTypes, H: Host + ?Sized>(
    mut context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::check!(context.interpreter, HOMESTEAD);
    revm_interpreter::popn!([local_gas_limit, to], context.interpreter);
    let to = Address::from_word(B256::from(to));
    let local_gas_limit = u64::try_from(local_gas_limit).unwrap_or(u64::MAX);

    let Some((input, return_memory_offset)) =
        monad_get_memory_input_and_out_ranges(context.interpreter, context.host.gas_params())
    else {
        return;
    };

    let Some((gas_limit, bytecode, bytecode_hash)) =
        load_acc_and_calc_gas(&mut context, to, false, false, local_gas_limit)
    else {
        return;
    };

    context.interpreter.bytecode.set_action(InterpreterAction::NewFrame(FrameInput::Call(
        Box::new(CallInputs {
            input: CallInput::SharedBuffer(input),
            gas_limit,
            target_address: context.interpreter.input.target_address(),
            caller: context.interpreter.input.caller_address(),
            bytecode_address: to,
            known_bytecode: Some((bytecode_hash, bytecode)),
            value: CallValue::Apparent(context.interpreter.input.call_value()),
            scheme: CallScheme::DelegateCall,
            is_static: context.interpreter.runtime_flag.is_static(),
            return_memory_offset,
        }),
    )));
}

/// STATICCALL with MIP-3 linear memory cost.
pub fn static_call<WIRE: InterpreterTypes, H: Host + ?Sized>(
    mut context: InstructionContext<'_, H, WIRE>,
) {
    revm_interpreter::check!(context.interpreter, BYZANTIUM);
    revm_interpreter::popn!([local_gas_limit, to], context.interpreter);
    let to = Address::from_word(B256::from(to));
    let local_gas_limit = u64::try_from(local_gas_limit).unwrap_or(u64::MAX);

    let Some((input, return_memory_offset)) =
        monad_get_memory_input_and_out_ranges(context.interpreter, context.host.gas_params())
    else {
        return;
    };

    let Some((gas_limit, bytecode, bytecode_hash)) =
        load_acc_and_calc_gas(&mut context, to, false, false, local_gas_limit)
    else {
        return;
    };

    context.interpreter.bytecode.set_action(InterpreterAction::NewFrame(FrameInput::Call(
        Box::new(CallInputs {
            input: CallInput::SharedBuffer(input),
            gas_limit,
            target_address: to,
            caller: context.interpreter.input.target_address(),
            bytecode_address: to,
            known_bytecode: Some((bytecode_hash, bytecode)),
            value: CallValue::Transfer(U256::ZERO),
            scheme: CallScheme::StaticCall,
            is_static: true,
            return_memory_offset,
        }),
    )));
}

// ---------------------------------------------------------------------------
// RETURN / REVERT
// ---------------------------------------------------------------------------

/// RETURN with MIP-3 linear memory cost.
pub fn ret<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    monad_return_inner(context.interpreter, InstructionResult::Return);
}

/// REVERT with MIP-3 linear memory cost.
pub fn revert<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    revm_interpreter::check!(context.interpreter, BYZANTIUM);
    monad_return_inner(context.interpreter, InstructionResult::Revert);
}
