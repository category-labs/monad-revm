//! MIP-7 extension decoding and MIP-18 call stack introspection.

use crate::{api::exec::MonadContextTr, MonadJournalTr};
use revm::{
    interpreter::{
        interpreter_types::{InputsTr, InterpreterTypes, Jumps, LegacyBytecode, StackTr},
        InstructionContext, InstructionExecResult, InstructionResult,
    },
    primitives::U256,
};

/// MIP-7 extension opcode prefix.
pub const EXTENSION: u8 = 0xAE;
/// MIP-18 selector for the current frame's zero-based depth.
pub const CALLSTACKDEPTH_SELECTOR: u8 = 0x00;
/// MIP-18 selector for an ancestor frame's caller.
pub const CALLERN_SELECTOR: u8 = 0x01;

/// Executes an extension opcode, including its selector-specific gas charge.
pub fn extension<WIRE: InterpreterTypes, H: MonadContextTr + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) -> InstructionExecResult {
    // The interpreter has consumed the prefix. Read only original code, since the
    // zero padding after a trailing prefix must not supply a selector.
    let selector = context
        .interpreter
        .bytecode
        .bytecode_slice()
        .get(context.interpreter.bytecode.pc())
        .copied()
        .ok_or(InstructionResult::InvalidFEOpcode)?;

    match selector {
        CALLSTACKDEPTH_SELECTOR => {
            context.interpreter.bytecode.relative_jump(1);
            revm_interpreter::gas!(context.interpreter, 2);
            revm_interpreter::push!(
                context.interpreter,
                U256::from(context.interpreter.input.depth())
            );
        }
        CALLERN_SELECTOR => {
            context.interpreter.bytecode.relative_jump(1);
            revm_interpreter::gas!(context.interpreter, 2);
            revm_interpreter::popn_top!([], n, context.interpreter);
            let depth = context.interpreter.input.depth();
            *n = if *n > U256::from(depth) {
                U256::ZERO
            } else {
                context.host.journal().call_frame_sender(depth - n.to::<usize>()).into_word().into()
            };
        }
        // MIP-7 leaves every other selector invalid, including JUMPDEST and PUSH1..PUSH32.
        _ => return Err(InstructionResult::InvalidFEOpcode),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instructions::monad_instructions, monad_context_with_db, MonadHardfork};
    use alloc::vec;
    use revm::{
        bytecode::opcode,
        context_interface::ContextTr,
        database::EmptyDB,
        handler::instructions::InstructionProvider,
        interpreter::{
            interpreter::{EthInterpreter, ExtBytecode, InputsImpl},
            Interpreter, SharedMemory,
        },
        primitives::{hardfork::SpecId, Address},
        state::Bytecode,
    };

    #[test]
    fn maximum_call_depth_and_full_width_indices() {
        let origin = Address::repeat_byte(0x11);
        let caller = Address::repeat_byte(0x22);
        let mut ctx = monad_context_with_db(EmptyDB::new());
        ctx.journal_mut().set_call_frame_sender(0, origin);
        ctx.journal_mut().set_call_frame_sender(1024, caller);
        let mut code = vec![
            EXTENSION,
            CALLSTACKDEPTH_SELECTOR,
            opcode::PUSH0,
            EXTENSION,
            CALLERN_SELECTOR,
            opcode::PUSH2,
            4,
            0,
            EXTENSION,
            CALLERN_SELECTOR,
            opcode::PUSH2,
            4,
            1,
            EXTENSION,
            CALLERN_SELECTOR,
        ];
        for n in [U256::from(1) << 32, U256::from(1) << 64, U256::MAX] {
            code.push(opcode::PUSH32);
            code.extend_from_slice(&n.to_be_bytes::<32>());
            code.extend_from_slice(&[EXTENSION, CALLERN_SELECTOR]);
        }
        let mut interpreter = Interpreter::<EthInterpreter>::new(
            SharedMemory::new(),
            ExtBytecode::new(Bytecode::new_raw(code.into())),
            InputsImpl { depth: 1024, caller_address: caller, ..Default::default() },
            false,
            SpecId::OSAKA,
            100_000,
        );
        let instructions = monad_instructions(MonadHardfork::MonadNext);
        interpreter.run_plain(instructions.instruction_table(), instructions.gas_table(), &mut ctx);
        assert_eq!(
            interpreter.stack.data(),
            &vec![
                U256::from(1024),
                caller.into_word().into(),
                origin.into_word().into(),
                U256::ZERO,
                U256::ZERO,
                U256::ZERO,
                U256::ZERO,
            ],
        );
    }
}
