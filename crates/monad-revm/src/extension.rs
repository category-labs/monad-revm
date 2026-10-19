//! MIP-7 extension opcode decoding.

use revm::interpreter::{
    interpreter_types::{InterpreterTypes, Jumps, LegacyBytecode},
    Host, InstructionContext, InstructionExecResult, InstructionResult,
};

/// MIP-7 extension opcode prefix.
pub const EXTENSION: u8 = 0xAE;

/// Decodes an extension opcode and rejects unassigned selectors.
pub fn extension<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) -> InstructionExecResult {
    // The interpreter has consumed the prefix. Read only original code, since the
    // zero padding after a trailing prefix must not supply a selector.
    let _selector = context
        .interpreter
        .bytecode
        .bytecode_slice()
        .get(context.interpreter.bytecode.pc())
        .copied()
        .ok_or(InstructionResult::InvalidFEOpcode)?;

    // MIP-7 assigns no selectors. Later extensions define their semantics and gas costs.
    Err(InstructionResult::InvalidFEOpcode)
}
