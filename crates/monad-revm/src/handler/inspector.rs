//! Inspection with reserve adjudication before the root frame's end callbacks.

use super::MonadHandler;
use crate::api::exec::MonadContextTr;
use revm::{
    context_interface::{ContextTr, JournalTr},
    handler::{
        handler::EvmTrError, instructions::InstructionProvider, EthFrame, FrameResult, ItemOrResult,
    },
    inspector::{
        handler::{frame_end, frame_start},
        inspect_instructions, Inspector, InspectorEvmTr, InspectorHandler,
    },
    interpreter::{
        interpreter::EthInterpreter, interpreter_action::FrameInit, CallOutcome, FrameInput,
        InstructionResult,
    },
};

impl<EVM, ERROR> InspectorHandler for MonadHandler<EVM, ERROR, EthFrame<EthInterpreter>>
where
    EVM: InspectorEvmTr<Context: MonadContextTr, Frame = EthFrame<EthInterpreter>>,
    ERROR: EvmTrError<EVM>,
{
    type IT = EthInterpreter;

    fn inspect_run_exec_loop(
        &mut self,
        evm: &mut Self::Evm,
        mut first_frame_input: FrameInit,
    ) -> Result<FrameResult, Self::Error> {
        let checkpoint = self.begin_execution(evm);
        let (ctx, inspector) = evm.ctx_inspector();
        if let Some(mut result) = frame_start(ctx, inspector, &mut first_frame_input.frame_input) {
            self.adjudicate_execution(evm, &mut result, checkpoint, true)?;
            self.inspect_root_end(evm, &first_frame_input.frame_input, &mut result);
            return Ok(result);
        }

        let input = first_frame_input.frame_input.clone();
        let logs_i = ctx.journal().logs().len();
        // Root initialization is separate so immediate results are adjudicated before callbacks.
        if let ItemOrResult::Result(mut result) = evm.frame_init(first_frame_input)? {
            let (ctx, inspector) = evm.ctx_inspector();
            let logs = ctx.journal().logs()[logs_i..].to_vec();
            for log in logs {
                inspector.log(ctx, log);
            }
            if let FrameResult::Call(CallOutcome {
                was_precompile_called: true,
                precompile_call_logs,
                ..
            }) = &result
            {
                for log in precompile_call_logs.clone() {
                    inspector.log(ctx, log);
                }
            }
            self.adjudicate_execution(evm, &mut result, checkpoint, false)?;
            self.inspect_root_end(evm, &input, &mut result);
            return Ok(result);
        }

        let (ctx, inspector, frame) = evm.ctx_inspector_frame();
        let logs = ctx.journal().logs()[logs_i..].to_vec();
        for log in logs {
            inspector.log(ctx, log);
        }
        inspector.initialize_interp(&mut frame.interpreter, ctx);

        loop {
            let root = evm.frame_stack().index() == Some(0);
            let action = if root {
                let (ctx, inspector, frame, instructions) = evm.ctx_inspector_frame_instructions();
                let action = inspect_instructions(
                    ctx,
                    &mut frame.interpreter,
                    inspector,
                    instructions.instruction_table(),
                    instructions.gas_table(),
                );
                frame.process_next_action::<_, ERROR>(ctx, action)?
            } else {
                evm.inspect_frame_run()?
            };

            let result = match action {
                ItemOrResult::Item(init) => match evm.inspect_frame_init(init)? {
                    ItemOrResult::Item(_) => continue,
                    ItemOrResult::Result(result) => result,
                },
                ItemOrResult::Result(mut result) => {
                    if root {
                        self.adjudicate_execution(evm, &mut result, checkpoint, false)?;
                        let input = evm.frame_stack().get().input.clone();
                        self.inspect_root_end(evm, &input, &mut result);
                        evm.frame_stack().get().set_finished(true);
                    }
                    result
                }
            };
            if let Some(result) = evm.frame_return_result(result)? {
                return Ok(result);
            }
        }
    }
}

impl<EVM, ERROR> MonadHandler<EVM, ERROR, EthFrame<EthInterpreter>>
where
    EVM: InspectorEvmTr<Context: MonadContextTr, Frame = EthFrame<EthInterpreter>>,
    ERROR: EvmTrError<EVM>,
{
    fn inspect_root_end(&mut self, evm: &mut EVM, input: &FrameInput, result: &mut FrameResult) {
        let adjudicated_output = self.reserve_balance_violation.then(|| result.output());
        let (ctx, inspector) = evm.ctx_inspector();
        frame_end(ctx, inspector, input, result);
        // Inspectors may override an adjudicated outcome, as they can other frame outcomes.
        self.reserve_balance_violation &= result.instruction_result() == InstructionResult::Revert
            && adjudicated_output.as_ref() == Some(&result.output());
    }
}
