// MonadEvm - wrapper around base Evm with Monad-specific types.
use crate::{
    instructions::{monad_instructions, MonadInstructionProvider, MonadInstructions},
    journal::MonadJournalTr,
    precompiles::MonadPrecompiles,
    MonadHardfork,
};
use alloc::vec::Vec;
use revm::{
    context::{Cfg, ContextError, ContextSetters, Evm, FrameStack},
    context_interface::{ContextTr, JournalTr},
    handler::{evm::FrameTr, EthFrame, EvmTr, FrameInitOrResult, ItemOrResult, PrecompileProvider},
    inspector::{InspectorEvmTr, JournalExt},
    interpreter::{interpreter::EthInterpreter, InterpreterResult},
    Database, Inspector,
};

/// Exact Monad hardforks selected for materialized execution frames.
#[derive(Debug, Clone)]
struct FrameSpecStack {
    specs: Vec<MonadHardfork>,
}

impl FrameSpecStack {
    fn new() -> Self {
        Self { specs: Vec::with_capacity(8) }
    }

    fn clear(&mut self) {
        self.specs.clear();
    }

    fn push(&mut self, spec: MonadHardfork) {
        self.specs.push(spec);
    }

    fn truncate(&mut self, len: usize) {
        self.specs.truncate(len);
    }

    fn current(&self) -> Option<MonadHardfork> {
        self.specs.last().copied()
    }

    const fn len(&self) -> usize {
        self.specs.len()
    }
}

/// Monad EVM with custom gas costs and precompiles.
#[derive(Debug, Clone)]
pub struct MonadEvm<
    CTX,
    INSP,
    I = MonadInstructions<CTX>,
    P = MonadPrecompiles,
    F = EthFrame<EthInterpreter>,
>(
    /// Inner REVM instance.
    pub Evm<CTX, INSP, I, P, F>,
    FrameSpecStack,
);

impl<CTX, INSP> MonadEvm<CTX, INSP, MonadInstructions<CTX>, MonadPrecompiles>
where
    CTX: ContextTr<Cfg: Cfg<Spec = MonadHardfork>>,
{
    /// Create a new Monad EVM with custom gas costs and precompiles.
    pub fn new(ctx: CTX, inspector: INSP) -> Self {
        let spec = ctx.cfg().spec();
        Self::from_inner(Evm {
            ctx,
            inspector,
            instruction: monad_instructions(spec),
            precompiles: MonadPrecompiles::new_with_spec(spec),
            frame_stack: FrameStack::new_prealloc(8),
        })
    }
}

impl<CTX, INSP, I, P, F> MonadEvm<CTX, INSP, I, P, F> {
    /// Wraps a REVM instance whose frame stack is empty.
    ///
    /// Use this constructor instead of directly constructing the tuple so exact Monad frame
    /// metadata stays synchronized with REVM's frame stack.
    pub fn from_inner(inner: Evm<CTX, INSP, I, P, F>) -> Self {
        assert!(inner.frame_stack.index().is_none(), "cannot wrap an EVM with active frames");
        Self(inner, FrameSpecStack::new())
    }

    /// Consumes this wrapper and returns the inner REVM instance.
    ///
    /// The frame stack must be empty because the inner REVM does not retain exact Monad frame
    /// metadata on its own.
    pub fn into_inner(self) -> Evm<CTX, INSP, I, P, F> {
        assert!(self.0.frame_stack.index().is_none(), "cannot unwrap an EVM with active frames");
        self.0
    }

    /// Consume self and return a new EVM with given Inspector.
    pub fn with_inspector<OINSP>(self, inspector: OINSP) -> MonadEvm<CTX, OINSP, I, P, F> {
        MonadEvm(self.0.with_inspector(inspector), self.1)
    }

    /// Consume self and return a new EVM with given Precompiles.
    pub fn with_precompiles<OP>(self, precompiles: OP) -> MonadEvm<CTX, INSP, I, OP, F> {
        MonadEvm(self.0.with_precompiles(precompiles), self.1)
    }

    /// Consume self and return the inner Inspector.
    pub fn into_inspector(self) -> INSP {
        self.0.into_inspector()
    }
}

impl<CTX, INSP, I, P> InspectorEvmTr for MonadEvm<CTX, INSP, I, P>
where
    CTX: ContextTr<Cfg: Cfg<Spec = MonadHardfork>, Journal: JournalExt + MonadJournalTr>
        + ContextSetters,
    I: MonadInstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
    INSP: Inspector<CTX, I::InterpreterTypes>,
{
    type Inspector = INSP;

    #[inline]
    fn all_inspector(
        &self,
    ) -> (
        &Self::Context,
        &Self::Instructions,
        &Self::Precompiles,
        &FrameStack<Self::Frame>,
        &Self::Inspector,
    ) {
        self.0.all_inspector()
    }

    #[inline]
    fn all_mut_inspector(
        &mut self,
    ) -> (
        &mut Self::Context,
        &mut Self::Instructions,
        &mut Self::Precompiles,
        &mut FrameStack<Self::Frame>,
        &mut Self::Inspector,
    ) {
        self.0.all_mut_inspector()
    }
}

impl<CTX, INSP, I, P> MonadEvm<CTX, INSP, I, P, EthFrame<EthInterpreter>>
where
    CTX: ContextTr<Cfg: Cfg<Spec = MonadHardfork>, Journal: MonadJournalTr>,
    I: MonadInstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    /// Applies all frame-scoped Monad behavior for an exact hardfork.
    fn apply_frame_spec(&mut self, spec: MonadHardfork) {
        self.0.instruction.set_spec(spec);
        let precompiles_changed = self.0.precompiles.set_spec(spec);
        self.0.ctx.journal_mut().reconfigure_reserve_balance(spec);
        let precompiles_empty = self.0.ctx.journal().precompile_addresses().is_empty();
        if precompiles_changed || precompiles_empty {
            self.0.ctx.journal_mut().warm_precompiles(self.0.precompiles.warm_addresses());
        }
    }

    /// Restores the current materialized frame's hardfork.
    fn restore_current_frame_spec(&mut self) {
        if let Some(spec) = self.1.current() {
            self.apply_frame_spec(spec);
        }
    }
}

impl<CTX, INSP, I, P> EvmTr for MonadEvm<CTX, INSP, I, P, EthFrame<EthInterpreter>>
where
    CTX: ContextTr<Cfg: Cfg<Spec = MonadHardfork>, Journal: MonadJournalTr>,
    I: MonadInstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type Context = CTX;
    type Instructions = I;
    type Precompiles = P;
    type Frame = EthFrame<EthInterpreter>;

    #[inline]
    fn all(
        &self,
    ) -> (&Self::Context, &Self::Instructions, &Self::Precompiles, &FrameStack<Self::Frame>) {
        self.0.all()
    }

    #[inline]
    fn all_mut(
        &mut self,
    ) -> (
        &mut Self::Context,
        &mut Self::Instructions,
        &mut Self::Precompiles,
        &mut FrameStack<Self::Frame>,
    ) {
        self.0.all_mut()
    }

    fn frame_init(
        &mut self,
        mut frame_input: <Self::Frame as FrameTr>::FrameInit,
    ) -> Result<
        ItemOrResult<&mut Self::Frame, <Self::Frame as FrameTr>::FrameResult>,
        ContextError<<<Self::Context as ContextTr>::Db as Database>::Error>,
    > {
        if self.0.frame_stack.index().is_none() {
            self.1.clear();
        }
        let spec = self.0.ctx.cfg().spec();
        self.apply_frame_spec(spec);
        frame_input.memory.set_memory_limit(self.0.ctx.cfg().memory_limit());

        let expected_depth = self.0.frame_stack.index().map_or(1, |index| index + 2);
        match self.0.frame_init(frame_input) {
            Ok(ItemOrResult::Item(_)) => {
                self.1.push(spec);
                debug_assert_eq!(self.1.len(), expected_depth);
                Ok(ItemOrResult::Item(self.0.frame_stack.get()))
            }
            Ok(ItemOrResult::Result(result)) => Ok(ItemOrResult::Result(result)),
            Err(error) => {
                self.restore_current_frame_spec();
                Err(error)
            }
        }
    }

    fn frame_run(
        &mut self,
    ) -> Result<
        FrameInitOrResult<Self::Frame>,
        ContextError<<<Self::Context as ContextTr>::Db as Database>::Error>,
    > {
        self.0.frame_run()
    }

    fn frame_return_result(
        &mut self,
        result: <Self::Frame as FrameTr>::FrameResult,
    ) -> Result<
        Option<<Self::Frame as FrameTr>::FrameResult>,
        ContextError<<<Self::Context as ContextTr>::Db as Database>::Error>,
    > {
        let previous_depth = self.0.frame_stack.index();
        let result = self.0.frame_return_result(result);
        let current_depth = self.0.frame_stack.index();
        let current_len = current_depth.map_or(0, |index| index + 1);
        debug_assert!(current_depth == previous_depth || current_len < self.1.len());
        self.1.truncate(current_len);
        if current_depth.is_some() {
            self.restore_current_frame_spec();
        }
        debug_assert_eq!(self.1.len(), current_len);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        api::{
            builder::MonadBuilder,
            default_ctx::{DefaultMonad, MonadContext},
        },
        reserve_balance::abi::RESERVE_BALANCE_ADDRESS,
        staking::storage::STAKING_ADDRESS,
    };
    use revm::{
        context_interface::{ContextTr, JournalTr},
        database::EmptyDB,
        handler::system_call::SystemCallEvm,
        precompile::u64_to_address,
    };

    #[test]
    fn test_fresh_system_call_warms_precompiles() {
        let mut evm = MonadContext::<EmptyDB>::monad().build_monad();
        assert!(evm.0.ctx.journal().precompile_addresses().is_empty());

        evm.system_call_one(STAKING_ADDRESS, Default::default())
            .expect("fresh system call should execute");

        let precompiles = evm.0.ctx.journal().precompile_addresses();
        assert!(precompiles.contains(&u64_to_address(1)));
        assert!(precompiles.contains(&STAKING_ADDRESS));
        assert!(precompiles.contains(&RESERVE_BALANCE_ADDRESS));
    }
}
