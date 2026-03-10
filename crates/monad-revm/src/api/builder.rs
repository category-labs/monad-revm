use crate::{
    evm::MonadEvm, instructions::MonadInstructions, precompiles::MonadPrecompiles,
    MonadChainContext, MonadJournal, MonadSpecId,
};
use revm::{
    context::Cfg,
    context_interface::{Block, LocalContextTr, Transaction},
    Context, Database,
};

/// Type alias for default MonadEvm.
pub type DefaultMonadEvm<CTX, INSP = ()> =
    MonadEvm<CTX, INSP, MonadInstructions<CTX>, MonadPrecompiles>;

/// Trait for building Monad EVM instances from a context.
pub trait MonadBuilder: Sized {
    /// The context type used by this builder.
    type Context;

    /// Build MonadEvm without inspector.
    fn build_monad(self) -> DefaultMonadEvm<Self::Context>;

    /// Build MonadEvm with inspector.
    fn build_monad_with_inspector<INSP>(
        self,
        inspector: INSP,
    ) -> DefaultMonadEvm<Self::Context, INSP>;
}

impl<BLOCK, TX, CFG, DB, LOCAL> MonadBuilder
    for Context<BLOCK, TX, CFG, DB, MonadJournal<DB>, MonadChainContext, LOCAL>
where
    BLOCK: Block,
    TX: Transaction,
    CFG: Cfg<Spec = MonadSpecId>,
    DB: Database,
    LOCAL: LocalContextTr,
{
    type Context = Self;

    fn build_monad(self) -> DefaultMonadEvm<Self::Context> {
        MonadEvm::new(self, ())
    }

    fn build_monad_with_inspector<INSP>(
        self,
        inspector: INSP,
    ) -> DefaultMonadEvm<Self::Context, INSP> {
        MonadEvm::new(self, inspector)
    }
}
