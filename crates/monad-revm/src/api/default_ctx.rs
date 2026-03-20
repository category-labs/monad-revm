// Default Monad context type and factory.

use crate::{MonadCfgEnv, MonadChainContext, MonadJournal, MonadJournalTr, MonadSpecId};
use revm::{
    context::{BlockEnv, LocalContext, TxEnv},
    context_interface::JournalTr,
    database_interface::{Database, EmptyDB},
    Context,
};

/// Type alias for the default Monad context.
///
/// Uses standard Ethereum types since Monad doesn't need custom tx/block types.
/// The key difference is:
/// - Using `MonadSpecId` instead of `SpecId`
/// - Using `MonadCfgEnv` which has Monad-specific defaults (128KB code size limit)
pub type MonadContext<DB> =
    Context<BlockEnv, TxEnv, MonadCfgEnv, DB, MonadJournal<DB>, MonadChainContext>;

/// Trait for creating a default Monad context.
pub trait DefaultMonad {
    /// Creates a new Monad context with default settings and an empty database.
    fn monad() -> MonadContext<EmptyDB>;
}

/// Creates a Monad context with the given database backend.
pub fn monad_context_with_db<DB: Database>(db: DB) -> MonadContext<DB> {
    let mut journaled_state = MonadJournal::new(db);
    journaled_state.set_monad_spec(MonadSpecId::default());
    journaled_state.set_spec_id(MonadSpecId::default().into());
    Context {
        block: BlockEnv::default(),
        tx: TxEnv::default(),
        cfg: MonadCfgEnv::new(),
        journaled_state,
        chain: MonadChainContext::default(),
        local: LocalContext::default(),
        error: Ok(()),
    }
}

impl DefaultMonad for MonadContext<EmptyDB> {
    fn monad() -> Self {
        monad_context_with_db(EmptyDB::new())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::api::builder::MonadBuilder;
    use revm::{inspector::NoOpInspector, ExecuteEvm};

    #[test]
    fn default_run_monad() {
        let ctx = Context::monad();
        let mut evm = ctx.build_monad_with_inspector(NoOpInspector {});
        let tx = TxEnv::default();
        let _ = evm.transact(tx);
    }
}
