// Default Monad context type and factory.

use crate::{MonadCfgEnv, MonadChainContext, MonadHardfork, MonadJournal};
use revm::{
    context::{BlockEnv, LocalContext, TxEnv},
    context_interface::JournalTr,
    database_interface::{Database, EmptyDB},
    Context,
};

/// Type alias for the default Monad context.
///
/// Uses [`MonadCfgEnv`] for the Monad hardfork and gas rules, [`MonadJournal`] for
/// reserve-balance tracking, and [`MonadChainContext`] for chain-dependent reserve decisions.
pub type MonadContext<DB> =
    Context<BlockEnv, TxEnv, MonadCfgEnv, DB, MonadJournal<DB>, MonadChainContext>;

/// Trait for creating a default Monad context.
pub trait DefaultMonad {
    /// Creates a MonadNine context with default settings and an empty database.
    ///
    /// This does not resolve the active hardfork from a chain ID and timestamp. Historical
    /// execution should select a spec explicitly and populate [`MonadChainContext`].
    fn monad() -> MonadContext<EmptyDB>;
}

/// Creates a MonadNine context with the given database backend.
///
/// The context applies Monad gas parameters, a 30 million transaction gas cap, and a default
/// 10 MON reserve threshold. Its chain metadata is empty; canonical replay must populate it.
pub fn monad_context_with_db<DB: Database>(db: DB) -> MonadContext<DB> {
    let mut journaled_state = MonadJournal::new(db);
    journaled_state.set_spec_id(MonadHardfork::default().into());
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
