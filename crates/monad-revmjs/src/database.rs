use core::convert::Infallible;
use revm::{
    database::CacheDB,
    primitives::{Address, B256, U256},
    state::{AccountInfo, Bytecode},
    DatabaseRef,
};

pub(super) type LocalDB = CacheDB<EmptyLocalDB>;

/// Empty state without synthetic block history; supplied values live in the cache.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EmptyLocalDB;

impl DatabaseRef for EmptyLocalDB {
    type Error = Infallible;

    fn basic_ref(&self, _address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        Ok(None)
    }

    fn code_by_hash_ref(&self, _code_hash: B256) -> Result<Bytecode, Self::Error> {
        Ok(Bytecode::default())
    }

    fn storage_ref(&self, _address: Address, _index: U256) -> Result<U256, Self::Error> {
        Ok(U256::ZERO)
    }

    fn block_hash_ref(&self, _number: u64) -> Result<B256, Self::Error> {
        Ok(B256::ZERO)
    }
}
