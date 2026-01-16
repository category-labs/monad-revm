//! Contains the `[MonadSpecId]` type and its implementation.
use core::str::FromStr;
use revm::primitives::hardfork::{SpecId, UnknownHardfork};

/// Monad spec id.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(non_camel_case_types)]
pub enum MonadSpecId {
    /// Monad launch spec id.
    #[default]
    MonadEight = 100,
}

impl MonadSpecId {
    /// Returns the underlying Ethereum [`SpecId`] this Monad hardfork is built upon.
    ///
    /// Used internally to:
    /// - Get the base instruction table (before Monad gas overrides)
    /// - Get the base precompiles (before Monad gas overrides)
    /// - Check Ethereum feature availability (e.g., blob support)
    ///
    /// Note: This returns the *foundation* spec, not an equivalence.
    /// Future Monad hardforks may add features beyond the base Ethereum spec.
    pub const fn into_eth_spec(self) -> SpecId {
        match self {
            Self::MonadEight => SpecId::PRAGUE,
        }
    }

    /// Checks if the [`MonadSpecId`] is enabled in the other [`MonadSpecId`].
    pub const fn is_enabled_in(self, other: MonadSpecId) -> bool {
        other as u8 <= self as u8
    }
}

impl From<MonadSpecId> for SpecId {
    fn from(spec: MonadSpecId) -> Self {
        spec.into_eth_spec()
    }
}

impl FromStr for MonadSpecId {
    type Err = UnknownHardfork;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            name::MONAD_EIGHT => Ok(MonadSpecId::MonadEight),
            _ => Err(UnknownHardfork),
        }
    }
}

impl From<MonadSpecId> for &'static str {
    fn from(spec_id: MonadSpecId) -> Self {
        match spec_id {
            MonadSpecId::MonadEight => name::MONAD_EIGHT,
        }
    }
}

/// String identifiers for Monad hardforks
pub mod name {
    /// Mainnet launch spec name.
    pub const MONAD_EIGHT: &str = "MonadEight";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_spec_id_default() {
        assert_eq!(MonadSpecId::default(), MonadSpecId::MonadEight);
    }

    #[test]
    fn test_monad_spec_into_eth_spec() {
        assert_eq!(MonadSpecId::MonadEight.into_eth_spec(), SpecId::PRAGUE);
    }

    #[test]
    fn test_monad_spec_from_str() {
        assert_eq!(
            "MonadEight".parse::<MonadSpecId>().unwrap(),
            MonadSpecId::MonadEight
        );
    }

    #[test]
    fn test_monad_spec_from_str_unknown() {
        assert!("Unknown".parse::<MonadSpecId>().is_err());
    }

    #[test]
    fn test_monad_spec_into_str() {
        let name: &'static str = MonadSpecId::MonadEight.into();
        assert_eq!(name, "MonadEight");
    }

    #[test]
    fn test_monad_spec_is_enabled_in() {
        // MonadEight is enabled in MonadEight
        assert!(MonadSpecId::MonadEight.is_enabled_in(MonadSpecId::MonadEight));
    }

    #[test]
    fn test_monad_spec_from_impl() {
        let spec_id: SpecId = MonadSpecId::MonadEight.into();
        assert_eq!(spec_id, SpecId::PRAGUE);
    }
}
