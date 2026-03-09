//! Contains the `[MonadSpecId]` type and its implementation.
use core::str::FromStr;
use revm::primitives::hardfork::{SpecId, UnknownHardfork};

/// Monad spec id.
///
/// Variants are ordered by activation order. Lower discriminant = earlier hardfork.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(non_camel_case_types)]
pub enum MonadSpecId {
    /// Monad launch spec id (based on Prague).
    #[default]
    MonadEight = 100,
    /// MIP-3: Linear memory cost model with 8 MB pooled limit (based on Osaka).
    MonadNine = 101,
    /// Next development spec (based on Osaka).
    MonadNext = 102,
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
            Self::MonadNine | Self::MonadNext => SpecId::OSAKA,
        }
    }

    /// Returns `true` if `self` is enabled when the active spec is `other`.
    ///
    /// A hardfork is enabled when its discriminant is ≤ the active spec's discriminant,
    /// i.e. it was activated at the same time or earlier.
    pub const fn is_enabled_in(self, other: Self) -> bool {
        self as u8 <= other as u8
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
            name::MONAD_EIGHT => Ok(Self::MonadEight),
            name::MONAD_NINE => Ok(Self::MonadNine),
            name::MONAD_NEXT => Ok(Self::MonadNext),
            _ => Err(UnknownHardfork),
        }
    }
}

impl From<MonadSpecId> for &'static str {
    fn from(spec_id: MonadSpecId) -> Self {
        match spec_id {
            MonadSpecId::MonadEight => name::MONAD_EIGHT,
            MonadSpecId::MonadNine => name::MONAD_NINE,
            MonadSpecId::MonadNext => name::MONAD_NEXT,
        }
    }
}

/// String identifiers for Monad hardforks
pub mod name {
    /// Mainnet launch spec name.
    pub const MONAD_EIGHT: &str = "MonadEight";
    /// MIP-3 spec name.
    pub const MONAD_NINE: &str = "MonadNine";
    /// Development spec name.
    pub const MONAD_NEXT: &str = "MonadNext";
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
        assert_eq!(MonadSpecId::MonadNine.into_eth_spec(), SpecId::OSAKA);
        assert_eq!(MonadSpecId::MonadNext.into_eth_spec(), SpecId::OSAKA);
    }

    #[test]
    fn test_monad_spec_from_str() {
        assert_eq!("MonadEight".parse::<MonadSpecId>().unwrap(), MonadSpecId::MonadEight);
        assert_eq!("MonadNine".parse::<MonadSpecId>().unwrap(), MonadSpecId::MonadNine);
        assert_eq!("MonadNext".parse::<MonadSpecId>().unwrap(), MonadSpecId::MonadNext);
    }

    #[test]
    fn test_monad_spec_from_str_unknown() {
        assert!("Unknown".parse::<MonadSpecId>().is_err());
    }

    #[test]
    fn test_monad_spec_into_str() {
        let name: &'static str = MonadSpecId::MonadEight.into();
        assert_eq!(name, "MonadEight");
        let name: &'static str = MonadSpecId::MonadNine.into();
        assert_eq!(name, "MonadNine");
    }

    #[test]
    fn test_monad_spec_is_enabled_in() {
        // MonadEight is enabled in every spec
        assert!(MonadSpecId::MonadEight.is_enabled_in(MonadSpecId::MonadEight));
        assert!(MonadSpecId::MonadEight.is_enabled_in(MonadSpecId::MonadNine));
        assert!(MonadSpecId::MonadEight.is_enabled_in(MonadSpecId::MonadNext));

        // MonadNine is NOT enabled in MonadEight
        assert!(!MonadSpecId::MonadNine.is_enabled_in(MonadSpecId::MonadEight));
        // MonadNine IS enabled in MonadNine and MonadNext
        assert!(MonadSpecId::MonadNine.is_enabled_in(MonadSpecId::MonadNine));
        assert!(MonadSpecId::MonadNine.is_enabled_in(MonadSpecId::MonadNext));

        // MonadNext is only enabled in MonadNext
        assert!(!MonadSpecId::MonadNext.is_enabled_in(MonadSpecId::MonadEight));
        assert!(!MonadSpecId::MonadNext.is_enabled_in(MonadSpecId::MonadNine));
        assert!(MonadSpecId::MonadNext.is_enabled_in(MonadSpecId::MonadNext));
    }

    #[test]
    fn test_monad_spec_ordering() {
        assert!(MonadSpecId::MonadEight < MonadSpecId::MonadNine);
        assert!(MonadSpecId::MonadNine < MonadSpecId::MonadNext);
    }

    #[test]
    fn test_monad_spec_from_impl() {
        let spec_id: SpecId = MonadSpecId::MonadEight.into();
        assert_eq!(spec_id, SpecId::PRAGUE);
        let spec_id: SpecId = MonadSpecId::MonadNine.into();
        assert_eq!(spec_id, SpecId::OSAKA);
    }
}
