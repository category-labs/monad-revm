//! Contains the [`MonadHardfork`] type and its implementation.
use core::{fmt, str::FromStr};
use revm::primitives::hardfork::{SpecId, UnknownHardfork};

/// Monad hardfork identifier.
///
/// Variants are ordered by activation order. Lower discriminant = earlier hardfork.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MonadHardfork {
    /// Monad launch hardfork (based on Prague).
    MonadEight = 100,
    /// MIP-3, MIP-4, MIP-5
    #[default]
    MonadNine = 101,
    /// Next development spec
    MonadNext = 102,
}

impl MonadHardfork {
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

impl From<MonadHardfork> for SpecId {
    fn from(spec: MonadHardfork) -> Self {
        spec.into_eth_spec()
    }
}

impl FromStr for MonadHardfork {
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

impl From<MonadHardfork> for &'static str {
    fn from(spec_id: MonadHardfork) -> Self {
        match spec_id {
            MonadHardfork::MonadEight => name::MONAD_EIGHT,
            MonadHardfork::MonadNine => name::MONAD_NINE,
            MonadHardfork::MonadNext => name::MONAD_NEXT,
        }
    }
}

impl fmt::Display for MonadHardfork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = (*self).into();
        f.write_str(name)
    }
}

/// String identifiers for Monad hardforks
pub mod name {
    /// Mainnet launch spec name.
    pub const MONAD_EIGHT: &str = "MonadEight";
    /// MIP-3, MIP-4 and MIP-5 spec name.
    pub const MONAD_NINE: &str = "MonadNine";
    /// Development spec name.
    pub const MONAD_NEXT: &str = "MonadNext";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_hardfork_default() {
        assert_eq!(MonadHardfork::default(), MonadHardfork::MonadNine);
    }

    #[test]
    fn test_monad_hardfork_type() {
        let spec: MonadHardfork = MonadHardfork::MonadNine;
        assert_eq!(spec, MonadHardfork::MonadNine);
        assert_eq!(spec.into_eth_spec(), SpecId::OSAKA);
    }

    #[test]
    fn test_monad_hardfork_into_eth_spec() {
        assert_eq!(MonadHardfork::MonadEight.into_eth_spec(), SpecId::PRAGUE);
        assert_eq!(MonadHardfork::MonadNine.into_eth_spec(), SpecId::OSAKA);
        assert_eq!(MonadHardfork::MonadNext.into_eth_spec(), SpecId::OSAKA);
    }

    #[test]
    fn test_monad_hardfork_from_str() {
        assert_eq!("MonadEight".parse::<MonadHardfork>().unwrap(), MonadHardfork::MonadEight);
        assert_eq!("MonadNine".parse::<MonadHardfork>().unwrap(), MonadHardfork::MonadNine);
        assert_eq!("MonadNext".parse::<MonadHardfork>().unwrap(), MonadHardfork::MonadNext);
    }

    #[test]
    fn test_monad_hardfork_from_str_unknown() {
        assert!("Unknown".parse::<MonadHardfork>().is_err());
    }

    #[test]
    fn test_monad_hardfork_into_str() {
        let name: &'static str = MonadHardfork::MonadEight.into();
        assert_eq!(name, "MonadEight");
        let name: &'static str = MonadHardfork::MonadNine.into();
        assert_eq!(name, "MonadNine");
    }

    #[test]
    fn test_monad_hardfork_display() {
        assert_eq!(MonadHardfork::MonadEight.to_string(), "MonadEight");
        assert_eq!(MonadHardfork::MonadNine.to_string(), "MonadNine");
        assert_eq!(MonadHardfork::MonadNext.to_string(), "MonadNext");
    }

    #[test]
    fn test_monad_hardfork_is_enabled_in() {
        // MonadEight is enabled in every spec
        assert!(MonadHardfork::MonadEight.is_enabled_in(MonadHardfork::MonadEight));
        assert!(MonadHardfork::MonadEight.is_enabled_in(MonadHardfork::MonadNine));
        assert!(MonadHardfork::MonadEight.is_enabled_in(MonadHardfork::MonadNext));

        // MonadNine is NOT enabled in MonadEight
        assert!(!MonadHardfork::MonadNine.is_enabled_in(MonadHardfork::MonadEight));
        // MonadNine IS enabled in MonadNine and MonadNext
        assert!(MonadHardfork::MonadNine.is_enabled_in(MonadHardfork::MonadNine));
        assert!(MonadHardfork::MonadNine.is_enabled_in(MonadHardfork::MonadNext));

        // MonadNext is only enabled in MonadNext
        assert!(!MonadHardfork::MonadNext.is_enabled_in(MonadHardfork::MonadEight));
        assert!(!MonadHardfork::MonadNext.is_enabled_in(MonadHardfork::MonadNine));
        assert!(MonadHardfork::MonadNext.is_enabled_in(MonadHardfork::MonadNext));
    }

    #[test]
    fn test_monad_hardfork_ordering() {
        assert!(MonadHardfork::MonadEight < MonadHardfork::MonadNine);
        assert!(MonadHardfork::MonadNine < MonadHardfork::MonadNext);
    }

    #[test]
    fn test_monad_hardfork_from_impl() {
        let spec_id: SpecId = MonadHardfork::MonadEight.into();
        assert_eq!(spec_id, SpecId::PRAGUE);
        let spec_id: SpecId = MonadHardfork::MonadNine.into();
        assert_eq!(spec_id, SpecId::OSAKA);
    }
}
