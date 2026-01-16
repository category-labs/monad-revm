//! Contains Monad specific precompiles.
//!
//! Monad reprices several precompiles to accurately reflect their relative costs.
//! See: <https://docs.monad.xyz/developer-essentials/opcode-pricing#precompiles>
//!
//! | Precompile  | Address | Ethereum | Monad   | Multiplier |
//! |-------------|---------|----------|---------|------------|
//! | ecRecover   | 0x01    | 3,000    | 6,000   | 2x         |
//! | ecAdd       | 0x06    | 150      | 300     | 2x         |
//! | ecMul       | 0x07    | 6,000    | 30,000  | 5x         |
//! | ecPairing   | 0x08    | 45,000*  | 225,000*| 5x         |
//! | blake2f     | 0x09    | rounds×1 | rounds×2| 2x         |
//! | point eval  | 0x0a    | 50,000   | 200,000 | 4x         |
//!
//! *Base cost per operation

use crate::MonadSpecId;
use revm::{
    context::Cfg,
    context_interface::ContextTr,
    handler::{EthPrecompiles, PrecompileProvider},
    interpreter::{CallInputs, InterpreterResult},
    precompile::{
        bn254, kzg_point_evaluation, secp256r1, Precompile, PrecompileError, PrecompileId,
        PrecompileOutput, PrecompileResult, Precompiles,
    },
    primitives::{alloy_primitives::B512, hardfork::SpecId, Address, Bytes, B256},
};
use std::{boxed::Box, string::String};

// ═══════════════════════════════════════════════════════════════════════════════
// Monad Gas Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Monad ecRecover gas cost (2x Ethereum's 3000)
pub const MONAD_ECRECOVER_GAS: u64 = 6_000;

/// Monad ecAdd gas cost (2x Ethereum's 150)
pub const MONAD_EC_ADD_GAS: u64 = 300;

/// Monad ecMul gas cost (5x Ethereum's 6000)
pub const MONAD_EC_MUL_GAS: u64 = 30_000;

/// Monad ecPairing base gas cost (5x Ethereum's 45000)
pub const MONAD_EC_PAIRING_BASE_GAS: u64 = 225_000;

/// Monad ecPairing per-point gas cost (5x Ethereum's 34000)
pub const MONAD_EC_PAIRING_PER_POINT_GAS: u64 = 170_000;

/// Monad blake2f gas multiplier (2x Ethereum's 1)
pub const MONAD_BLAKE2F_ROUND_GAS: u64 = 2;

/// Monad KZG point evaluation gas cost (4x Ethereum's 50000)
pub const MONAD_POINT_EVALUATION_GAS: u64 = 200_000;

// ═══════════════════════════════════════════════════════════════════════════════
// Monad Precompile Run Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Monad ecRecover precompile run function
fn monad_ecrecover_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    use revm::precompile::{crypto, utilities::right_pad};

    if MONAD_ECRECOVER_GAS > gas_limit {
        return Err(PrecompileError::OutOfGas);
    }

    let input = right_pad::<128>(input);

    // `v` must be a 32-byte big-endian integer equal to 27 or 28.
    if !(input[32..63].iter().all(|&b| b == 0) && matches!(input[63], 27 | 28)) {
        return Ok(PrecompileOutput::new(MONAD_ECRECOVER_GAS, Bytes::new()));
    }

    let msg = <&B256>::try_from(&input[0..32]).unwrap();
    let recid = input[63] - 27;
    let sig = <&B512>::try_from(&input[64..128]).unwrap();

    let res = crypto().secp256k1_ecrecover(&sig.0, recid, &msg.0).ok();
    let out = res.map(|o| o.to_vec().into()).unwrap_or_default();
    Ok(PrecompileOutput::new(MONAD_ECRECOVER_GAS, out))
}

/// Monad ecAdd precompile run function
fn monad_ec_add_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    bn254::run_add(input, MONAD_EC_ADD_GAS, gas_limit)
}

/// Monad ecMul precompile run function
fn monad_ec_mul_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    bn254::run_mul(input, MONAD_EC_MUL_GAS, gas_limit)
}

/// Monad ecPairing precompile run function
fn monad_ec_pairing_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    bn254::run_pair(
        input,
        MONAD_EC_PAIRING_PER_POINT_GAS,
        MONAD_EC_PAIRING_BASE_GAS,
        gas_limit,
    )
}

/// Monad blake2f precompile run function
fn monad_blake2f_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    use revm::precompile::crypto;

    const INPUT_LENGTH: usize = 213;

    if input.len() != INPUT_LENGTH {
        return Err(PrecompileError::Blake2WrongLength);
    }

    // Parse number of rounds (4 bytes)
    let rounds = u32::from_be_bytes(input[..4].try_into().unwrap());
    let gas_used = rounds as u64 * MONAD_BLAKE2F_ROUND_GAS;
    if gas_used > gas_limit {
        return Err(PrecompileError::OutOfGas);
    }

    // Parse final block flag
    let f = match input[212] {
        0 => false,
        1 => true,
        _ => return Err(PrecompileError::Blake2WrongFinalIndicatorFlag),
    };

    // Parse state vector h (8 × u64)
    let mut h = [0u64; 8];
    input[4..68]
        .chunks_exact(8)
        .enumerate()
        .for_each(|(i, chunk)| {
            h[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        });

    // Parse message block m (16 × u64)
    let mut m = [0u64; 16];
    input[68..196]
        .chunks_exact(8)
        .enumerate()
        .for_each(|(i, chunk)| {
            m[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        });

    // Parse offset counters
    let t_0 = u64::from_le_bytes(input[196..204].try_into().unwrap());
    let t_1 = u64::from_le_bytes(input[204..212].try_into().unwrap());

    crypto().blake2_compress(rounds, &mut h, m, [t_0, t_1], f);

    let mut out = [0u8; 64];
    for (i, h) in (0..64).step_by(8).zip(h.iter()) {
        out[i..i + 8].copy_from_slice(&h.to_le_bytes());
    }

    Ok(PrecompileOutput::new(gas_used, out.into()))
}

/// Monad KZG point evaluation precompile run function
fn monad_point_evaluation_run(input: &[u8], gas_limit: u64) -> PrecompileResult {
    use revm::precompile::crypto;

    if gas_limit < MONAD_POINT_EVALUATION_GAS {
        return Err(PrecompileError::OutOfGas);
    }

    // Verify input length.
    if input.len() != 192 {
        return Err(PrecompileError::BlobInvalidInputLength);
    }

    // Verify commitment matches versioned_hash
    let versioned_hash = &input[..32];
    let commitment = &input[96..144];
    if kzg_point_evaluation::kzg_to_versioned_hash(commitment) != versioned_hash {
        return Err(PrecompileError::BlobMismatchedVersion);
    }

    // Verify KZG proof with z and y in big endian format
    let commitment: &[u8; 48] = commitment.try_into().unwrap();
    let z = input[32..64].try_into().unwrap();
    let y = input[64..96].try_into().unwrap();
    let proof = input[144..192].try_into().unwrap();
    crypto().verify_kzg_proof(z, y, commitment, proof)?;

    // Return FIELD_ELEMENTS_PER_BLOB and BLS_MODULUS as padded 32 byte big endian values
    Ok(PrecompileOutput::new(
        MONAD_POINT_EVALUATION_GAS,
        kzg_point_evaluation::RETURN_VALUE.into(),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Monad Precompile Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Monad ecRecover precompile (address 0x01, 6000 gas)
pub const MONAD_ECRECOVER: Precompile = Precompile::new(
    PrecompileId::EcRec,
    revm::precompile::u64_to_address(1),
    monad_ecrecover_run,
);

/// Monad ecAdd precompile (address 0x06, 300 gas)
pub const MONAD_EC_ADD: Precompile = Precompile::new(
    PrecompileId::Bn254Add,
    revm::precompile::u64_to_address(6),
    monad_ec_add_run,
);

/// Monad ecMul precompile (address 0x07, 30000 gas)
pub const MONAD_EC_MUL: Precompile = Precompile::new(
    PrecompileId::Bn254Mul,
    revm::precompile::u64_to_address(7),
    monad_ec_mul_run,
);

/// Monad ecPairing precompile (address 0x08, 225000 base + 170000 per point)
pub const MONAD_EC_PAIRING: Precompile = Precompile::new(
    PrecompileId::Bn254Pairing,
    revm::precompile::u64_to_address(8),
    monad_ec_pairing_run,
);

/// Monad blake2f precompile (address 0x09, rounds × 2 gas)
pub const MONAD_BLAKE2F: Precompile = Precompile::new(
    PrecompileId::Blake2F,
    revm::precompile::u64_to_address(9),
    monad_blake2f_run,
);

/// Monad KZG point evaluation precompile (address 0x0a, 200000 gas)
pub const MONAD_POINT_EVALUATION: Precompile = Precompile::new(
    PrecompileId::KzgPointEvaluation,
    revm::precompile::u64_to_address(0x0A),
    monad_point_evaluation_run,
);

// ═══════════════════════════════════════════════════════════════════════════════
// Monad Precompile Provider
// ═══════════════════════════════════════════════════════════════════════════════

/// Monad precompile provider
#[derive(Debug, Clone)]
pub struct MonadPrecompiles {
    /// Inner precompile provider with Monad-specific gas costs.
    inner: EthPrecompiles,
    /// Spec id of the precompile provider.
    spec: MonadSpecId,
}

impl MonadPrecompiles {
    /// Create a new precompile provider with the given spec.
    #[inline]
    pub fn new_with_spec(spec: MonadSpecId) -> Self {
        // Start with Ethereum precompiles for the underlying spec
        let mut precompiles = Precompiles::new(spec.into_eth_spec().into()).clone();

        // Override with Monad-specific gas costs
        precompiles.extend([
            MONAD_ECRECOVER,
            MONAD_EC_ADD,
            MONAD_EC_MUL,
            MONAD_EC_PAIRING,
            MONAD_BLAKE2F,
            MONAD_POINT_EVALUATION,
        ]);

        // Add P256VERIFY precompile (RIP-7212 / EIP-7951)
        // Address: 0x0100, Gas: 3450 (same as Ethereum pre-Osaka)
        precompiles.extend([secp256r1::P256VERIFY]);

        Self {
            inner: EthPrecompiles {
                precompiles: Box::leak(Box::new(precompiles)),
                spec: SpecId::default(),
            },
            spec,
        }
    }

    /// Precompiles getter.
    #[inline]
    pub fn precompiles(&self) -> &'static Precompiles {
        self.inner.precompiles
    }
}

impl<CTX> PrecompileProvider<CTX> for MonadPrecompiles
where
    CTX: ContextTr<Cfg: Cfg<Spec = MonadSpecId>>,
{
    type Output = InterpreterResult;

    #[inline]
    fn set_spec(&mut self, spec: <CTX::Cfg as Cfg>::Spec) -> bool {
        if spec == self.spec {
            return false;
        }
        *self = Self::new_with_spec(spec);
        true
    }

    #[inline]
    fn run(
        &mut self,
        context: &mut CTX,
        inputs: &CallInputs,
    ) -> Result<Option<Self::Output>, String> {
        self.inner.run(context, inputs)
    }

    #[inline]
    fn warm_addresses(&self) -> Box<impl Iterator<Item = Address>> {
        self.inner.warm_addresses()
    }

    #[inline]
    fn contains(&self, address: &Address) -> bool {
        self.inner.contains(address)
    }
}

impl Default for MonadPrecompiles {
    fn default() -> Self {
        Self::new_with_spec(MonadSpecId::MonadEight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_precompile_gas_costs() {
        // Verify Monad gas costs are correctly set
        assert_eq!(MONAD_ECRECOVER_GAS, 6_000);
        assert_eq!(MONAD_EC_ADD_GAS, 300);
        assert_eq!(MONAD_EC_MUL_GAS, 30_000);
        assert_eq!(MONAD_EC_PAIRING_BASE_GAS, 225_000);
        assert_eq!(MONAD_EC_PAIRING_PER_POINT_GAS, 170_000);
        assert_eq!(MONAD_BLAKE2F_ROUND_GAS, 2);
        assert_eq!(MONAD_POINT_EVALUATION_GAS, 200_000);
    }

    #[test]
    fn test_monad_precompiles_contains_addresses() {
        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Check all standard Ethereum precompiles are present (0x01-0x0a)
        // <https://docs.monad.xyz/developer-essentials/precompiles>
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x01)),
            "ecRecover (0x01) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x02)),
            "sha256 (0x02) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x03)),
            "ripemd160 (0x03) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x04)),
            "identity (0x04) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x05)),
            "modexp (0x05) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x06)),
            "ecAdd (0x06) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x07)),
            "ecMul (0x07) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x08)),
            "ecPairing (0x08) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x09)),
            "blake2f (0x09) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0a)),
            "point_eval (0x0a) should exist"
        );

        // Check BLS12-381 precompiles are present (0x0b-0x11)
        // These are part of the PRAGUE spec
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0b)),
            "bls12_g1_add (0x0b) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0c)),
            "bls12_g1_msm (0x0c) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0d)),
            "bls12_g2_add (0x0d) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0e)),
            "bls12_g2_msm (0x0e) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0f)),
            "bls12_pairing_check (0x0f) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x10)),
            "bls12_map_fp_to_g1 (0x10) should exist"
        );
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x11)),
            "bls12_map_fp2_to_g2 (0x11) should exist"
        );

        // P256VERIFY precompile (RIP-7212 / EIP-7951)
        assert!(
            precompiles.contains(&revm::precompile::u64_to_address(0x0100)),
            "p256_verify (0x0100) should exist"
        );

        // TODO: Add Monad-specific precompiles when implemented
        // assert!(precompiles.contains(&revm::precompile::u64_to_address(0x1000)), "staking (0x1000) should exist");
    }

    #[test]
    fn test_monad_ecadd_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the ecAdd precompile (address 0x06)
        let ec_add_address = revm::precompile::u64_to_address(6);
        let precompile = precompiles
            .get(&ec_add_address)
            .expect("ecAdd should exist");

        // Valid ecAdd input: two points on BN254 curve
        // P = (1, 2) which is the generator point
        let input = hex::decode(
            "0000000000000000000000000000000000000000000000000000000000000001\
             0000000000000000000000000000000000000000000000000000000000000002\
             0000000000000000000000000000000000000000000000000000000000000001\
             0000000000000000000000000000000000000000000000000000000000000002",
        )
        .unwrap();

        // Execute with high gas limit
        let result = precompile
            .execute(&input, 100_000)
            .expect("ecAdd should succeed");

        // Verify Monad gas cost is used (300, not Ethereum's 150)
        assert_eq!(
            result.gas_used, MONAD_EC_ADD_GAS,
            "ecAdd should use Monad gas cost of 300"
        );
    }

    #[test]
    fn test_monad_ecmul_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the ecMul precompile (address 0x07)
        let ec_mul_address = revm::precompile::u64_to_address(7);
        let precompile = precompiles
            .get(&ec_mul_address)
            .expect("ecMul should exist");

        // Valid ecMul input: point (1, 2) and scalar 2
        let input = hex::decode(
            "0000000000000000000000000000000000000000000000000000000000000001\
             0000000000000000000000000000000000000000000000000000000000000002\
             0000000000000000000000000000000000000000000000000000000000000002",
        )
        .unwrap();

        // Execute with high gas limit
        let result = precompile
            .execute(&input, 100_000)
            .expect("ecMul should succeed");

        // Verify Monad gas cost is used (30000, not Ethereum's 6000)
        assert_eq!(
            result.gas_used, MONAD_EC_MUL_GAS,
            "ecMul should use Monad gas cost of 30000"
        );
    }

    #[test]
    fn test_monad_ecrecover_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the ecRecover precompile (address 0x01)
        let ecrecover_address = revm::precompile::u64_to_address(1);
        let precompile = precompiles
            .get(&ecrecover_address)
            .expect("ecRecover should exist");

        // Valid ecrecover input: hash + v + r + s
        let input = hex::decode(
            "456e9aea5e197a1f1af7a3e85a3212fa4049a3ba34c2289b4c860fc0b0c64ef3\
             000000000000000000000000000000000000000000000000000000000000001c\
             9242685bf161793cc25603c231bc2f568eb630ea16aa137d2664ac8038825608\
             4f8ae3bd7535248d0bd448298cc2e2071e56992d0774dc340c368ae950852ada",
        )
        .unwrap();

        // Execute with high gas limit
        let result = precompile
            .execute(&input, 100_000)
            .expect("ecRecover should succeed");

        // Verify Monad gas cost is used (6000, not Ethereum's 3000)
        assert_eq!(
            result.gas_used, MONAD_ECRECOVER_GAS,
            "ecRecover should use Monad gas cost of 6000"
        );
    }

    #[test]
    fn test_monad_ecpairing_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the ecPairing precompile (address 0x08)
        let ec_pairing_address = revm::precompile::u64_to_address(8);
        let precompile = precompiles
            .get(&ec_pairing_address)
            .expect("ecPairing should exist");

        // Single pairing with G1 point at infinity (0,0) + valid G2 point
        let input = hex::decode(
            "0000000000000000000000000000000000000000000000000000000000000000\
             0000000000000000000000000000000000000000000000000000000000000000\
             198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2\
             1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed\
             090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b\
             12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
        )
        .unwrap();

        // Execute with high gas limit (1 point = base + per_point = 225000 + 170000 = 395000)
        let result = precompile
            .execute(&input, 500_000)
            .expect("ecPairing should succeed");

        // Verify Monad gas cost is used (225000 base + 170000 per point = 395000)
        let expected_gas = MONAD_EC_PAIRING_BASE_GAS + MONAD_EC_PAIRING_PER_POINT_GAS;
        assert_eq!(
            result.gas_used, expected_gas,
            "ecPairing should use Monad gas cost of 395000"
        );
    }

    #[test]
    fn test_monad_blake2f_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the blake2f precompile (address 0x09)
        let blake2f_address = revm::precompile::u64_to_address(9);
        let precompile = precompiles
            .get(&blake2f_address)
            .expect("blake2f should exist");

        // blake2f input: 4 bytes rounds + 64 bytes h + 128 bytes m + 8 bytes t[0] + 8 bytes t[1] + 1 byte f
        // 12 rounds (0x0000000c)
        let input = hex::decode(
            "0000000c\
             48c9bdf267e6096a3ba7ca8485ae67bb2bf894fe72f36e3cf1361d5f3af54fa5\
             d182e6ad7f520e511f6c3e2b8c68059b6bbd41fbabd9831f79217e1319cde05b\
             6162630000000000000000000000000000000000000000000000000000000000\
             0000000000000000000000000000000000000000000000000000000000000000\
             0000000000000000000000000000000000000000000000000000000000000000\
             0000000000000000000000000000000000000000000000000000000000000000\
             0300000000000000\
             0000000000000000\
             01",
        )
        .unwrap();

        // Execute with high gas limit (12 rounds × 2 = 24 gas)
        let result = precompile
            .execute(&input, 100)
            .expect("blake2f should succeed");

        // Verify Monad gas cost is used (rounds × 2)
        let expected_gas = 12 * MONAD_BLAKE2F_ROUND_GAS;
        assert_eq!(
            result.gas_used, expected_gas,
            "blake2f should use Monad gas cost of 24 for 12 rounds"
        );
    }

    #[test]
    fn test_p256verify_precompile_gas_cost() {
        use revm::primitives::hex;

        let monad_precompiles = MonadPrecompiles::default();
        let precompiles = monad_precompiles.precompiles();

        // Get the P256VERIFY precompile (address 0x0100)
        let p256verify_address = revm::precompile::u64_to_address(0x0100);
        let precompile = precompiles
            .get(&p256verify_address)
            .expect("P256VERIFY should exist");

        // Valid P256 signature verification input (160 bytes):
        // msg hash (32) + r (32) + s (32) + pubkey x (32) + pubkey y (32)
        // Test vector from RIP-7212
        let input = hex::decode(
            "4cee90eb86eaa050036147a12d49004b6b9c72bd725d39d4785011fe190f0b4d\
             a73bd4903f0ce3b639bbbf6e8e80d16931ff4bcf5993d58468e8fb19086e8cac\
             36dbcd03009df8c59286b162af3bd7fcc0450c9aa81be5d10d312af6c66b1d60\
             4aebd3099c618202fcfe16ae7770b0c49ab5eadf74b754204a3bb6060e44eff3\
             7618b065f9832de4ca6ca971a7a1adc826d0f7c00181a5fb2ddf79ae00b4e10e",
        )
        .unwrap();

        // Execute with high gas limit
        let result = precompile
            .execute(&input, 10_000)
            .expect("P256VERIFY should succeed");

        // Verify Ethereum pre-Osaka gas cost is used (3450)
        assert_eq!(
            result.gas_used,
            revm::precompile::secp256r1::P256VERIFY_BASE_GAS_FEE,
            "P256VERIFY should use Ethereum gas cost of 3450"
        );
    }
}
