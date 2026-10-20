//! Staking argument decoding matching the C++ precompile's input consumption.

use super::interface::IMonadStaking::addValidatorCall;
use alloy_sol_types::{SolCall, SolType};
use revm::{
    precompile::PrecompileHalt,
    primitives::{Bytes, U256},
};

/// Decode a fixed-size call, ignoring dirty high bits but rejecting leftover bytes.
pub(super) fn decode_fixed<C: SolCall>(input: &[u8]) -> Result<C, PrecompileHalt> {
    let data = input.get(4..).ok_or_else(|| PrecompileHalt::other_static("input too short"))?;
    let size = <C::Parameters<'_> as SolType>::ENCODED_SIZE
        .expect("fixed staking arguments must have a static ABI size");
    if data.len() < size {
        return Err(PrecompileHalt::other_static("input too short"));
    }
    if data.len() > size {
        return Err(PrecompileHalt::other_static("invalid input"));
    }

    // C++ reads only the low bytes of integers and addresses, without padding validation.
    C::abi_decode_raw(data).map_err(|_| PrecompileHalt::other_static("invalid input"))
}

/// Decode the three sequential fixed-length byte tails, ignoring head offsets and padding.
pub(super) fn decode_add_validator(input: &[u8]) -> Result<addValidatorCall, PrecompileHalt> {
    let mut data = input.get(4..).ok_or_else(|| PrecompileHalt::other_static("input too short"))?;
    take_bytes(&mut data, 3 * 32)?;
    let payload = decode_bytes_tail(&mut data, 165)?;
    let signed_secp_message = decode_bytes_tail(&mut data, 64)?;
    let signed_bls_message = decode_bytes_tail(&mut data, 96)?;
    if !data.is_empty() {
        return Err(PrecompileHalt::other_static("invalid input"));
    }
    Ok(addValidatorCall {
        payload,
        signedSecpMessage: signed_secp_message,
        signedBlsMessage: signed_bls_message,
    })
}

fn decode_bytes_tail(data: &mut &[u8], size: usize) -> Result<Bytes, PrecompileHalt> {
    let length = U256::from_be_slice(take_bytes(data, 32)?);
    if length != U256::from(size) {
        return Err(PrecompileHalt::other_static("length mismatch"));
    }
    let padded = take_bytes(data, size.next_multiple_of(32))?;
    Ok(Bytes::copy_from_slice(&padded[..size]))
}

fn take_bytes<'a>(data: &mut &'a [u8], size: usize) -> Result<&'a [u8], PrecompileHalt> {
    let (head, tail) = data
        .split_at_checked(size)
        .ok_or_else(|| PrecompileHalt::other_static("input too short"))?;
    *data = tail;
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staking::interface::IMonadStaking::{
        getConsensusValidatorSetCall, getWithdrawalRequestCall, undelegateCall,
    };
    use revm::primitives::Address;

    #[test]
    fn fixed_values_retain_low_bytes() {
        let address = Address::repeat_byte(0x11);
        let mut input = getWithdrawalRequestCall {
            validatorId: 0x123456789abcdef0,
            delegator: address,
            withdrawId: 7,
        }
        .abi_encode();
        input[4..28].fill(0xff);
        input[36..48].fill(0xff);
        input[68..99].fill(0xff);
        let call = decode_fixed::<getWithdrawalRequestCall>(&input).unwrap();
        assert_eq!(call.validatorId, 0x123456789abcdef0);
        assert_eq!(call.delegator, address);
        assert_eq!(call.withdrawId, 7);

        let mut input = getConsensusValidatorSetCall { startIndex: 0x12345678 }.abi_encode();
        input[4..32].fill(0xff);
        assert_eq!(
            decode_fixed::<getConsensusValidatorSetCall>(&input).unwrap().startIndex,
            0x12345678
        );

        let input =
            undelegateCall { validatorId: 1, amount: U256::MAX, withdrawId: 7 }.abi_encode();
        assert_eq!(decode_fixed::<undelegateCall>(&input).unwrap().amount, U256::MAX);
    }
}
