use std::usize;

use crate::fields::goldilocks::Fr as F64;
use crate::mds_12;
use crate::mds_8;
use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_ff::Field;
use ark_ff::PrimeField;
use ark_ff::Zero;
use ark_ff::{BigInteger, BigInteger64};
use ark_serialize::Read;
use ark_std::{iterable::Iterable, marker::PhantomData};
use sha3::digest::ExtendableOutput;
use sha3::digest::Update;
use sha3::Shake128;
use sha3::Shake128Reader;
pub struct CRH64<const T: usize> {
    field_phantom: PhantomData<F64>,
}
#[derive(Clone)]
pub struct MonolithParams {
    bar_per_round: u8,
    rounds: u8,
    state_size: u32,
    round_constants: Vec<Vec<F64>>,
}
impl<const T: usize> CRH64<T> {
    pub fn s(byt: u8) -> u8 {
        (byt ^ (!byt.rotate_left(1) & byt.rotate_left(2) & byt.rotate_left(3))).rotate_left(1)
    }
    pub fn bar(element: F64) -> F64 {
        let mut be_bytes = element.into_bigint().to_bytes_be();
        for byt in &mut be_bytes {
            *byt = Self::s(*byt);
        }
        let ele = <F64 as PrimeField>::from_be_bytes_mod_order(&be_bytes);
        <F64 as PrimeField>::from_be_bytes_mod_order(&be_bytes)
    }
    pub fn bars(input: &mut [F64], params: &MonolithParams) {
        let mut out_bars: Vec<_> = vec![];
        for (ind, ele) in input.iter().enumerate().into_iter() {
            if ind >= params.bar_per_round.into() {
                out_bars.push(*ele);
                continue;
            }
            out_bars.push(Self::bar(*ele));
        }
        input.copy_from_slice(&out_bars[..]);
    }
    pub fn bricks(input: &mut [F64]) {
        for i in (1..input.len()).rev() {
            input[i] += input[i - 1] * input[i - 1];
        }
    }
    pub fn concrete_wrc(input: &mut [F64; T], round_constant: &[F64]) {
        if T == 8 {
            mds_8::mds_multiply_with_rc(
                input.as_mut().try_into().unwrap(),
                round_constant.try_into().unwrap(),
            );
        } else if T == 12 {
            mds_12::mds_multiply_with_rc(
                input.as_mut().try_into().unwrap(),
                round_constant.try_into().unwrap(),
            );
        }
    }
    pub fn concrete(input: &mut [F64; T]) {
        if T == 8 {
            mds_8::mds_multiply(input.as_mut().try_into().unwrap());
        } else if T == 12 {
            mds_12::mds_multiply(input.as_mut().try_into().unwrap());
        }
    }
}
impl<const Y: usize> CRHScheme for CRH64<Y> {
    type Input = [F64; Y];
    type Output = F64;
    type Parameters = MonolithParams;
    fn setup<R: ark_std::rand::prelude::Rng>(
        _r: &mut R,
    ) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        let rounds: u8 = 6;
        let state_size: u32 = Y.try_into()?;
        let mut round_constants: Vec<Vec<F64>> = vec![];
        let mut shake = Shake128::default();
        shake.update(b"Monolith");
        shake.update(&[Y as u8, rounds]);
        shake.update(&F64::MODULUS.to_bytes_le());
        shake.update(&[8, 8, 8, 8, 8, 8, 8, 8]);
        let mut shake_reader: Shake128Reader = shake.finalize_xof();
        while round_constants.len() + 1 < rounds.into() {
            let mut rands: Vec<F64> = vec![];
            loop {
                let mut rnd = [0u8; 8];
                shake_reader.read(&mut rnd)?;
                let ele = <F64 as Field>::from_random_bytes(&rnd);
                if ele.is_some() {
                    rands.push(ele.unwrap());
                }
                if rands.len() == Y {
                    break;
                }
            }
            round_constants.push(rands);
        }
        let last_rc = [<F64 as Zero>::zero(); Y];
        round_constants.push(last_rc.to_vec());
        Ok(MonolithParams {
            bar_per_round: 4,
            rounds,
            state_size,
            round_constants,
        })
    }
    fn evaluate<T: std::borrow::Borrow<Self::Input>>(
        parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut inp: Self::Input = *input.borrow();
        Self::concrete(&mut inp);

        for rc in parameters.round_constants.iter() {
            Self::bars(&mut inp, parameters);
            Self::bricks(&mut inp);
            Self::concrete_wrc(&mut inp, rc);
        }
        Ok(inp[0])
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::fields::goldilocks::Fr as FP64;
    use ark_ff::Zero;
    use ark_std::One;
    use rand::thread_rng;
    #[test]
    pub fn hash() {
        let input = [
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
            <FP64 as One>::one(),
        ];
        let mut rng = thread_rng();
        let params = CRH64::<8>::setup(&mut rng).unwrap();
        println!("rc: {:?}", params.round_constants);
        let out = CRH64::<8>::evaluate(&params, input);
    }
}
