use ark_ff::fields::{Fp64, MontBackend, MontConfig};
#[derive(MontConfig)]
#[modulus = "2147483647"]
#[generator = "7"]
pub struct FrConfig;
pub type Fr = Fp64<MontBackend<FrConfig, 1>>;
