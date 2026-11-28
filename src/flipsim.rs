//! Model-free simulation of label flips under reduction-order-sized noise.
//!
//! Scores are uniform on [-1, 1] with the decision boundary at 0. Each score
//! is perturbed by noise; a raw flip is a label change, and a flip is
//! adjudicated genuine only if the score moved by more than a fixed margin.

use serde::Serialize;

/// xorshift64* PRNG, so the simulation needs no dependency.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        // splitmix-style scramble so small seeds do not start near zero
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        XorShift(if z == 0 { 1 } else { z })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sum of 12 uniforms minus 6: mean 0, variance 1.
    pub fn approx_normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FlipStats {
    pub items: usize,
    pub raw_flips: usize,
    pub adjudicated_genuine: usize,
    pub raw_flip_rate: f64,
    pub adjudicated_flip_rate: f64,
    pub adjudicated_to_raw_flip_ratio: f64,
}

pub fn simulate(seed: u64, items: usize, noise_sd: f64, margin: f64) -> FlipStats {
    let mut rng = XorShift::new(seed);
    let (mut raw, mut genuine) = (0usize, 0usize);
    for _ in 0..items {
        let s = rng.uniform() * 2.0 - 1.0;
        let delta = noise_sd * rng.approx_normal();
        let p = s + delta;
        if (s >= 0.0) != (p >= 0.0) {
            raw += 1;
            if delta.abs() > margin {
                genuine += 1;
            }
        }
    }
    let n = items.max(1) as f64;
    FlipStats {
        items,
        raw_flips: raw,
        adjudicated_genuine: genuine,
        raw_flip_rate: raw as f64 / n,
        adjudicated_flip_rate: genuine as f64 / n,
        adjudicated_to_raw_flip_ratio: if raw == 0 { 0.0 } else { genuine as f64 / raw as f64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_seed() {
        assert_eq!(simulate(7, 2000, 0.02, 0.05), simulate(7, 2000, 0.02, 0.05));
    }

    #[test]
    fn zero_noise_has_no_flips() {
        let s = simulate(7, 5000, 0.0, 0.05);
        assert_eq!(s.raw_flips, 0);
        assert_eq!(s.adjudicated_to_raw_flip_ratio, 0.0);
    }

    #[test]
    fn genuine_never_exceeds_raw() {
        let s = simulate(3, 10000, 0.2, 0.05);
        assert!(s.adjudicated_genuine <= s.raw_flips);
        assert!(s.raw_flips > 0);
    }

    #[test]
    fn huge_margin_adjudicates_nothing() {
        let s = simulate(7, 10000, 0.05, 10.0);
        assert!(s.raw_flips > 0);
        assert_eq!(s.adjudicated_genuine, 0);
    }

    #[test]
    fn approx_normal_moments() {
        let mut r = XorShift::new(1);
        let v: Vec<f64> = (0..20000).map(|_| r.approx_normal()).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(m.abs() < 0.05 && (var - 1.0).abs() < 0.05);
    }
}
