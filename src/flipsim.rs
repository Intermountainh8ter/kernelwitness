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

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScreenPoint {
    pub threshold: f64,
    pub flagged: usize,
    pub true_flagged: usize,
    pub screen_precision: f64,
}

/// Candidate flips (label changed) are flagged when the score moved by more
/// than each threshold. A fraction `true_flip_rate` of items are true
/// behavioural flips: the score moves by 0.1 to 0.5 across the boundary.
/// Precision is true flips among flagged; 0 when nothing is flagged.
pub fn screen_sweep(
    seed: u64,
    items: usize,
    noise_sd: f64,
    true_flip_rate: f64,
    thresholds: &[f64],
) -> Vec<ScreenPoint> {
    let mut rng = XorShift::new(seed);
    let mut flagged = vec![0usize; thresholds.len()];
    let mut truef = vec![0usize; thresholds.len()];
    for _ in 0..items {
        let is_true = rng.uniform() < true_flip_rate;
        let (s, base) = if is_true {
            let m = 0.1 + 0.4 * rng.uniform();
            let a = m * rng.uniform();
            let sign = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
            (sign * a, -sign * (m - a) - sign * a)
        } else {
            (rng.uniform() * 2.0 - 1.0, 0.0)
        };
        let delta = base + noise_sd * rng.approx_normal();
        let p = s + delta;
        if (s >= 0.0) == (p >= 0.0) {
            continue;
        }
        for (i, t) in thresholds.iter().enumerate() {
            if delta.abs() > *t {
                flagged[i] += 1;
                if is_true {
                    truef[i] += 1;
                }
            }
        }
    }
    thresholds
        .iter()
        .enumerate()
        .map(|(i, t)| ScreenPoint {
            threshold: *t,
            flagged: flagged[i],
            true_flagged: truef[i],
            screen_precision: if flagged[i] == 0 { 0.0 } else { truef[i] as f64 / flagged[i] as f64 },
        })
        .collect()
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

    #[test]
    fn screen_deterministic_and_bounded() {
        let t = [0.01, 0.05];
        let a = screen_sweep(7, 5000, 0.02, 0.01, &t);
        assert_eq!(a, screen_sweep(7, 5000, 0.02, 0.01, &t));
        assert!(a.iter().all(|p| p.true_flagged <= p.flagged));
        assert!(a[0].flagged >= a[1].flagged);
    }

    #[test]
    fn screen_zero_true_rate_has_zero_precision() {
        let a = screen_sweep(7, 20000, 0.02, 0.0, &[0.01]);
        assert!(a[0].flagged > 0);
        assert_eq!(a[0].screen_precision, 0.0);
    }

    #[test]
    fn screen_true_flips_always_cross() {
        let a = screen_sweep(7, 20000, 0.0, 0.01, &[0.0]);
        assert!(a[0].flagged > 0);
        assert_eq!(a[0].flagged, a[0].true_flagged);
    }
}
