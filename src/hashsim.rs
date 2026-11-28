//! Model-free simulation of how often three output hashes break when only
//! the batch size changes.
//!
//! Each sequence has `steps` greedy steps. At a step the top logit is 0 and
//! the runner-up sits a gap below it, the gap drawn from an exponential. The
//! batch-1 and batch-8 runs each add independent Gaussian noise of the given
//! sd to both logits. Hashes compared between the two runs:
//! - exact: breaks if any logit differs at all;
//! - token: breaks if any step picks a different argmax;
//! - tolerance: breaks if any logit differs by more than `tolerance`.
//!
//! The batch-invariant control adds no noise.

use crate::flipsim::XorShift;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Rates {
    pub rate_batch_1_vs_8: f64,
    pub broken: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HashSimResult {
    pub sequences: usize,
    pub steps: usize,
    pub gap_mean: f64,
    pub noise_sd: f64,
    pub tolerance: f64,
    pub hash_break: BTreeMap<String, Rates>,
    pub control: BTreeMap<String, Rates>,
}

fn run(rng: &mut XorShift, sequences: usize, steps: usize, gap_mean: f64, sd: f64, tol: f64) -> BTreeMap<String, Rates> {
    let (mut exact, mut token, mut tolerance) = (0usize, 0usize, 0usize);
    for _ in 0..sequences {
        let (mut e, mut t, mut o) = (false, false, false);
        for _ in 0..steps {
            let gap = -gap_mean * (1.0 - rng.uniform()).ln();
            let a1 = sd * rng.approx_normal();
            let b1 = -gap + sd * rng.approx_normal();
            let a8 = sd * rng.approx_normal();
            let b8 = -gap + sd * rng.approx_normal();
            if a1 != a8 || b1 != b8 {
                e = true;
            }
            if (a1 >= b1) != (a8 >= b8) {
                t = true;
            }
            if (a1 - a8).abs() > tol || (b1 - b8).abs() > tol {
                o = true;
            }
        }
        exact += e as usize;
        token += t as usize;
        tolerance += o as usize;
    }
    let n = sequences.max(1) as f64;
    let mut m = BTreeMap::new();
    for (k, v) in [("exact", exact), ("token", token), ("tolerance", tolerance)] {
        m.insert(k.to_string(), Rates { rate_batch_1_vs_8: v as f64 / n, broken: v });
    }
    m
}

pub fn simulate(seed: u64, sequences: usize, steps: usize, gap_mean: f64, noise_sd: f64, tolerance: f64) -> HashSimResult {
    let mut rng = XorShift::new(seed);
    let hash_break = run(&mut rng, sequences, steps, gap_mean, noise_sd, tolerance);
    let control = run(&mut rng, sequences, steps, gap_mean, 0.0, tolerance);
    HashSimResult { sequences, steps, gap_mean, noise_sd, tolerance, hash_break, control }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_never_breaks() {
        let r = simulate(7, 200, 16, 2.0, 0.001, 0.01);
        assert!(r.control.values().all(|x| x.broken == 0));
    }

    #[test]
    fn exact_breaks_always_with_noise_and_tolerance_rarely() {
        let r = simulate(7, 200, 16, 2.0, 0.001, 0.01);
        assert_eq!(r.hash_break["exact"].broken, 200);
        assert!(r.hash_break["tolerance"].rate_batch_1_vs_8 < 0.01);
    }

    #[test]
    fn deterministic() {
        assert_eq!(simulate(3, 50, 8, 2.0, 0.01, 0.01), simulate(3, 50, 8, 2.0, 0.01, 0.01));
    }
}
