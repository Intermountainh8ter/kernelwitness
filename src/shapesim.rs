//! Model-free simulation of bug detection versus shape-sampling strategy.
//!
//! A shape pool is the sizes 0..pool. Regular shapes are multiples of 64 and
//! no simulated bug fails on one. Bugs fail on a known subset of the other
//! shapes; a suite detects a bug if any sampled shape is in that subset.

use crate::flipsim::XorShift;
use serde::Serialize;
use std::collections::BTreeMap;

pub const BLOCK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    TailMask,
    OffByOne,
    BlockSize,
    None,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::TailMask => "tail_mask",
            Class::OffByOne => "off_by_one",
            Class::BlockSize => "block_size",
            Class::None => "zero_bug_control",
        }
    }
}

/// Failing subset (as a pool-length mask) for one simulated bug.
pub fn failing_set(rng: &mut XorShift, class: Class, pool: usize) -> Vec<bool> {
    let mut f = vec![false; pool];
    match class {
        // the last w shapes before each block boundary, w in 1..=8
        Class::TailMask => {
            let w = 1 + (rng.next_u64() % 8) as usize;
            for (i, x) in f.iter_mut().enumerate() {
                if i % BLOCK >= BLOCK - w {
                    *x = true;
                }
            }
        }
        // exactly one side of each block boundary: size = 64k-1 or 64k+1
        Class::OffByOne => {
            let r = if rng.uniform() < 0.5 { BLOCK - 1 } else { 1 };
            for (i, x) in f.iter_mut().enumerate() {
                if i % BLOCK == r {
                    *x = true;
                }
            }
        }
        // a random eighth of the irregular shapes
        Class::BlockSize => {
            let irregular: Vec<usize> = (0..pool).filter(|i| i % BLOCK != 0).collect();
            let k = irregular.len() / 8;
            let mut n = 0;
            while n < k {
                let i = irregular[(rng.next_u64() % irregular.len() as u64) as usize];
                if !f[i] {
                    f[i] = true;
                    n += 1;
                }
            }
        }
        Class::None => {}
    }
    f
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    FixedRegular,
    Uniform,
    BoundaryBiased,
}

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Strategy::FixedRegular => "fixed_regular",
            Strategy::Uniform => "uniform",
            Strategy::BoundaryBiased => "boundary_biased",
        }
    }
}

/// Draw `n` distinct shapes. Boundary-biased: each draw comes with
/// probability 0.5 from the shapes adjacent to a block boundary, otherwise
/// uniform from the pool.
pub fn sample(rng: &mut XorShift, strategy: Strategy, pool: usize, n: usize) -> Vec<usize> {
    let regular: Vec<usize> = (0..pool).filter(|i| i % BLOCK == 0).collect();
    let near: Vec<usize> = (0..pool)
        .filter(|i| i % BLOCK == 1 || i % BLOCK == BLOCK - 1)
        .collect();
    let n = n.min(pool);
    let mut out: Vec<usize> = Vec::new();
    match strategy {
        Strategy::FixedRegular => {
            // regular shapes in order, repeating when n exceeds their count
            for k in 0..n {
                out.push(regular[k % regular.len()]);
            }
        }
        _ => {
            while out.len() < n {
                let from_near = strategy == Strategy::BoundaryBiased && rng.uniform() < 0.5;
                let s = if from_near {
                    near[(rng.next_u64() % near.len() as u64) as usize]
                } else {
                    (rng.next_u64() % pool as u64) as usize
                };
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ShapeSimResult {
    pub bugs_per_class: usize,
    pub pool: usize,
    pub detection: BTreeMap<String, f64>,
    pub per_class: BTreeMap<String, f64>,
    pub control_rate: f64,
}

pub fn simulate(seed: u64, bugs_per_class: usize, pool: usize, counts: &[usize]) -> ShapeSimResult {
    let mut rng = XorShift::new(seed);
    let strategies = [Strategy::FixedRegular, Strategy::Uniform, Strategy::BoundaryBiased];
    let classes = [Class::TailMask, Class::OffByOne, Class::BlockSize];
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_class: BTreeMap<String, f64> = BTreeMap::new();
    let mut control_hits = 0usize;
    for class in classes.iter().chain(std::iter::once(&Class::None)) {
        for _ in 0..bugs_per_class {
            let f = failing_set(&mut rng, *class, pool);
            for st in strategies {
                for &n in counts {
                    let s = sample(&mut rng, st, pool, n);
                    let det = s.iter().any(|&i| f[i]);
                    if *class == Class::None {
                        if det {
                            control_hits += 1;
                        }
                    } else if det {
                        *hits.entry(format!("{}_rate_at_{}_shapes", st.name(), n)).or_insert(0) += 1;
                        if n == *counts.last().unwrap() {
                            *per_class
                                .entry(format!("{}_{}_rate_at_{}_shapes", st.name(), class.name(), n))
                                .or_insert(0.0) += 1.0;
                        }
                    }
                }
            }
        }
    }
    let total = (bugs_per_class * classes.len()).max(1) as f64;
    let mut detection = BTreeMap::new();
    for st in strategies {
        for &n in counts {
            let k = format!("{}_rate_at_{}_shapes", st.name(), n);
            detection.insert(k.clone(), *hits.get(&k).unwrap_or(&0) as f64 / total);
        }
    }
    for v in per_class.values_mut() {
        *v /= bugs_per_class.max(1) as f64;
    }
    let control_rate = control_hits as f64 / (bugs_per_class * strategies.len() * counts.len()).max(1) as f64;
    ShapeSimResult { bugs_per_class, pool, detection, per_class, control_rate }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_regular_never_detects_and_control_is_zero() {
        let r = simulate(7, 100, 256, &[1, 4]);
        assert_eq!(r.detection["fixed_regular_rate_at_1_shapes"], 0.0);
        assert_eq!(r.detection["fixed_regular_rate_at_4_shapes"], 0.0);
        assert_eq!(r.control_rate, 0.0);
    }

    #[test]
    fn no_bug_fails_on_a_regular_shape() {
        let mut rng = XorShift::new(1);
        for c in [Class::TailMask, Class::OffByOne, Class::BlockSize] {
            for _ in 0..50 {
                let f = failing_set(&mut rng, c, 256);
                assert!(f.iter().enumerate().all(|(i, x)| !x || i % BLOCK != 0));
                assert!(f.iter().any(|x| *x));
            }
        }
    }

    #[test]
    fn samples_are_distinct_and_deterministic() {
        let mut a = XorShift::new(3);
        let s = sample(&mut a, Strategy::BoundaryBiased, 256, 16);
        let mut u = s.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), 16);
        assert_eq!(simulate(7, 50, 256, &[2, 8]), simulate(7, 50, 256, &[2, 8]));
    }
}
