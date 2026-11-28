//! Model-free simulation of how often a prefix-cache hit and a cache miss
//! disagree on the argmax because the f32 accumulation order differs.
//!
//! Each position has two competing logits, each a dot product of a
//! standard-normal activation with a standard-normal weight vector of length
//! `hidden`. A bias on the second logit sets the true gap to an exponential
//! draw with mean `gap_mean`. The dot product sums products sequentially in
//! f32 within chunks, then adds the chunk sums sequentially. A cache hit uses
//! `chunk_hit`, a miss uses `chunk_miss`. The control compares the hit order
//! with itself.

use crate::flipsim::XorShift;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Flip {
    pub flips: usize,
    pub rate_overall: f64,
    pub max_abs_logit_diff: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CacheSimResult {
    pub positions: usize,
    pub hidden: usize,
    pub gap_mean: f64,
    pub chunk_hit: usize,
    pub chunk_miss: usize,
    pub argmax_flip: Flip,
    pub control: Flip,
}

pub fn chunked_dot(x: &[f32], w: &[f32], chunk: usize) -> f32 {
    let chunk = chunk.max(1);
    let mut total = 0.0f32;
    for (xc, wc) in x.chunks(chunk).zip(w.chunks(chunk)) {
        let mut s = 0.0f32;
        for (a, b) in xc.iter().zip(wc) {
            s += a * b;
        }
        total += s;
    }
    total
}

fn normal_vec(rng: &mut XorShift, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.approx_normal() as f32).collect()
}

pub fn simulate(seed: u64, positions: usize, hidden: usize, gap_mean: f64, chunk_hit: usize, chunk_miss: usize) -> CacheSimResult {
    let mut rng = XorShift::new(seed);
    let (mut flips, mut cflips) = (0usize, 0usize);
    let (mut maxd, mut cmaxd) = (0.0f64, 0.0f64);
    for _ in 0..positions {
        let x = normal_vec(&mut rng, hidden);
        let wa = normal_vec(&mut rng, hidden);
        let wb = normal_vec(&mut rng, hidden);
        let gap = -gap_mean * (1.0 - rng.uniform()).ln();
        let a_hit = chunked_dot(&x, &wa, chunk_hit);
        let b_raw_hit = chunked_dot(&x, &wb, chunk_hit);
        let bias = a_hit - b_raw_hit - gap as f32;
        let b_hit = b_raw_hit + bias;
        let a_miss = chunked_dot(&x, &wa, chunk_miss);
        let b_miss = chunked_dot(&x, &wb, chunk_miss) + bias;
        // control: same order recomputed
        let a_c = chunked_dot(&x, &wa, chunk_hit);
        let b_c = chunked_dot(&x, &wb, chunk_hit) + bias;
        if (a_hit >= b_hit) != (a_miss >= b_miss) {
            flips += 1;
        }
        if (a_hit >= b_hit) != (a_c >= b_c) {
            cflips += 1;
        }
        maxd = maxd.max((a_hit - a_miss).abs() as f64).max((b_hit - b_miss).abs() as f64);
        cmaxd = cmaxd.max((a_hit - a_c).abs() as f64).max((b_hit - b_c).abs() as f64);
    }
    let n = positions.max(1) as f64;
    CacheSimResult {
        positions,
        hidden,
        gap_mean,
        chunk_hit,
        chunk_miss,
        argmax_flip: Flip { flips, rate_overall: flips as f64 / n, max_abs_logit_diff: maxd },
        control: Flip { flips: cflips, rate_overall: cflips as f64 / n, max_abs_logit_diff: cmaxd },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_is_exact() {
        let r = simulate(7, 100, 256, 0.05, 16, 48);
        assert_eq!(r.control.flips, 0);
        assert_eq!(r.control.max_abs_logit_diff, 0.0);
    }

    #[test]
    fn orders_differ_and_are_deterministic() {
        let r = simulate(7, 100, 512, 0.05, 16, 48);
        assert!(r.argmax_flip.max_abs_logit_diff > 0.0);
        assert_eq!(r, simulate(7, 100, 512, 0.05, 16, 48));
    }

    #[test]
    fn same_chunking_gives_no_difference() {
        let r = simulate(3, 100, 256, 0.05, 32, 32);
        assert_eq!(r.argmax_flip.flips, 0);
        assert_eq!(r.argmax_flip.max_abs_logit_diff, 0.0);
    }
}
