//! Core statistics for kernel benchmark reports.
//!
//! A single kernel run is a `Trial` with its elapsed microseconds. A
//! `Report` aggregates N trials into mean, stdev, p50/p95/p99, plus
//! the rig fingerprint so reports from different machines can be
//! compared.

use serde::{Deserialize, Serialize};

/// What machine the trial was run on. Reproducing the trial requires
/// matching every field here.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rig {
    pub gpu: String,
    pub driver: String,
    pub runtime: String, // ROCm 7.2.3, CUDA 12.4, etc
    pub torch: String,
    pub os: String,
}

/// One trial's measurement.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Trial {
    pub elapsed_us: f64,
}

/// A full report: mean, stdev, percentiles, plus the rig fingerprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub rig: Rig,
    pub kernel: String,
    pub shape: String,
    pub n: usize,
    pub mean_us: f64,
    pub stdev_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty());
    let idx = (sorted.len() as f64 * q).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn aggregate(rig: Rig, kernel: &str, shape: &str, trials: &[Trial]) -> Report {
    aggregate_with_warmup(rig, kernel, shape, trials, 0)
}

/// Aggregate `trials` after dropping the first `n_warm` entries, per
/// pre-reg §3's N_warm = 5 default. The scaffold's `aggregate` kept
/// every trial; this is the primitive callers should prefer when the
/// trials come from a live benchmark driver that captures the warm-up.
///
/// Panics if fewer than `n_warm + 1` trials are supplied.
pub fn aggregate_with_warmup(
    rig: Rig,
    kernel: &str,
    shape: &str,
    trials: &[Trial],
    n_warm: usize,
) -> Report {
    assert!(
        trials.len() > n_warm,
        "aggregate_with_warmup requires at least {} trials (got {})",
        n_warm + 1,
        trials.len()
    );
    let sample = &trials[n_warm..];
    let n = sample.len();
    let mut xs: Vec<f64> = sample.iter().map(|t| t.elapsed_us).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stdev = var.sqrt();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Report {
        rig,
        kernel: kernel.into(),
        shape: shape.into(),
        n,
        mean_us: mean,
        stdev_us: stdev,
        p50_us: percentile(&xs, 0.5),
        p95_us: percentile(&xs, 0.95),
        p99_us: percentile(&xs, 0.99),
    }
}

/// Difference between two reports. Positive `delta_mean_us` means `b`
/// was slower than `a` by that many microseconds. The rig_match flag
/// is false when any rig field differs; the number is still useful but
/// the comparison is cross-rig.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diff {
    pub rig_match: bool,
    pub delta_mean_us: f64,
    pub delta_pct: f64,
}

pub fn diff(a: &Report, b: &Report) -> Diff {
    let delta = b.mean_us - a.mean_us;
    let pct = if a.mean_us > 0.0 {
        delta / a.mean_us * 100.0
    } else {
        f64::NAN
    };
    Diff {
        rig_match: a.rig == b.rig,
        delta_mean_us: delta,
        delta_pct: pct,
    }
}

/// Pre-reg §4 reproducibility verdict. Rig fingerprints must match
/// AND `|b.mean - a.mean| <= 3 * a.stdev`. The repo refuses to call a
/// cross-rig comparison a reproduction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReproducibilityReport {
    pub rig_match: bool,
    pub delta_mean_us: f64,
    /// `3 * a.stdev_us`, the pre-reg §4 bound.
    pub three_sigma_bound_us: f64,
    /// `delta_mean_us.abs() <= three_sigma_bound_us`.
    pub within_bound: bool,
    /// `rig_match && within_bound`. The only honest "reproduced" verdict.
    pub reproduced: bool,
}

/// Compute the reproducibility verdict for a candidate `b` against a
/// baseline `a`. The baseline's stdev defines the three-sigma band.
pub fn reproducibility(baseline: &Report, candidate: &Report) -> ReproducibilityReport {
    let delta = candidate.mean_us - baseline.mean_us;
    let bound = 3.0 * baseline.stdev_us;
    let rig_match = baseline.rig == candidate.rig;
    let within_bound = delta.abs() <= bound;
    ReproducibilityReport {
        rig_match,
        delta_mean_us: delta,
        three_sigma_bound_us: bound,
        within_bound,
        reproduced: rig_match && within_bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> Rig {
        Rig {
            gpu: "gfx906".into(),
            driver: "26.1".into(),
            runtime: "ROCm 7.2.3".into(),
            torch: "2.4.1+rocm6.0".into(),
            os: "Ubuntu 24.04".into(),
        }
    }

    #[test]
    fn aggregate_produces_sensible_percentiles() {
        let trials: Vec<Trial> = (1..=100)
            .map(|i| Trial {
                elapsed_us: i as f64,
            })
            .collect();
        let r = aggregate(rig(), "matmul", "mnk=1024", &trials);
        assert_eq!(r.n, 100);
        assert_eq!(r.p50_us, 51.0);
        assert_eq!(r.p95_us, 96.0);
        assert_eq!(r.p99_us, 100.0);
    }

    #[test]
    fn diff_rig_match_flag_is_correct() {
        let r1 = aggregate(rig(), "k", "s", &[Trial { elapsed_us: 100.0 }]);
        let mut r2_rig = rig();
        r2_rig.driver = "26.2".into();
        let r2 = aggregate(r2_rig, "k", "s", &[Trial { elapsed_us: 120.0 }]);
        let d = diff(&r1, &r2);
        assert!(!d.rig_match);
        assert!((d.delta_mean_us - 20.0).abs() < 1e-12);
        assert!((d.delta_pct - 20.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_with_warmup_discards_the_first_n_warm_trials() {
        // 5 warm-up trials at 1000us, then 10 real trials at 100us.
        let mut trials: Vec<Trial> =
            (0..5).map(|_| Trial { elapsed_us: 1000.0 }).collect();
        trials.extend((0..10).map(|_| Trial { elapsed_us: 100.0 }));
        let r = aggregate_with_warmup(rig(), "matmul", "mnk=1024", &trials, 5);
        assert_eq!(r.n, 10);
        assert!((r.mean_us - 100.0).abs() < 1e-12);
        assert!(r.stdev_us.abs() < 1e-12);
    }

    #[test]
    fn aggregate_with_warmup_panics_when_everything_is_warmup() {
        let trials: Vec<Trial> = (0..5).map(|_| Trial { elapsed_us: 1.0 }).collect();
        let result = std::panic::catch_unwind(|| {
            aggregate_with_warmup(rig(), "k", "s", &trials, 5)
        });
        assert!(result.is_err());
    }

    fn report_mean_stdev(mean: f64, stdev: f64) -> Report {
        Report {
            rig: rig(),
            kernel: "matmul-bf16".into(),
            shape: "mnk=1024".into(),
            n: 50,
            mean_us: mean,
            stdev_us: stdev,
            p50_us: mean,
            p95_us: mean,
            p99_us: mean,
        }
    }

    #[test]
    fn reproducibility_passes_within_three_sigma_on_same_rig() {
        let a = report_mean_stdev(100.0, 5.0);
        let b = report_mean_stdev(110.0, 5.0); // delta=10, bound=15
        let r = reproducibility(&a, &b);
        assert!(r.rig_match);
        assert!(r.within_bound);
        assert!(r.reproduced);
    }

    #[test]
    fn reproducibility_fails_outside_three_sigma_on_same_rig() {
        let a = report_mean_stdev(100.0, 1.0);
        let b = report_mean_stdev(110.0, 1.0); // delta=10, bound=3
        let r = reproducibility(&a, &b);
        assert!(r.rig_match);
        assert!(!r.within_bound);
        assert!(!r.reproduced);
    }

    #[test]
    fn reproducibility_never_true_cross_rig_even_if_within_bound() {
        let mut a = report_mean_stdev(100.0, 50.0);
        let b = report_mean_stdev(102.0, 50.0);
        a.rig.driver = "26.1".into();
        let mut b = b;
        b.rig.driver = "26.2".into();
        let r = reproducibility(&a, &b);
        assert!(!r.rig_match);
        assert!(r.within_bound);
        assert!(!r.reproduced, "pre-reg §4: cross-rig is never reproduced");
    }
}
