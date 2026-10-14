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
    assert!(!trials.is_empty(), "aggregate requires at least one trial");
    let n = trials.len();
    let mut xs: Vec<f64> = trials.iter().map(|t| t.elapsed_us).collect();
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
}
