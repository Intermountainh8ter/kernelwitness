//! Fixture importers for external kernel-benchmark output.
//!
//! A real GPU rig is required to produce the raw measurements, so the
//! "live adapter" here is a JSON importer. It reads a canonical result
//! file written by whichever benchmark harness ran on the rig
//! (`torch.utils.benchmark.Timer.timeit`, nvbench, hipBench, Triton's
//! `do_bench`, or kernelwitness's own canonical shape) and lifts it to
//! a `Report` via `aggregate` / `aggregate_with_warmup`.
//!
//! Two shapes are supported today:
//!
//! * **Canonical kernelwitness** — `{"rig": {...}, "kernel": "...",
//!   "shape": "...", "trials_us": [1.2, 1.5, ...], "warmup": 0}`.
//! * **PyTorch Timer** — `{"gpu": "...", "driver": "...",
//!   "runtime": "...", "torch": "...", "os": "...", "kernel": "...",
//!   "shape": "...", "times": [1.2e-6, 1.5e-6, ...]}` with times in
//!   seconds (as `Timer.timeit().raw_times` yields).

use crate::{aggregate_with_warmup, is_canonical_kernel, Report, Rig, Trial};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("JSON could not be parsed: {0}")]
    MalformedJson(String),
    #[error("trials array is empty")]
    EmptyTrials,
    #[error("trial at index {index} is not finite: {value}")]
    NotFinite { index: usize, value: f64 },
    #[error("trial at index {index} is negative: {value}")]
    Negative { index: usize, value: f64 },
    #[error("warmup={warmup} is not smaller than trials.len={len}")]
    WarmupExceedsTrials { warmup: usize, len: usize },
    #[error("kernel `{0}` is not in the canonical set (see is_canonical_kernel)")]
    NonCanonicalKernel(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelwitnessJson {
    pub rig: Rig,
    pub kernel: String,
    pub shape: String,
    pub trials_us: Vec<f64>,
    #[serde(default)]
    pub warmup: usize,
}

/// Lift a canonical kernelwitness JSON file to a `Report`. Pass
/// `allow_noncanonical=true` to override the kernel-name gate (useful
/// for one-off exploratory kernels; the pre-reg's leaderboard still
/// refuses them).
pub fn import_kernelwitness_json(
    json: &str,
    allow_noncanonical: bool,
) -> Result<Report, ImportError> {
    let doc: KernelwitnessJson =
        serde_json::from_str(json).map_err(|e| ImportError::MalformedJson(e.to_string()))?;
    if !allow_noncanonical && !is_canonical_kernel(&doc.kernel) {
        return Err(ImportError::NonCanonicalKernel(doc.kernel));
    }
    if doc.trials_us.is_empty() {
        return Err(ImportError::EmptyTrials);
    }
    if doc.warmup >= doc.trials_us.len() {
        return Err(ImportError::WarmupExceedsTrials {
            warmup: doc.warmup,
            len: doc.trials_us.len(),
        });
    }
    let trials = validate_and_wrap_us(&doc.trials_us)?;
    Ok(aggregate_with_warmup(
        doc.rig,
        &doc.kernel,
        &doc.shape,
        &trials,
        doc.warmup,
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorchBenchmarkJson {
    pub gpu: String,
    pub driver: String,
    pub runtime: String,
    pub torch: String,
    pub os: String,
    pub kernel: String,
    pub shape: String,
    /// Times in SECONDS, as `Timer.timeit().raw_times` returns.
    pub times: Vec<f64>,
    #[serde(default)]
    pub warmup: usize,
}

pub fn import_torch_benchmark_json(
    json: &str,
    allow_noncanonical: bool,
) -> Result<Report, ImportError> {
    let doc: TorchBenchmarkJson =
        serde_json::from_str(json).map_err(|e| ImportError::MalformedJson(e.to_string()))?;
    if !allow_noncanonical && !is_canonical_kernel(&doc.kernel) {
        return Err(ImportError::NonCanonicalKernel(doc.kernel));
    }
    if doc.times.is_empty() {
        return Err(ImportError::EmptyTrials);
    }
    if doc.warmup >= doc.times.len() {
        return Err(ImportError::WarmupExceedsTrials {
            warmup: doc.warmup,
            len: doc.times.len(),
        });
    }
    // Seconds → microseconds.
    let times_us: Vec<f64> = doc.times.iter().map(|t| t * 1_000_000.0).collect();
    let trials = validate_and_wrap_us(&times_us)?;
    let rig = Rig {
        gpu: doc.gpu,
        driver: doc.driver,
        runtime: doc.runtime,
        torch: doc.torch,
        os: doc.os,
    };
    Ok(aggregate_with_warmup(
        rig,
        &doc.kernel,
        &doc.shape,
        &trials,
        doc.warmup,
    ))
}

fn validate_and_wrap_us(values: &[f64]) -> Result<Vec<Trial>, ImportError> {
    let mut out = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(ImportError::NotFinite { index: i, value: v });
        }
        if v < 0.0 {
            return Err(ImportError::Negative { index: i, value: v });
        }
        out.push(Trial { elapsed_us: v });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig_fragment() -> &'static str {
        r#""rig": {"gpu":"MI50","driver":"6.3.0","runtime":"ROCm 7.2.3","torch":"2.4.0","os":"Ubuntu 22.04"}"#
    }

    #[test]
    fn canonical_kw_json_imports_cleanly() {
        let kernel = {
            // Use one from the canonical set. We don't know the exact
            // list here, so cheat by asking the library.
            let candidates = [
                "fp16_matmul",
                "fp8_matmul",
                "flash_attn_fwd",
                "softmax",
                "rmsnorm",
                "layernorm",
                "rope",
                "fp16_gemv",
                "paged_attn",
                "all_reduce",
            ];
            candidates
                .iter()
                .find(|k| is_canonical_kernel(k))
                .expect("one canonical kernel must exist")
                .to_string()
        };
        let json = format!(
            r#"{{{rig},"kernel":"{kernel}","shape":"m=1024,n=1024,k=1024","trials_us":[100.0,101.0,99.5,100.5,100.2],"warmup":1}}"#,
            rig = rig_fragment(),
        );
        let r = import_kernelwitness_json(&json, false).expect("ok");
        assert_eq!(r.n, 4); // warmup=1 skipped
        assert_eq!(r.kernel, kernel);
    }

    #[test]
    fn noncanonical_kernel_is_rejected_by_default() {
        let json = format!(
            r#"{{{rig},"kernel":"experiment_only","shape":"x","trials_us":[1.0,2.0],"warmup":0}}"#,
            rig = rig_fragment()
        );
        match import_kernelwitness_json(&json, false) {
            Err(ImportError::NonCanonicalKernel(k)) => assert_eq!(k, "experiment_only"),
            o => panic!("wrong variant: {o:?}"),
        }
        // Opt-in allows it.
        let r = import_kernelwitness_json(&json, true).expect("allowed");
        assert_eq!(r.n, 2);
    }

    #[test]
    fn empty_trials_rejected() {
        let json = format!(
            r#"{{{rig},"kernel":"exp","shape":"x","trials_us":[],"warmup":0}}"#,
            rig = rig_fragment()
        );
        match import_kernelwitness_json(&json, true) {
            Err(ImportError::EmptyTrials) => {}
            o => panic!("wrong variant: {o:?}"),
        }
    }

    #[test]
    fn negative_trial_rejected_with_index() {
        let json = format!(
            r#"{{{rig},"kernel":"exp","shape":"x","trials_us":[1.0,-0.5,2.0],"warmup":0}}"#,
            rig = rig_fragment()
        );
        match import_kernelwitness_json(&json, true) {
            Err(ImportError::Negative { index, value }) => {
                assert_eq!(index, 1);
                assert_eq!(value, -0.5);
            }
            o => panic!("wrong variant: {o:?}"),
        }
    }

    #[test]
    fn warmup_at_or_above_len_rejected() {
        let json = format!(
            r#"{{{rig},"kernel":"exp","shape":"x","trials_us":[1.0,2.0],"warmup":2}}"#,
            rig = rig_fragment()
        );
        match import_kernelwitness_json(&json, true) {
            Err(ImportError::WarmupExceedsTrials { warmup, len }) => {
                assert_eq!(warmup, 2);
                assert_eq!(len, 2);
            }
            o => panic!("wrong variant: {o:?}"),
        }
    }

    #[test]
    fn torch_benchmark_json_converts_seconds_to_microseconds() {
        let json = r#"{
            "gpu":"MI50","driver":"6.3.0","runtime":"ROCm 7.2.3","torch":"2.4.0","os":"Ubuntu 22.04",
            "kernel":"exp","shape":"x",
            "times":[1.0e-6, 2.0e-6, 3.0e-6],
            "warmup":0
        }"#;
        let r = import_torch_benchmark_json(json, true).expect("ok");
        assert_eq!(r.n, 3);
        // Mean should be 2 microseconds.
        assert!((r.mean_us - 2.0).abs() < 1e-6);
    }

    #[test]
    fn torch_benchmark_json_rejects_malformed_json() {
        match import_torch_benchmark_json("not json", true) {
            Err(ImportError::MalformedJson(_)) => {}
            o => panic!("wrong variant: {o:?}"),
        }
    }
}
