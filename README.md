# kernelwitness

**Deterministic, hardware-labeled GPU kernel benchmarks.** For the ten
kernels that actually matter in LLM inference and training, capture
`(hardware, driver, runtime, torch, OS, kernel, shape, FLOPs, bandwidth,
elapsed_us, stdev)` tuples so a published number is reproducible on a
specific rig and attributable across rigs.

![status: initial scaffold](https://img.shields.io/badge/status-initial%20scaffold-lightgrey)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why

MLPerf is coarse-grained and expensive. Private kernel benchmarks exist
(NVIDIA's own, HPL, HPCG, triton-bench) but none capture *reproducibility
across the exact rig*. gfx906 experience shows how fragile ROCm kernel
performance is across driver versions — the published FLOPs number
cannot be reproduced on a nominally identical rig when the ROCm patch
version differs.

## What is in this scaffold

- `Rig`, `Trial`, `Report` types with full rig fingerprint.
- `aggregate(rig, kernel, shape, trials)` and
  `aggregate_with_warmup(rig, kernel, shape, trials, n_warm)` producing
  mean / stdev / p50 / p95 / p99 — pre-reg §3's N_warm=5 default is
  wired through the CLI's `--warmup`.
- `diff(a, b)` with `rig_match` flag and per-mean delta.
- `reproducibility(baseline, candidate) -> ReproducibilityReport`:
  pre-reg §4's "same rig AND |delta_mean| ≤ 3·stdev" verdict. The repo
  refuses to call a cross-rig comparison a reproduction; the
  `reproduced` field reflects that explicitly.
- CLI: `kernelwitness aggregate`, `compare`, `reproducibility`.
  Exit codes on `reproducibility`: 0 reproduced, 1 within-bound but
  cross-rig, 2 out of 3-sigma band.
- CLI: `kernelwitness flip-sim`, a model-free seeded simulation of raw
  label flips against flips adjudicated genuine by a score-move margin.
  Result `results/751-flip-adjudication.json`: 152 raw flips of 20000
  items (raw_flip_rate 0.0076), 8 adjudicated genuine, so
  adjudicated_to_raw_flip_ratio 0.05263157894736842 (hypothesis: below
  0.20, met). Synthetic data only; it says nothing about real kernels.
- CLI: `kernelwitness screen-sim`, a seeded sweep of screening thresholds
  over mixed true and numerical-noise flips. Result
  `results/765-screen-yield.json`: at threshold 0.01, 178 flagged
  candidates of which 35 true, screen_precision 0.19662921348314608
  (hypothesis: between 0.10 and 0.50, met); the batch-invariant arm has
  screen_precision 0.0 at every threshold. Synthetic data only.
- CLI: `kernelwitness shape-sim`, a seeded sweep of how many shapes a suite
  samples and how it samples them, against injected tail-mask, off-by-one
  and block-size bugs. Result `results/794-shape-detection.json`: at 8
  shapes boundary_biased_rate_at_8_shapes is 0.8386666666666667 and
  uniform_rate_at_8_shapes is 0.38333333333333336; the fixed regular shape
  detects 0.0 and the zero-bug control_rate is 0.0 (hypothesis: boundary
  at or above 0.70, uniform at least 0.10 lower, met). Synthetic data only.
- CLI: `kernelwitness hash-sim`, a seeded simulation of how often a
  bit-exact logit hash, a token-sequence hash and a tolerance hash break
  between batch 1 and batch 8. Result `results/818-hash-break.json`: exact
  rate 1.0, token rate 0.024, tolerance rate 0.0, and 0.0 for all three in
  the zero-noise control (hypothesis: exact at least 0.99, token 0.01 to
  0.15, tolerance below 0.01, met). Synthetic data only.

Not in this commit (openly tracked):

- Ten canonical kernels: matmul (bf16/fp16/fp8), SDPA, RMS-norm,
  SwiGLU MLP, flash-attn backward, LayerNorm backward, embedding
  lookup, int8 gemm.
- `kernelwitness bench --kernel <name> --shape <mnk>` that runs the
  N trials and emits a Trials JSON directly.
- Rig fingerprint auto-collector (reads nvidia-smi / rocm-smi + torch
  module versions).

## License

MIT. See `LICENSE`.
