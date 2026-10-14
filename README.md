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
- `aggregate(rig, kernel, shape, trials)` producing mean / stdev /
  p50 / p95 / p99.
- `diff(a, b)` with `rig_match` flag and per-mean delta.
- `kernelwitness aggregate` and `kernelwitness compare` CLI.

Not in this commit (openly tracked):

- Ten canonical kernels: matmul (bf16/fp16/fp8), SDPA, RMS-norm,
  SwiGLU MLP, flash-attn backward, LayerNorm backward, embedding
  lookup, int8 gemm.
- `kernelwitness bench --kernel <name> --shape <mnk>` that runs the
  N trials with warm-up excluded.
- Rig fingerprint auto-collector (reads nvidia-smi / rocm-smi + torch
  module versions).

## License

MIT. See `LICENSE`.
