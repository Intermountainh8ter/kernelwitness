# kernelwitness pre-registration

Locked before the first live-benchmark result is committed.
Append-only.

## 1. Scope

Kernels under benchmark are a fixed set of ten canonical operations
with shape-space knobs. Each benchmark run is tied to a specific
`Rig` fingerprint. A report on rig X does not claim anything about rig
Y; cross-rig comparisons are explicit `diff` outputs.

## 2. The ten kernels

| id | op | dtype | shape-space |
|----|----|-------|-------------|
| matmul-bf16 | GEMM | bf16 | (m, n, k) |
| matmul-fp16 | GEMM | fp16 | (m, n, k) |
| matmul-fp8  | GEMM | fp8  | (m, n, k) |
| sdpa        | scaled dot-product attention | bf16/fp16 | (b, h, s, d) |
| rmsnorm     | RMSNorm forward | bf16/fp16 | (b, s, h) |
| swiglu_mlp  | gate-up-down MLP | bf16/fp16 | (b, s, h, hd) |
| fa_bwd      | flash-attn backward | bf16/fp16 | (b, h, s, d) |
| ln_bwd      | LayerNorm backward | bf16/fp16 | (b, s, h) |
| emb_lookup  | embedding gather | int32→bf16 | (v, h, b, s) |
| gemm_i8     | int8 gemm | int8 | (m, n, k) |

Shape sets are committed in `fixtures/shapes.json`; a new shape is a
new row.

## 3. Measurement protocol

For each kernel + shape + rig:

- `N` trials, default 50.
- Warm-up: `N_warm = 5` runs excluded from the sample.
- Each trial measures GPU elapsed via vendor-standard timers
  (CUDA events or HIP events).
- Report: mean, stdev, p50, p95, p99 in microseconds.
- Trials sort for percentiles; no outlier removal.

## 4. The reproducibility claim

A benchmark number is **reproducible on rig R** when a second run of
the same kernel + shape on rig R produces a mean within `3 * stdev` of
the first run's mean, with every field of `rig` identical.

The repo refuses to call a cross-rig comparison a reproduction. The
`diff` command's `rig_match` flag is the honest answer.

## 5. What this repo refuses to do

- No MLPerf-style regimes that measure end-to-end training throughput.
  That is a different benchmark.
- No "equivalent kernel" substitution. The ten kernels are the ten
  kernels.
- No results in README that are not produced from a committed
  `results/*.json`.

## 6. Threats to validity

- **Vendor-timer precision.** CUDA events have microsecond-ish
  precision; HIP events can be noisier. We report the timer source.
- **Thermal state.** Hot GPUs run slower; the warm-up includes
  temperature stabilization but a laptop GPU without active cooling
  will still drift. Each report ships the entry temperature.
- **Driver scheduling.** Some runtimes batch kernel launches; the
  measured wall time can include launch queueing. We report the
  launch-queue depth where the runtime exposes it.
