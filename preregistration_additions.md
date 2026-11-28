## 2026-11-28T00:21:56Z flip-adjudication proposal:50ab53bfed43446fb5162eb325031d5494799d2e8f93466afcb00562f2a4beef

Hypothesis: with 20000 synthetic items, scores uniform on [-1,1], a decision boundary at 0, and perturbation noise (sum of 12 uniforms, scaled to standard deviation 0.02), fewer than 20% of raw label flips are adjudicated genuine under a score-move margin of 0.05, because flips happen almost only for items whose score lies within the perturbation of the boundary.

Metric: /adjudicated_to_raw_flip_ratio in results/751-flip-adjudication.json.

Expected: ratio below 0.20 (roughly 0.05 to 0.10) with a raw flip rate above 0.005.

Falsifier: ratio of 0.20 or more, or a raw flip rate of 0.005 or less.

Run: cargo run --locked -q --release -- flip-sim --seed 7 --items 20000 --noise-sd 0.02 --margin 0.05 --out results/751-flip-adjudication.json

## 2026-11-28T01:19:27Z screen-yield proposal:7fb1dcf00ecd6179a91ea9584fea99f602252e73945fb4f058348df873d18d8e

Hypothesis: with 20000 synthetic items, noise sd 0.02, a true flip rate of 0.002 (true flips move the score by 0.1 to 0.5 across the boundary) and the lowest screening threshold of 0.01 on the score move, fewer than half of flagged candidate flips are true flips (screen_precision between 0.10 and 0.50), and the batch-invariant arm (zero true flips) has screen precision 0 at every threshold.

Metric: /threshold_sweep/0/screen_precision in results/765-screen-yield.json.

Expected: screen_precision at threshold 0.01 below 0.50 (roughly 0.2 to 0.4), rising monotonically with threshold to above 0.90 at threshold 0.10.

Falsifier: screen_precision at the lowest threshold of 0.50 or more, or below 0.10, or a nonzero precision in the batch-invariant arm.

Run: cargo run --locked -q --release -- screen-sim --seed 7 --items 20000 --noise-sd 0.02 --true-flip-rate 0.002 --thresholds 0.01,0.02,0.05,0.10 --out results/765-screen-yield.json

## 2026-11-28T03:14:12Z shape-detection proposal:831819b39f2e3a7ed47c3723cdde24807301591d9e4c47a66d809a022e7d7cb0

Hypothesis: with 2000 simulated bugs per class (tail-mask, off-by-one, block-size), each failing on a known subset of a 256-size shape pool whose regular shapes are the multiples of 64 (no bug fails on a regular shape), boundary-biased sampling of 8 shapes detects at least 0.70 of bugs, uniform random sampling of 8 shapes detects at least 0.10 less than boundary-biased, a fixed regular single shape detects 0.0, and the zero-bug control flags 0.0.

Metric: /detection/boundary_biased_rate_at_8_shapes in results/794-shape-detection.json.

Expected: metric at or above 0.70; fixed regular detection 0.0; uniform random at least 0.10 below boundary-biased at 8 shapes.

Falsifier: boundary-biased detection at 8 shapes below 0.70, or uniform random within 0.10 of boundary-biased at 8 shapes, or any detection by the fixed regular shape or in the zero-bug control.

Run: cargo run --locked -q --release -- shape-sim --seed 7 --bugs-per-class 2000 --pool 256 --counts 1,2,4,8,16 --out results/794-shape-detection.json

## 2026-11-28T04:56:04Z hash-break proposal:89cdd8f47629fbcdc84a22377bf678adb7de1aa76382c4ea6baef8f520b28b8f

Hypothesis: with 5000 synthetic sequences of 64 greedy steps, top-2 logit gaps exponentially distributed with mean 2.0, and per-logit batch-size noise of sd 0.001, a bit-exact logit hash breaks on at least 0.99 of sequences between batch 1 and batch 8, a token-sequence hash breaks on between 0.01 and 0.15, a tolerance hash (max abs logit difference above 0.01) breaks on under 0.01, and the batch-invariant zero-noise control breaks on 0.0 under all three.

Metric: /hash_break/exact/rate_batch_1_vs_8 in results/818-hash-break.json.

Expected: exact rate at least 0.99; token-sequence rate 0.01 to 0.15; tolerance rate below 0.01.

Falsifier: exact rate below 0.99, token-sequence rate outside 0.01 to 0.15, tolerance rate of 0.01 or more, or any break in the zero-noise control.

Run: cargo run --locked -q --release -- hash-sim --seed 7 --sequences 5000 --steps 64 --gap-mean 2.0 --noise-sd 0.001 --tolerance 0.01 --out results/818-hash-break.json

## 2026-11-28T05:53:28Z cache-flip proposal:16a3085730dd1d33fd8dd8ef7622cc16daa22fa8cb53eb345bb88ff1487e7b4c

Hypothesis: with 20000 synthetic positions, hidden size 2048, standard-normal activations and weights, two competing logits whose true gap is exponentially distributed with mean 0.05, and f32 sequential accumulation within chunks of 64 (cache hit) versus chunks of 192 (cache miss), the max abs logit difference between hit and miss is nonzero, the argmax flip rate is below 0.01, and the same-chunking control has 0 flips and 0.0 max abs logit difference.

Metric: /argmax_flip/rate_overall in results/833-cache-flip.json.

Expected: argmax flip rate overall below 0.01 (zero allowed); control flips equal 0; hit-vs-miss max abs logit difference above 0.0.

Falsifier: argmax flip rate of 0.01 or more, any flip or nonzero logit difference in the same-chunking control, or a max abs logit difference of exactly 0.0 between hit and miss orders.

Run: cargo run --locked -q --release -- cache-sim --seed 7 --positions 20000 --hidden 2048 --gap-mean 0.05 --chunk-hit 64 --chunk-miss 192 --out results/833-cache-flip.json
