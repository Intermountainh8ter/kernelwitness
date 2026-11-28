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
