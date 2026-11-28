## 2026-11-28T00:21:56Z flip-adjudication proposal:50ab53bfed43446fb5162eb325031d5494799d2e8f93466afcb00562f2a4beef

Hypothesis: with 20000 synthetic items, scores uniform on [-1,1], a decision boundary at 0, and perturbation noise (sum of 12 uniforms, scaled to standard deviation 0.02), fewer than 20% of raw label flips are adjudicated genuine under a score-move margin of 0.05, because flips happen almost only for items whose score lies within the perturbation of the boundary.

Metric: /adjudicated_to_raw_flip_ratio in results/751-flip-adjudication.json.

Expected: ratio below 0.20 (roughly 0.05 to 0.10) with a raw flip rate above 0.005.

Falsifier: ratio of 0.20 or more, or a raw flip rate of 0.005 or less.

Run: cargo run --locked -q --release -- flip-sim --seed 7 --items 20000 --noise-sd 0.02 --margin 0.05 --out results/751-flip-adjudication.json
