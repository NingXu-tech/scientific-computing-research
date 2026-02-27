# Meeting Action Items (2-week plan)

This document turns meeting discussion into concrete, reproducible tasks.

## A. Define & document the 3 direction-selection methods (must)
1. Clarify method definitions
   - Method 1: Transition-effect (probability shift)
   - Method 2: LTE (as implemented in code; document exact formula)
   - Method 3: Transition-count based direction
2. Write a short doc for each method
   - Input / Output
   - Core idea + formula (or algorithm steps)
   - Parameters (if any)
   - Failure modes / expected behavior

## B. Implement unified, reproducible pipeline (must)
3. Unify the I/O format
   - Input: shared datasets (no re-generating different random data per method)
   - Discretization: KMeans -> 4 states
   - Build transition counts + transition matrix
   - Output edges in a single long-table CSV:
     - dataset, method, src, dst, score, extra_fields...

4. Benchmark on the same dataset set
   - Synthetic data: 01..05 (v1..v4)
   - Brownian: y1_0, y2_0, y3_0
   - Analytic example: analytic-ex1
   - (Optional) MD data when available

## C. Evaluation & robustness (must)
5. Accuracy metrics
   - TP / FP / FN / TN
   - Precision / Recall / F1
   - Lag-aware version if method produces lag
6. Multi-seed robustness
   - Run seeds = 0..9 (or larger)
   - Report mean ± std

## D. MD pipeline (next stage)
7. MD preprocessing
   - Feature extraction -> KMeans (4 states)
   - Transition matrix + direction methods
8. Apply best method(s) to MD
   - Visualize inferred network
   - Discuss interpretability & stability

## Deliverables checklist (what we will upload)
- docs/methods/direction_methods.md
- results/direction/edges_long.csv
- results/direction/summary.csv
- results/direction/accuracy_per_run.csv
- results/direction/accuracy_mean_std.csv
- (optional) results/direction/figures/*.png
