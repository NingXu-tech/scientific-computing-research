# Do-Statistics Analysis Walkthrough

We have implemented the "Do-Statistics" framework as described in the provided document, specifically focusing on the **Markov Divergence (Option B)** and **Mechanism Shift (Option E)**.

## Implemented Components
1. **Generators (`generators.py`)**: Implemented `DatasetA`, `DatasetB`, `DatasetC`, and `DatasetD` with support for structural replacement interventions.
2. **Statistics (`statistics.py`)**: Implemented:
    - `compute_markov_divergence`: $\Delta JS$ on discretized state transitions.
    - `compute_mechanism_shift_var`: $\Delta LL$ using a Ridge Regression (VAR) probe.
3. **Runner (`run_analysis.py`)**: automated generation, intervention, and scoring.

## Results Summary

We ran the analysis with `max_lag=15` to capture high-order dependencies.

### Dataset A (Linear Chain)
`X -> Y -> Z`
- **Intervene X**: High Shift on X (~33), Low on Y/Z (~0.05). **Correct**.
- **Intervene Y**: High Shift on Y (~45), Low on X/Z. **Correct**.

### Dataset C (High-Order Linear)
Deep lags (up to 15).
- **Intervene X**: High Shift on X (~120). Low on Y/Z (< 0.5).
- **Previous `max_lag=1` result**: Showed significant leakage (~30) into Y/Z.
- **Conclusion**: Increasing the lag depth to match the true process removed leakage, verifying that the **Mechanism Shift statistic implies causal structure** when the model is well-specified.

### Dataset D (Nonlinear)
Nonlinear dependencies ($\sin$, $square$).
- **Intervene X**: High Shift on X (~116). Leakage into Z (~47).
- **Reason**: We used a **Linear** probe model. Since the true mechanism is nonlinear, the best linear fit changes when the input distribution changes.
- **Takeaway**: Nonlinear mechanisms require nonlinear probes (e.g. Kernel Ridge, Neural Net) for strict invariance.

## Artifacts
- **[generators.py](file:///c:/Users/edina/Dropbox/bdudas/causality/antigravity/generators.py)**: Dataset definitions.
- **[statistics.py](file:///c:/Users/edina/Dropbox/bdudas/causality/antigravity/statistics.py)**: Statistical methods.
- **[run_analysis.py](file:///c:/Users/edina/Dropbox/bdudas/causality/antigravity/run_analysis.py)**: Analysis script.
- **[do_statistics_results.csv](file:///c:/Users/edina/Dropbox/bdudas/causality/antigravity/do_statistics_results.csv)**: Detailed outputs.
