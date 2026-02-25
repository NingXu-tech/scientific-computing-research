# detailed_results_interpretation.md

## Overview of the Analysis

We performed an interventional analysis on four synthetic datasets (A, B, C, D) using two primary "do-statistics":

1.  **DeltaJS ($\Delta JS$)**: Markov Divergence. This measures how much the *global dynamics* of the system change when we intervene on a variable. A high value means the system behaves very differently under intervention.
2.  **Mechanism Shift ($\Delta LL$)**: Likelihood Ratio / Generalized Mechanism Shift. This measures how much the *conditional probability mechanism* of a specific variable changes.
    *   **Logic**: If we intervene on variable $X_i$, its mechanism changes completely (we forced it). So $\Delta LL$ for $X_i$ should be very high.
    *   **Invariance**: If $X_j$ is a different variable, its mechanism (how it reacts to its parents) should *not* change, even if the values of its parents change. So $\Delta LL$ for $X_j$ should be low (ideally 0).
    *   **Leakage**: If $\Delta LL$ is high for a non-intervened variable, it suggests our model of that variable is "misspecified" (e.g., we used a linear model for a nonlinear system, or missed a lag).

## Interpreting the "do_statistics_results.csv" File

The CSV file contains one row per seed and intervention target. Key columns:

*   **dataset**: Name of the model (A, B, C, or D).
*   **intervention_target**: Which variable we "clamped" (0=X, 1=Y, 2=Z).
*   **DeltaJS**: Global change score.
*   **DLL_var0, DLL_var1, DLL_var2**: The mechanism shift score for variables X, Y, and Z respectively.

---

## Detailed Findings

### Dataset A: Linear Lag Chain ($X \to Y \to Z$)
*   **Intervention on X (Target 0)**:
    *   `DLL_var0` is High (~33). Correct (we broke X).
    *   `DLL_var1` is Low (~0.07). Correct (Y's mechanism $Y_t = 0.6Y_{t-1} + 0.5X_{t-1}$ is invariant).
    *   `DLL_var2` is Low (~0.01). Correct.
*   **Intervention on Y (Target 1)**:
    *   `DLL_var1` is High (~45). Correct.
    *   `DLL_var0` and `DLL_var2` are Low. Correct.
*   **Conclusion**: The statistics perfectly identify the causal structure. The "Mechanism Shift" isolates the intervened variable without leaking to others.

### Dataset B: Linear Multivariate
*   **Results**: Similar to Dataset A. Intervening on any variable yields a massive shift in its own mechanism score (>30) and negligible shifts (<0.3) in others.
*   **Conclusion**: Highly robust identification of linear causal mechanisms.

### Dataset C: High-Order Lags (Linear)
*   **Context**: This dataset has dependencies up to 15 steps back (e.g., $Z_t$ depends on $X_{t-15}$).
*   **Results**:
    *   Initially (with lag=1), we saw "leakage" (scores of ~30 on non-intervened vars).
    *   **Current Run (lag=15)**: `DLL_var0` (~120) vs `DLL_var1` (~0.4).
*   **Conclusion**: By increasing the analysis lag to 15, we correctly captured the long-term dependencies. The low cross-variable scores confirm that the method works for long-memory linear processes *if* the analysis window is sufficient.

### Dataset D: Nonlinear High-Order
*   **Context**: Variables interact via sine waves and squares (e.g., $Y_t \sim \sin(X_{t-3})$). We analyzed this using a **Linear** probe model.
*   **Intervention on X (Target 0)**:
    *   `DLL_var0` (X): ~116 (High, Correct).
    *   `DLL_var1` (Y): ~4.2 (Low-ish, but non-zero).
    *   `DLL_var2` (Z): ~47 (High - LEAKAGE).
*   **Interpretation**:
    *   Variable Z depends on $Y^2$. When we intervene on X, we change the range of values Y takes, which pushes Z into a different region of the parabola $Y^2$.
    *   A **Linear Model** cannot fit a parabola globally. It fits a local line. When the data distribution moves, the "best linear fit" changes.
    *   **Do-Statistic Sensitivity**: The fact that `DLL_var2` is high tells us that **our linear model for Z is wrong** (it is not invariant). This is a useful diagnostic! It flags nonlinearity or missing finding.

---

## Summary Recommendation
1.  **For Linear Systems**: The $\Delta LL$ statistic is precise. It pinpoints exactly which variable was intervened on and confirms structural invariance for others.
2.  **For Nonlinear Systems**: A Linear $\Delta LL$ statistic will show "leakage". If you see high mechanism shifts on variables you *didn't* touch, it implies the system is nonlinear (or the model is too simple).
3.  **Next Steps**: To fix the leakage in Dataset D, we would need to replace the Ridge Regression in our scanner with a Non-linear regressor (e.g., Gaussian Process or Neural Network).
