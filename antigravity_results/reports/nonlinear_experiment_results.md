# Nonlinear Probe Experiment (Dataset D)

We implemented a **Kernel Ridge Regression (RBF Kernel)** probe to test if it could account for the nonlinearity in Dataset D ($Y^2$ term) and reduce the "leakage" observed with the Linear probe.

### Hypothesis
A linear probe fails to model $Z \sim Y^2$ invariantly. If the intervention shifts the distribution of $Y$, the best linear fit changes, causing a high "Mechanism Shift" score ($\Delta LL$) even though the mechanism is physically invariant. An RBF Kernel probe should fit the surface $Y^2$ better and remain invariant.

### Result
*   **Linear Probe Leakage**: ~47
*   **Kernel Probe Leakage**: ~272 (Significantly Worse)

### Interpretation: The Extrapolation Problem
The intervention "Clams X to 2.0".
*   In the baseline dataset, variables fluctuate around 0 with standard deviation $\approx 0.15$.
*   The intervention value $X=2.0$ is $>10$ standard deviations away from the mean.
*   This pushes the system into a region of state-space completely unseen during training.
*   **Linear Models** extrapolate linearly (which is often a "reasonable" guess for physics, or at least consistent with an overall trend).
*   **RBF Kernels** rely on local similarity. When far from any training point, the prediction collapses (often to 0) or becomes unreliable. This result in a massive prediction error on the interventional data, leading to a huge "Mechanism Shift" score.

### Conclusion
Nonlinear invariance testing is powerful but **fragile to out-of-distribution (OOD) interventions**. To use nonlinear probes effectively:
1.  Use **gentler interventions** (e.g., clamp to $1\sigma$, not $10\sigma$) so the system stays within the support of the training data.
2.  Use **Polynomial Kernels** or Neural Networks with appropriate inductive biases that extrapolate better than RBF.
