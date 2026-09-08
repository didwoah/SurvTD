# Multi-step $\lambda$-Returns & Effective Horizon Invariance

The learner mastered how duration-geometric discounting $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ overcomes DeepTCSR's sampling-frequency trap (step-geometric count $\lambda^k$), guaranteeing that the effective bootstrapping horizon (half-life $t_{1/2}$) remains strictly invariant on the continuous physical time axis regardless of irregular observation density.

## Evidence
- `notebooks/04_lambda_returns.ipynb` was created, executed, and baked with visual and mathematical verification of two patients (1-hour vs 4-hour sampling rates) achieving identical weight decay on the physical time axis (`assert np.isclose` verified).
- `lessons/0004-multi-step-lambda-returns-and-horizon-invariance.html` was published and opened in the browser.

## Implications
- Directly forms **Figure 2 ("Sampling Rate Invariance of the Effective Bootstrapping Horizon")** in Method §3.3.
- Underpins Negative Control **NC-C** (Table 3 ablation: Count-geometric vs Duration-geometric), demonstrating the empirical necessity of continuous time-scaling on MIMIC-IV and PhysioNet benchmarks.
