# Abstract & Contribution Claims

## Abstract

Dynamic survival analysis from continuous, irregularly sampled longitudinal telemetry is crucial for proactive clinical intervention, yet existing deep frameworks face a structural dilemma. Terminal Monte Carlo negative log-likelihood (NLL) methods suffer from severe trajectory variance and high-frequency alarm jittering, while discrete temporal consistency models enforce uniform unit steps ($\Delta t = 1$) that lead to numerical division explosions ($p / S(\Delta t)$) when extended to irregular continuous intervals. We resolve this dilemma by reformulating continuous-time dynamic survival analysis as policy evaluation in an absorbing Semi-Markov Decision Process (SMDP). Our framework, **SurvTD**, introduces a probability-conserving Survival Bellman Operator that couples continuous elapsed duration $\Delta t$ with a non-expansive categorical Cramér projection ($\Pi$) and backward multi-step geometric $\lambda$-returns, stabilized via Exponential Moving Average (EMA) target network decoupling. Theoretically, we prove that our rightward temporal shift operator conserves total probability mass ($\sum p = 1.0$) and forms a non-expansive mapping under the Cramér Wasserstein-1 metric, eliminating both the $\div S$ divergence of prior consistency methods and the variance of terminal NLL. Across non-linear biophysical ODE dynamics, industrial turbofan degradation (NASA C-MAPSS), and real clinical cohorts subject to heavy right-censoring, SurvTD consistently establishes the superior Pareto frontier: improving dynamic concordance by up to 24.2%p over signature baselines, reducing Integrated Brier Score by 84.4%, and suppressing alarm fatigue by dampening high-frequency prediction jittering. These performance gains strictly depend on non-linear degradation and irregular temporal intervals; in purely linear stationary Markov processes, classical Cox models remain competitive without Bellman bootstrapping.

---

## Falsifiable Contribution Bullets

1. **Probability-Conserving Semi-Markov Bellman Operator (`thm:1`)**:  
   Formulating continuous-time survival temporal difference learning via rightward support shifting ($s + \Delta t$) and categorical projection $\Pi$ preserves total probability mass ($\sum P_{\text{target}} = 1.0$) and guarantees non-expansiveness under the Cramér metric; naive leftward shifting or unprojected continuous shifts violate causality and diverge.
2. **Elimination of Numerical Division Explosion (`fig:1`)**:  
   Replacing TCSR's downstream survival division ($p / S(\Delta t)$) with forward contractive renewal multiplication ($S(\Delta t) \times$) maintains bounded gradients across arbitrary observation gaps $\Delta t > 0$, eliminating the numerical instability that collapses prior discrete consistency models.
3. **Empirical Bias-Variance Inverted-U Frontier (`fig:2`)**:  
   Sweeping the geometric return mixture $\lambda \in [0.0, 1.0]$ produces Sutton & Barto's classic inverted-U discrimination curve, peaking at $\lambda \in [0.6, 0.8]$ (C-index $> 0.99$, IBS $< 0.10$) and crashing at $\lambda = 1.0$ (pure Monte Carlo NLL, C-index $= 0.3474$) due to unmitigated trajectory variance.
4. **Clinical Calibration & Alarm Stability (`tab:1`, `tab:2`)**:  
   On real right-censored cohorts and physiological ODEs, SurvTD improves time-dependent C-index by up to 24.2%p over CoxSig and reduces Integrated Brier Score by 84.4%, while significantly reducing spurious alarm jittering (FAR per bed-day) without sacrificing early warning lead time.
