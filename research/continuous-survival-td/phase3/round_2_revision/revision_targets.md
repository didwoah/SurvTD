# Phase 3 Round 2: Revision Targets for Round 2 Revision

These revision targets were issued from Round 2 Evaluation (`research/continuous-survival-td/phase3/round_2_eval/`) based on the Theoretician's dissent in `critic_report_theorist.md`.

---

### Target 1: Distinguish Empirical Sample Realization from Expected Mixture in Step 5
- **Flaw**: Between observed patient visits $[t_j, t_{j+1})$, the transition is a non-event survival interval by construction ($E_j = 0$). Conflating the expected mixture $\mathcal{T} p_j = (1-\gamma_j)\mu + \gamma_j \Pi \Phi p$ with the sample estimator left $\mu$ undefined on non-event visits.
- **Mandatory Fix**: Explicitly branch the sample update: on surviving visits, evaluate along the continuation branch $(\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$ (which in conditional expectation yields the $\gamma_j$-contractive mixture); on terminal death visits, evaluate at the localized intra-interval projected Dirac $\delta_{\tau_j - t_j}$.

### Target 2: Relocate IPCW Censoring Weighting to Scalar Loss Function in Step 6
- **Flaw**: Weighting target distribution predictions directly by $1/\hat{G}(t_M \mid X) \le 10.0$ inflates total probability mass up to $10.0$, creating an improper defective distribution that violates probability axioms.
- **Mandatory Fix**: Apply inverse probability of censoring weighting $1/\hat{G}(t_M \mid X) \le 10.0$ as a scalar importance sample weight in the squared Cramér loss function $\mathcal{L} = \frac{1}{\hat{G}} \ell_2^2(F_M, F_{G_M})$, preserving exact unit mass ($\sum p_k = 1.0$).

### Target 3: Nest Interval Survival Discount $\gamma_j$ into Multi-Step Continuation Recursion in Step 7
- **Flaw**: The backward recursion $G_j = (1-\lambda_j)\mathcal{T} p_j + \lambda_j \Pi \Phi G_{j+1}$ omitted $\gamma_j$ from the multi-step continuation term, causing survival mortality to un-compound across multi-step chains as $\lambda \to 1$.
- **Mandatory Fix**: Incorporate $\gamma_j$ into the recursive continuation term: $G_j = (1 - \lambda_j) \mathcal{T} p_j + \lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$.
