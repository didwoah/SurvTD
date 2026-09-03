# Research Note: Weight Initialization Strategies for Continuous Survival TD Models

**Author / Contributor**: Yang Jae-mo (didwoah) & Antigravity Research Agent  
**Date**: 2026-09-04  
**Status**: Proposal for Follow-up Ablation & Discussion Section  
**Target Modules**: `src/models/hazard_head.py`, `src/models/survtd.py`, `src/models/baselines/deeptcsr_clamped.py`

---

## 1. Problem Diagnosis: The "Early Death Explosion" of Random Initialization

Standard deep learning frameworks initialize final linear layer weights $W \sim \mathcal{N}(0, \sigma^2)$ and biases $b = 0$.
In discrete-time hazard formulation with hazard head logits $z_k$:
$$h_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

When $z_k \approx 0$ at step 0:
1. **Per-bin hazard explodes**: $h_k \approx 0.5$ (50% mortality rate in every single time bin).
2. **Survival curve plummets**:
   $$S(k) = \prod_{m=0}^k (1 - h_m) \approx (0.5)^{k+1}$$
   Across $K = 36$ bins, the predicted survival probability at the horizon is:
   $$S(36) \approx (0.5)^{36} \approx 1.45 \times 10^{-11}$$
3. **Catastrophic Failure in DeepTCSR**:
   - DeepTCSR relies on pseudo-target division $p_{j+1} / \max(S(\Delta t_j), 10^{-3})$.
   - Under $S(\Delta t_j) \approx 0$, the denominator collapses immediately into the clamp floor ($10^{-3}$) on the very first batch.
   - The backward gradients explode or saturate, trapping the model in a degenerate sub-chance state ($C^{td} < 0.50$, as observed in Seed 123 where DeepTCSR collapsed to $0.4497$).
4. **SurvTD's Cold-Start Burden**:
   - SurvTD avoids division by zero via renewal mixture $((1-\gamma_j) + \gamma_j \Pi \Phi p_{j+1})$.
   - However, with random initialization, $\gamma_j = S(\Delta t_j) \approx 0$, forcing the model to place almost all probability mass on the near-interval branch $[0, \Delta t_j)$ during early epochs, slowing down convergence and increasing seed variance.

---

## 2. Three Proposed Initialization Strategies

### Strategy 1: Marginal Kaplan-Meier Prior Bias Initialization (Recommended)

#### Conceptual Rationale:
Inspired by Lin et al. (Focal Loss, 2017) where output bias is pre-set to reflect marginal foreground class frequency, we initialize the output layer so that at step 0, **the neural network emits the empirical marginal Kaplan-Meier survival curve of the training cohort**.

#### Mathematical Formulation:
Let $\hat{S}_{\text{KM}}(s_k)$ be the non-parametric Kaplan-Meier survival probability estimated on the entire training set at bin center $s_k$.
The marginal interval hazard $\hat{h}_{\text{KM}}(k)$ is:
$$\hat{h}_{\text{KM}}(k) = \frac{\hat{S}_{\text{KM}}(s_{k-1}) - \hat{S}_{\text{KM}}(s_k)}{\hat{S}_{\text{KM}}(s_{k-1})}$$
The output bias vector $b \in \mathbb{R}^K$ of the final linear projection is initialized to:
$$b_k = \text{logit}(\hat{h}_{\text{KM}}(k)) = \log\left(\frac{\hat{h}_{\text{KM}}(k)}{1 - \hat{h}_{\text{KM}}(k)}\right)$$
The final projection weight matrix $W$ is initialized to very small values (e.g., $W \sim \mathcal{N}(0, 10^{-4})$ or zeros).

#### Advantages:
- **Zero-step calibration**: At initialization (before any gradient step), the model outputs the valid population-level survival curve ($C^{td} = 0.500$, valid Brier score).
- **Smooth personalization**: Training gradients do not need to fight an unnatural 99.999% mortality prior; they merely learn patient-specific deviations from the cohort baseline.
- **Divergence prevention in DeepTCSR**: $S(\Delta t)$ starts at realistic values ($> 0.8 \sim 0.95$), completely preventing the clamp bound from firing on early batches.

---

### Strategy 2: Anchor-Supervised Warm-Start (Curriculum Annealing)

#### Conceptual Rationale:
Temporal Difference (TD) targets are non-stationary because they depend on moving target network weights $\theta^-$. If $\theta^-$ emits noisy predictions early in training, the online network learns from corrupted targets (the "blind leading the blind" bootstrap instability).

#### Algorithmic Design:
1. **Warm-up Phase (Epochs 1–2)**:
   - Freeze the TD loss component by locking $\alpha_{\text{anchor}} = 1.0$.
   - Train purely on the strictly proper Censored CRPS Anchor Loss:
     $$\mathcal{L} = \mathcal{L}_{\text{anchor}} = \text{CRPS}(F_{\theta}, \text{residual\_time})$$
   - This shapes the latent representation to correlate with actual patient survival times.
2. **Annealing Phase (Epochs 3–5)**:
   - Anneal $\alpha$ linearly or cosine-wise from $1.0$ down to the target weight $\alpha = 0.0$ (or $0.25$).
3. **Consistency Phase (Epochs 6+)**:
   - Fine-tune with full temporal difference renewal consistency.

#### Advantages:
- Completely eliminates initial bootstrap variance.
- Guarantees that target network predictions are semantically meaningful before they are used as Bellman pseudo-ground truth.

---

### Strategy 3: Optimistic Survival Bias (Conservative Prior)

#### Conceptual Rationale:
Borrowed from the "Optimistic Initialization" principle in reinforcement learning: encourage exploration and stable value propagation by assuming favorable initial conditions.

#### Mathematical Formulation:
Initialize all final hazard biases to a constant negative scalar:
$$b_k = -3.5 \quad \forall k \in \{0, \dots, K-1\}$$
This yields an initial per-bin hazard:
$$h_k = \sigma(-3.5) \approx 0.029 \quad (2.9\% \text{ per bin})$$
Across 36 bins, initial survival at horizon is:
$$S(36) \approx (1 - 0.029)^{36} \approx (0.971)^{36} \approx 0.345$$

#### Advantages:
- Trivial 1-line implementation (`self.head[-1].bias.data.fill_(-3.5)`).
- Keeps survival discounts $\gamma_j = S(\Delta t_j) \in [0.90, 0.99]$ well above zero during early updates, ensuring full gradient flow across multi-step chains.

---

## 3. Implementation Blueprint & Verification Plan

```python
# In src/models/hazard_head.py
def init_marginal_km_bias(self, km_hazards: np.ndarray, eps: float = 1e-4):
    """
    Initializes the final linear layer bias to match the empirical Kaplan-Meier hazard.
    """
    assert len(km_hazards) == self.num_bins
    clamped_h = np.clip(km_hazards, eps, 1.0 - eps)
    km_logits = np.log(clamped_h / (1.0 - clamped_h))
    
    with torch.no_grad():
        self.output_layer.bias.copy_(torch.tensor(km_logits, dtype=torch.float32))
        self.output_layer.weight.normal_(mean=0.0, std=1e-4)
```

### Proposed Experiment for Follow-up:
- Compare:
  1. `Standard Init (Random / Zero Bias)`
  2. `Optimistic Init (b = -3.5)`
  3. `Marginal KM Prior Init`
  4. `Anchor Warm-Up (2 epochs)`
- Evaluation Metrics:
  - Epoch 1 validation $C^{td}$ (early convergence rate)
  - DeepTCSR clamp bound rate in Epoch 1 (target stabilization)
  - Final test $C^{td}$ and seed variance across 5 seeds
