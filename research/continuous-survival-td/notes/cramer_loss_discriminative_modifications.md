# Research Note: Discriminative & Steep Formulations of Cramér Loss for Survival Analysis

**Author / Contributor**: Yang Jae-mo (didwoah) & Antigravity Research Agent  
**Date**: 2026-09-04  
**Status**: Proposal for Discriminative Loss Redesign & SurvTD Anchor Upgrade  
**Target Modules**: `src/models/loss.py`, `src/bellman.py`, `src/models/trainer.py`

> **REVIEW STATUS (2026-09-04, post-KC5).** Formulation 1 is **validated and adopted**.
> Formulations 2 and 3 are **withdrawn** — both were measured to be improper scoring
> rules, and Formulation 3 is directionally backwards against this note's own
> diagnosis. See §5 "Review findings" at the end. The §1 diagnosis itself is correct
> and is confirmed by the anchor-geometry diagnostic (`deviation_log.md` D-anchor:
> loss geometry accounts for +0.0998 of the deficit, grid expansion for -0.0026).

---

## 1. Problem Diagnosis: The "Mild Gradient" Pathology of $L_2$ Cramér Loss

In recent benchmarks on dynamic survival prediction, a sharp discrepancy was observed between models trained under Binary Cross-Entropy (BCE / MLE) versus standard squared Cramér distance ($L_2$ on CDF):

* **Person-Period (BCE / MLE)**: Achieves $C^{td} = 0.6367$.
* **SurvTD Anchor-Only ($\alpha = 1$, $L_2$ Cramér)**: Drops to $C^{td} \approx 0.5318 \sim 0.5400$.

Both models share identical backbones, identical hazard heads, and zero Temporal Difference bootstrapping. The root cause lies in the **loss geometry** and the gradient response to incorrect predictions:

### 1.1 Binary Cross-Entropy (MLE)
$$\mathcal{L}_{\text{BCE}} = - \sum_{k} \Big[ y_k \log h_k + (1 - y_k) \log (1 - h_k) \Big]$$
$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z_k} \propto \frac{1}{h_k} \quad (\text{or } \frac{1}{1 - h_k})$$

When the network is confidently wrong (e.g., predicting negligible hazard $h_k \to 0$ when an event actually occurs $y_k = 1$), the gradient **explodes toward $\infty$**. This violently penalizes misclassifications and forces the model to aggressively separate high-risk subjects from low-risk subjects, maximizing pairwise ranking ($C^{td}$).

### 1.2 Standard Squared Cramér Loss ($L_2$ on CDF)
$$\mathcal{L}_{\text{Cramer}} = \int_0^\infty \big( F(t) - F^*(t) \big)^2 dt$$
$$\frac{\partial \mathcal{L}_{\text{Cramer}}}{\partial F(t)} = 2 \big( F(t) - F^*(t) \big)$$

Because $F, F^* \in [0, 1]$, the gradient is strictly linear in the residual and bounded in $[-2, 2]$. 
* When two patients have subtle risk differences ($\Delta F \approx 0.05$), the gradient is a minuscule $0.10$.
* The $L_2$ loss allocates almost all its capacity to minimizing aggregate population residuals (matching the global survival curve shape, yielding excellent Integrated Brier Scores), while providing insufficient gradient force to order individuals accurately.

---

## 2. Three Proposed Formulations with Steep/Exploding Penalties

To grant Cramér-style continuous losses the same aggressive separation power as MLE, we introduce three mathematical variants that severely penalize confident errors.

---

### Formulation 1: Threshold-Integrated Cross-Entropy (Logit-Cramér Loss) — *Recommended*

#### Mathematical Rationale:
The cumulative failure distribution $F(t) = P(T \le t)$ can be viewed as the probability of a Bernoulli trial at every continuous threshold $t$: "Has the subject experienced the event by time $t$?" ($F^*(t) = \mathbb{I}[T \le t]$).

Instead of measuring the $L_2$ distance between $F(t)$ and $F^*(t)$, we integrate the binary cross-entropy across all continuous thresholds $t$:

$$\mathcal{L}_{\text{Logit-Cramer}} = - \int_0^\infty \Big[ F^*(t) \log F(t) + \big(1 - F^*(t)\big) \log\big(1 - F(t)\big) \Big] dt$$

#### Gradient Analysis:
Differentiating with respect to the predicted cumulative probability $F(t)$:

$$\frac{\partial \mathcal{L}_{\text{Logit-Cramer}}}{\partial F(t)} = \frac{F(t) - F^*(t)}{\mathbf{F(t)\big(1 - F(t)\big)}}$$

#### Why This Solves the Discrimination Deficit:
1. **Exploding Gradient on Confident Errors**:
   - If the model confidently predicts low risk ($F(t) \to 0$) but the event has occurred ($F^*(t) = 1$), the denominator collapses to zero ($F(1-F) \to 0$), and the gradient **explodes to $+\infty$**.
   - If the model predicts high risk ($F(t) \to 1$) for a survivor ($F^*(t) = 0$), the gradient **explodes to $-\infty$**.
2. **Continuous Temporal Geometry**:
   - Unlike standard discrete-visit BCE which evaluates isolated hazard points, Logit-Cramér integrates across the entire continuous timeline, penalizing the entire temporal span where the prediction was incorrectly confident.

#### Discretized Computation on Grid with Bin Width $\delta_s$:
$$\mathcal{L}_{\text{Logit-Cramer}} = - \delta_s \sum_{k=1}^K \Big[ F^*(s_k) \log\big(F(s_k) + \epsilon\big) + \big(1 - F^*(s_k)\big) \log\big(1 - F(s_k) + \epsilon\big) \Big]$$

---

### Formulation 2: Inverse-Margin Focal-Cramér Loss

#### Mathematical Rationale:
Inspired by Lin et al. (Focal Loss, 2017), we scale the squared Cramér distance by an inverse margin factor that blows up as the absolute error approaches 1:

$$\mathcal{L}_{\text{Focal-Cramer}} = \int_0^\infty \frac{\big(F(t) - F^*(t)\big)^2}{\Big(1 - |F(t) - F^*(t)| + \epsilon\Big)^\gamma} dt \quad (\gamma \ge 1)$$

#### Properties:
* For minor residuals ($|F - F^*| \le 0.1$), the denominator is $\approx 1$, behaving identically to standard Cramér distance.
* For severe errors ($|F - F^*| \to 1$), the denominator shrinks to $\epsilon^\gamma$, amplifying the loss and gradients by $10^2 \sim 10^4\times$.
* Focuses gradient updates exclusively on hard, misordered patients while ignoring well-calibrated survivors.

---

### Formulation 3: Higher-Order $L_p$ Cramér Distance ($p = 4$)

$$\mathcal{L}_{L_4} = \int_0^\infty \big( F(t) - F^*(t) \big)^4 dt$$
$$\frac{\partial \mathcal{L}_{L_4}}{\partial F(t)} = 4 \big( F(t) - F^*(t) \big)^3$$

* Suppresses small errors ($0.1^3 = 0.001$) while heavily punishing large discrepancies ($0.9^3 = 0.729$, a $729\times$ relative difference).
* Remnants of the $L_p$ family preserve metric properties while enhancing tail separation.

---

## 3. Theoretical Decoupling: Preserving SurvTD's Contraction Theorem

A central theoretical contribution of SurvTD is that the categorical projection operator $\Pi$ is a **non-expansion strictly under the $L_2$ Cramér metric**:
$$\|\Pi p - \Pi q\|_{l_2} \le \|p - q\|_{l_2}$$

Changing the metric of the Bellman target could potentially compromise the non-expansiveness proof of the operator $\mathcal{T} = \Pi \Phi$.

### The Solution: Asymmetric Objective Architecture
The anchor loss ($\mathcal{L}_{\text{anchor}}$) evaluates the network's prediction directly against the ground-truth empirical step function ($F^*(t) = \mathbb{I}[t \ge T_i]$). **It does not undergo categorical projection $\Pi$.**

Therefore, the objective function can naturally decouple the two components:

$$\mathcal{L}_{\text{Total}} = (1 - \alpha) \underbrace{\mathcal{L}_{\text{TD}}(\text{Pure } L_2 \text{ Cramér})}_{\text{Maintains Bellman Contraction Theorem } C_1} + \alpha \underbrace{\mathcal{L}_{\text{Anchor}}(\text{Logit-Cramér / BCE})}_{\text{Restores Sharp Discriminative Separation } (C^{td} \ge 0.65)}$$

* **Theoretical Integrity**: The Bellman contraction theorem $C_1$ remains 100% mathematically valid because the bootstrap operator $\Pi \Phi$ is optimized in its native $L_2$ Cramér metric.
* **Empirical Parity**: The ground-truth supervision provides the necessary steep gradients to match or exceed Person-Period MLE baseline discrimination.

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def logit_cramer_loss(
    pred_cdf: torch.Tensor,    # Shape: (B, K), predicted CDF values in (0, 1)
    target_cdf: torch.Tensor,  # Shape: (B, K), ground-truth CDF step function in {0, 1}
    delta_s: float,            # Grid bin spacing
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Threshold-Integrated Cross-Entropy (Logit-Cramér Loss).
    Gradients scale with 1 / [F(1-F)], exploding on confident errors.
    """
    # Clamp for numerical stability
    pred_clamped = torch.clamp(pred_cdf, min=eps, max=1.0 - eps)
    
    # Binary cross-entropy integrated across all threshold bins
    bce = - (target_cdf * torch.log(pred_clamped) + (1.0 - target_cdf) * torch.log(1.0 - pred_clamped))
    
    # Continuous Riemann sum: delta_s * sum_k
    loss_per_subject = delta_s * torch.sum(bce, dim=-1)
    return loss_per_subject.mean()


def focal_cramer_loss(
    pred_cdf: torch.Tensor,    # Shape: (B, K)
    target_cdf: torch.Tensor,  # Shape: (B, K)
    delta_s: float,
    gamma: float = 2.0,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Computes Inverse-Margin Focal-Cramér Loss.
    Scales standard L2 Cramér loss by 1 / (1 - |error| + eps)^gamma.
    """
    error = torch.abs(pred_cdf - target_cdf) # (B, K)
    focal_weight = 1.0 / torch.pow(1.0 - error + eps, gamma)
    
    squared_error = torch.pow(pred_cdf - target_cdf, 2)
    weighted_loss = focal_weight * squared_error
    
    loss_per_subject = delta_s * torch.sum(weighted_loss, dim=-1)
    return loss_per_subject.mean()
```

---

## 5. Review findings (2026-09-04)

Every claim below was checked numerically, not by argument.

### 5.1 Propriety — the decisive test

For a Bernoulli threshold with true probability `p`, a proper scoring rule has
`argmin_F E[loss(F, Y)] = p`. Measured:

| loss | p=0.10 | p=0.30 | p=0.50 | p=0.70 | p=0.90 | verdict |
|---|---|---|---|---|---|---|
| L2 Cramér `(F-Y)^2` | 0.100 | 0.300 | 0.500 | 0.700 | 0.900 | **PROPER** |
| **Logit-Cramér** | **0.100** | **0.300** | **0.500** | **0.700** | **0.900** | **PROPER** |
| L4 Cramér `(F-Y)^4` | 0.325 | 0.430 | 0.500 | 0.570 | 0.675 | **IMPROPER** |
| Focal-Cramér (γ=2) | 0.366 | 0.447 | 0.500 | 0.553 | 0.634 | **IMPROPER** |

Formulations 2 and 3 shrink every minimiser toward 0.5. A subject whose true risk
is 0.10 is optimally predicted at 0.33–0.37. Calibration is destroyed structurally,
not incidentally. Only `p = 2` is proper in the `L_p` family; focal weighting is
known to be improper for the same reason.

**This also contradicts the sibling note.** `pairwise_cramer_stochastic_dominance.md`
criticises DeepHit because "pairwise ranking losses are non-proper scoring rules".
Adopting two non-proper losses here would forfeit that argument.

### 5.2 Formulation 3 is backwards

This note diagnoses the pathology as "when two patients have subtle risk differences
(ΔF ≈ 0.05), the gradient is a minuscule 0.10". Measured gradient at exactly that
operating point (`|F - F*| = 0.05`):

| loss | `dL/dF` | vs L2 |
|---|---|---|
| L2 Cramér | 0.1000 | 1.00x |
| **Logit-Cramér** | **1.0526** | **10.53x** |
| L4 Cramér | 0.0005 | **0.01x** |
| Focal-Cramér (γ=2) | 0.1166 | 1.17x |

L4 makes the diagnosed problem **100x worse**. The note frames "suppresses small
errors" as a virtue, but that suppression *is* the bottleneck it set out to fix.
Focal-Cramér's advertised "10^2–10^4x amplification" occurs only as `|error| -> 1`,
a region where L2 already supplies adequate signal; in the discrimination regime it
is 1.17x, i.e. nothing.

### 5.3 Formulation 1 delivers what it claims

Proper (above), and 10.5x the gradient in the discrimination regime, 500x on a
confident error (`F = 1e-3` while `F* = 1`). Adopted.

It is also, in effect, a rediscovery of standard practice in categorical
distributional RL: C51 (Bellemare et al., 2017) proves its projection a contraction
in the Cramér metric but trains with cross-entropy, and Rowland et al. (2018)
analyse that deliberate mismatch. **A-04 went the other way** — it matched the loss
to the metric for coherence, and the diagnostic measured that coherence costing
-0.0998 C^td. *(Citations from memory; verify before use in the paper — this is
both the strongest supporting precedent and the most likely point of attack.)*

### 5.4 §3's asymmetric-objective argument is correct, and is the key

The anchor never passes through the projection `Π`, so `C_1` constrains the TD
target operator and not the anchor. The two terms may live in different geometries
with no cost to `thm:1`. This is the observation that makes the whole change safe,
and it is the most valuable contribution in this note.

### 5.5 Three risks §2 does not address

1. **The grid tails dominate.** `1/(F(1-F))` diverges as `F -> 0` or `F -> 1`.
   Early in the grid nearly every subject has `F ~ 0`; late, `F ~ 1`. So the
   steepest gradients land where the data is thinnest. Person-Period's BCE avoids
   this through at-risk masking, which the CDF-threshold formulation has no
   analogue for. **Log the per-bin gradient distribution when this first runs.**
2. **It collides with gradient clipping.** `trainer.py:205` already clips at
   `max_norm = 2.0`. With `eps = 1e-6` the loss admits gradients up to ~1e6, so the
   clip would fire on nearly every step, degrading the optimiser to sign-SGD.
   **Log the clip activation rate.**
3. **It is a scoring rule, not a likelihood.** The per-threshold Bernoulli terms are
   not independent (`F` is monotone), so the sum over-counts. It is proper, so this
   is legitimate — but "switching the anchor to MLE" is the wrong description, and
   `alpha = 1` is *not* guaranteed to reproduce Person-Period exactly.

### 5.6 A fourth option this note omits: cross-entropy on the PMF

`-log p_{k_j}` (event) / `-log S_{k_j}` (censored) is Dynamic-DeepHit's `L1`
(`dynamic_deephit.py:82-89`), and Person-Period's masked BCE is its close relative.
**Both reach ~0.64 on this cohort**, which makes CE the option with the strongest
empirical support of any considered here — and this note does not consider it.

It has one structural property worth knowing. Measured through the model's actual
hazard-cumprod parameterisation, CE's loss and gradient are **bit-identical**
whether the misplaced mass sits 1 bin or 24 bins from the truth (loss 3.8111,
`|grad|` 1.2699 in all cases), because `p_{k_j} = h_{k_j} * prod_{m<k_j}(1 - h_m)`
does not involve any hazard beyond bin `k_j`. CE is ordinal-blind and the
parameterisation does not rescue it.

That is not automatically a defect: the likelihood genuinely carries no information
about the period after the observed event or censoring time, so CE declining to
constrain it is honest. The Cramér family additionally penalises `F != 1` after the
event, which is true information but lies outside the likelihood. Both are proper;
they differ in how much information they use. **Which is better here is an empirical
question, and it is the one the A-17 experiment answers.**

