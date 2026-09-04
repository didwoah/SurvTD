# Research Note: Continuous Pairwise Ranking via First-Order Stochastic Dominance in Cramér Space

**Author / Contributor**: Yang Jae-mo (didwoah) & Antigravity Research Agent  
**Date**: 2026-09-04  
**Status**: Methodological Extension Proposal (SurvTD-Rank Arm)  
**Target Modules**: `src/bellman.py`, `src/models/loss.py`, `src/models/trainer.py`

---

## 1. Problem Diagnosis: The Success and Pathology of DeepHit's Ranking Loss

In benchmark audits (`deviation_log.md:M-05`), Dynamic-DeepHit (Lee et al., 2018; 2019) was shown to derive a massive empirical performance boost (+0.185 C-index) entirely from its auxiliary pairwise ranking objective ($L_2$):

$$\mathcal{L}_{\text{DeepHit-Rank}} = \sum_{(i,j) \in \mathcal{P}} \exp\left( -\frac{F_i(t_i) - F_j(t_i)}{\sigma} \right)$$

where $F_i(t) = P(T_i \le t)$ is the predicted cumulative failure probability at event time $t_i$ for a comparable pair $(i, j)$ with $t_i < t_j$ and $E_i = 1$. When this ranking loss was ablated ($\alpha_{\text{rank}} = 0$), Dynamic-DeepHit's C-index collapsed from **0.8481 to 0.6633**.

### Pathologies of DeepHit's Formulation:
1. **Point Evaluation Discarding Distributional Information**:
   - The comparison is evaluated strictly at a single point in time ($t = t_i$). The model receives zero guidance about the relative ordering of survival probabilities at future horizons $t > t_i$.
2. **Dimensionally Incompatible Softmax Penalty**:
   - The exponential penalty requires tuning an arbitrary temperature hyperparameter $\sigma$. It is dimensionally decoupled from the log-likelihood or Cramér distance units.
3. **Severe Probability Miscalibration**:
   - Pairwise ranking losses are non-proper scoring rules. They aggressively push predictions toward extreme binary extremes (0 or 1) to maximize margin, severely degrading absolute calibration (Brier Score) and generating high false-alarm variance (alarm jitter) in clinical and industrial telemetry.

---

## 2. Proposed Formulation: Continuous First-Order Stochastic Dominance in Cramér Space

To incorporate pairwise discriminative power without sacrificing SurvTD's distributional rigor, we formulate pairwise ranking through the lens of **First-Order Stochastic Dominance (FSD)** directly within the **Cramér metric**.

### 2.1 Theoretical Rationale (Stochastic Dominance)
In decision theory and continuous survival analysis, if subject $j$ survives strictly longer than subject $i$ ($T_j > T_i, E_i = 1$), the remaining lifetime distribution of $j$ must **stochastically dominate** that of $i$:

$$T_j \succeq_{\text{st}} T_i \iff S_j(t) \ge S_i(t) \iff F_j(t) \le F_i(t) \quad \forall t \ge 0$$

Any state where $F_j(t) > F_i(t)$ represents a structural ranking reversal at horizon $t$.

### 2.2 Continuous Cramér Hinge Loss
SurvTD already trains via the squared $L_2$ Cramér distance on CDFs:
$$\mathcal{L}_{\text{Cramer}}(F, F^*) = \int_0^\infty \big(F(t) - F^*(t)\big)^2 \, dt$$

We define the **Pairwise Stochastic Dominance Cramér Loss** as the integrated one-sided violation of First-Order Stochastic Dominance across the entire prediction horizon:

$$\mathcal{L}_{\text{FSD}}(i, j) = \int_0^\infty \max\Big(0, \, F_j(t) - F_i(t) + \epsilon\Big)^2 \, dt$$

where $\epsilon \ge 0$ is an optional separation margin.

### 2.3 Discretized Implementation on the SurvTD Bin Grid
On SurvTD's fixed temporal grid with bin width $\delta_s$ and $K$ evaluation points $\{s_1, s_2, \dots, s_K\}$:

$$\mathcal{L}_{\text{FSD}}(i, j) = \delta_s \sum_{k=1}^K \max\Big(0, \, F_j(s_k) - F_i(s_k) + \epsilon\Big)^2$$

### 2.4 Full Batch Objective with IPCW Weighting
For a mini-batch with set of permissible pairs $\mathcal{P} = \{(i, j) : T_i < T_j, E_i = 1\}$:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{SurvTD-Bellman}} + \beta \cdot \mathcal{L}_{\text{Rank}}$$

$$\mathcal{L}_{\text{Rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} w_{ij}^{\text{IPCW}} \cdot \mathcal{L}_{\text{FSD}}(i, j)$$

where $w_{ij}^{\text{IPCW}} = \frac{1}{\hat{G}(T_i)^2}$ is the standard Uno-type Inverse Probability of Censoring Weighting.

---

## 3. Key Advantages of the Cramér FSD Formulation

1. **Dimensional & Physical Consistency**:
   - Both $\mathcal{L}_{\text{SurvTD-Bellman}}$ and $\mathcal{L}_{\text{FSD}}$ are expressed in the exact same physical units:
     $$\text{time} \times (\text{probability})^2$$
   - The scaling hyperparameter $\beta$ is a well-behaved scalar dimensionless trade-off weight, unlike DeepHit's temperature-dependent scaling.
2. **Whole-Horizon Curve Preservation**:
   - Rather than optimizing a single point $F(t_i)$, it forces the entire survival trajectory $S_j(t)$ to sit above $S_i(t)$, preserving smooth, monotonic survival dynamics.
3. **Automatic Gradient Vanishing on Concordant Pairs**:
   - When the predicted distributions satisfy $F_j(t) \le F_i(t) - \epsilon$ across all $t$, the hinge loss is exactly zero ($\nabla = 0$). It does not over-saturate concordant predictions or push valid probabilities toward degenerate endpoints.

---

## 4. Alternative Exploration: Preference-Guided Bellman Bootstrapping (DPO-Style)

Beyond adding an auxiliary loss, pairwise information can be directly embedded into the **Survival Bellman Operator target itself**:

Let subject $i$ and subject $j$ be observed at landmark $L$. If subject $i$ fails before subject $j$, we construct a **comparative advantage shift**:
$$\mathcal{T}_{\text{pair}} p_i = \mathcal{T}_{\text{Bellman}} p_i \circledast \Delta_{-\delta_{\text{margin}}}$$
$$\mathcal{T}_{\text{pair}} p_j = \mathcal{T}_{\text{Bellman}} p_j \circledast \Delta_{+\delta_{\text{margin}}}$$

This moves the target distribution of the shorter-lived subject leftward and the longer-lived subject rightward in lifetime space during target generation, embedding peer rankings directly into the Bellman consistency loop without adding a separate loss term.

---

## 5. Paper Strategy & Narrative (The "Double-Win")

In the research paper, this yields a decisive two-tier presentation:

| Model Arm | Primary Strength | Academic Purpose |
|---|---|---|
| **SurvTD-Pure** ($\beta = 0$) | SOTA Calibration (Brier Score) & Zero Alarm Jitter | Demonstrates the mathematical superiority of the continuous Semi-Markov Bellman operator over point-in-time maximum likelihood. |
| **SurvTD-Rank** ($\beta > 0$) | High C-index + Maintained Calibration | Directly challenges and surpasses Dynamic-DeepHit's discriminative performance on C-index, while exposing DeepHit's severe calibration flaws. |

---

## 6. PyTorch Prototype Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def continuous_cramer_pairwise_loss(
    cdf_preds: torch.Tensor,       # Shape: (B, K), cumulative distributions F(s_k)
    survival_times: torch.Tensor,  # Shape: (B,), observed event/censoring times
    event_indicators: torch.Tensor,# Shape: (B,), 1 if failure, 0 if censored
    delta_s: float,                # Grid bin width
    margin: float = 0.0,
    ipcw_weights: torch.Tensor = None # Shape: (B,), 1 / G(T_i)
) -> torch.Tensor:
    """
    Computes First-Order Stochastic Dominance (FSD) violation in Cramér space.
    For all pairs (i, j) where T_i < T_j and E_i == 1:
        Penalizes max(0, F_j(t) - F_i(t) + margin)^2 integrated over t.
    """
    B, K = cdf_preds.shape
    
    # 1. Identify permissible comparable pairs
    # time_diff[i, j] > 0 means T_j > T_i
    time_diff = survival_times.unsqueeze(0) - survival_times.unsqueeze(1) # (B, B), row=i, col=j
    events = event_indicators.unsqueeze(1)                                # (B, 1)
    
    pair_mask = (time_diff > 0) & (events == 1) # (B, B): True if (i, j) is valid pair
    
    if not pair_mask.any():
        return torch.tensor(0.0, device=cdf_preds.device, dtype=cdf_preds.dtype)
    
    # 2. Pairwise CDF difference across all bins
    # F_j(s_k) - F_i(s_k): Shape (B, B, K)
    # F_diff[i, j, k] = F_j(s_k) - F_i(s_k)
    F_i = cdf_preds.unsqueeze(1) # (B, 1, K)
    F_j = cdf_preds.unsqueeze(0) # (1, B, K)
    F_diff = F_j - F_i          # (B, B, K)
    
    # 3. One-sided Cramér violation: max(0, F_j - F_i + margin)^2
    fsd_violations = F.relu(F_diff + margin).pow(2) # (B, B, K)
    
    # Integrated over continuous time via Riemann sum: delta_s * sum_k
    pair_cramer_loss = delta_s * fsd_violations.sum(dim=-1) # (B, B)
    
    # 4. Apply IPCW weights if provided
    if ipcw_weights is not None:
        w_i = ipcw_weights.unsqueeze(1) # (B, 1)
        w_pair = w_i.pow(2)
        pair_cramer_loss = pair_cramer_loss * w_pair
        
    # 5. Mean over valid comparable pairs
    loss = pair_cramer_loss[pair_mask].mean()
    return loss
```
