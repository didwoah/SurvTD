# Phase 1 — Bottleneck Identification & Method-Lineage Tree

## 1. Structural Bottleneck Statement

> Prior temporal consistency survival frameworks (`Maystre2022`, `DeepTCSR2024`) pioneered step-by-step risk consistency by formulating Bellman updates under **uniform discrete intervals $\Delta t = 1$** via integer index shifts. 
> However, extending this temporal consistency to real-world continuous, irregular telemetry remains an open challenge: naive continuous extensions via textbook Bayes conditioning introduce severe numerical instability ($\div S(\Delta t) \to \infty$) in high-risk regimes, while the theoretical conditions for contraction under non-linear neural representations remained unformalized.
> Meanwhile, unregularized dynamic survival models (`Lee2019`) suffer from severe trajectory variance and bedside alarm fatigue. Deep survival analysis has thus lacked a continuous-time temporal consistency operator with rigorous contraction guarantees.

---

## 2. Method-Lineage Tree

```text
[Continuous-Time RL / SMDP] (Bradtke1994)
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
[Statistical Informative Sampling]           [Distributional RL (C51)]
(Lin2001, Alaa2017)                          (Bellemare2017)
       │                                             │
       │ (Load-bearing assumption:                   │ (Contractive Bellman
       │  Observation is Endogenous)                 │  on Distributions)
       │                                             │
       ▼                                             ▼
[Dynamic Survival (Monte Carlo NLL)] ──► [Discrete TCSR] (Maystre2022)
(Lee2019 Dynamic-DeepHit)                      │ (Pioneered consistency;
(Bleistein2024 CoxSig)                         │  assumed discrete Δt=1)
       │ (Bottleneck: Variance,                ▼
       │  Temporal Jittering)            [DeepTCSR] (DeepTCSR2024)
       │                                       │ (Inherited discrete Δt=1)
       └───────────────────┬───────────────────┘
                           ▼
          [PROPOSED MOVE: SurvTD(λ) / AEGIS]
          - Subtractive Gap: Eliminate discrete unit-step Δt=1 restriction
          - Additive Gap: Continuous-Time Semi-Markov Bellman Operator
                          with Cramér Contraction & Duration-Geometric λ-Returns
```

### Gap Classification:
- **Subtractive Gap (at shared ancestor of TCSR)**:
  Every temporal-consistency survival method inherits the assumption that observation intervals are uniform and discrete ($\Delta t = 1$). Removing this constraint allows the model to handle arbitrary continuous, irregular telemetry ($\Delta t \in \mathbb{R}^+$) without artificial grid distortion or boundary leaks.
- **Additive Gap (at leaf)**:
  A continuous-time Semi-Markov Bellman Operator with Cramér-distance contraction mapping (Theorem 1) and duration-geometric multi-step $\lambda$-returns ($\lambda^{\Delta t / \delta_s}$) that preserves physical horizon invariance and suppresses clinical alarm fatigue.

---

## 3. Routing
- **Status**: `proceed`
- **Justification**: The gap is anchored in verified top-tier literature (`Maystre2022`, `Lin2001`, `Alaa2017`, `Lee2019`, `Bradtke1994`), isolates a structural mathematical conflict, and possesses a clear, falsifiable solution path.
