# NASA C-MAPSS FD001 Literature Benchmark Protocol

**Date**: 2026-09-04  
**Author**: Antigravity & Research Team  
**Target Context**: Direct Literature Parity Evaluation (DeepTCSR, CoxSig, SurvTD)  
**Spine Document**: `research/continuous-survival-td/deviation_log.md` (Amendment under X-10)

---

## 1. Background & Protocol Origins

In the time-varying survival regression literature, evaluating on large longitudinal degradation datasets has been an ongoing challenge. While medical datasets (such as PBC2 with 312 patients or AIDS with 467 patients) are relatively small, the NASA C-MAPSS turbofan engine degradation dataset (Saxena et al., 2008) provides rich, long-horizon telemetry.

Two prominent recent baselines established a specific evaluation convention on C-MAPSS FD001:
1. **CoxSig** (*Bleistein et al., 2023, "Path Signature Cox Proportional Hazards for Dynamic Survival"*, NeurIPS/GitHub):
   - Merged `train_FD001.txt` (100 engines) and `test_FD001.txt` (100 engines) into a 200-engine dataset.
   - Assigned `label=1` to train engines (run-to-failure) and `label=0` to test engines (cut off mid-operation).
   - Applied an 80/20 random split (160 train / 40 test).
2. **DeepTCSR** (*Mariana Vargas Vieyra & Pascal Frossard, EPFL, arXiv:2410.06786, Oct 2024*):
   - Directly adopted this exact 200-engine, 50% censored setup (Appendix Table 3: `Size: 200, H: 361, #Time-var: 16`).
   - Evaluated 80/20 random splits over 11 seeds (Table 1: DeepTCSR(0.95) CI $0.730 \pm 0.109$, IBS $0.092 \pm 0.032$; SA Landmarking CI $0.638 \pm 0.043$).

---

## 2. Pragmatic & Operational Justification

While C-MAPSS was originally developed for remaining useful life (RUL) regression, framing it as survival analysis under this protocol is **operationally and methodologically legitimate**:

1. **Realistic Observation Window in Predictive Maintenance**:
   - In industrial IoT and fleet health monitoring, data is collected at an observation date $T_{\text{obs}}$.
   - Assets that have experienced catastrophic failure are recorded with failure timestamps ($E=1$).
   - Assets that remain operational without failure up to $T_{\text{obs}}$ are legitimately right-censored at their current operating duration ($t_{\text{cutoff}}, E=0$). The future failure timestamp is unknown at the moment of inspection.
2. **Zero Reviewer Friction & Benchmark Parity**:
   - Competing on this exact benchmark removes any suspicion of cherry-picking or changing the rules to favor SurvTD.
   - It allows direct insertion of our numbers into the published benchmark tables of DeepTCSR and CoxSig.

---

## 3. Strict Methodological Amendment: Zero Data Leakage

In prior baseline implementations (`baselines/signature_survival/data_loader/load_NASA.py` and `baselines/deep_tcsr/utils.py`), a global normalization was applied across all 200 engines before the train/test split occurred. 

To maintain strict scientific integrity, our protocol enforces:
- **Rule**: The 200 engines are partitioned into Train (80%, 160 engines) and Test (20%, 40 engines) **prior to any feature statistics computation**.
- **Scaler**: `StandardScaler().fit(X_train)` is executed strictly on the 160 training trajectories. The learned mean and variance are then used to transform both `X_train` and `X_test`.

---

## 4. Benchmark Protocol Specification

| Parameter | Specification |
| :--- | :--- |
| **Dataset** | NASA C-MAPSS FD001 (`train_FD001.txt` + `test_FD001.txt`) |
| **Sample Size** | 200 units (100 uncensored $E=1$ + 100 right-censored $E=0$) |
| **Features** | 16 time-varying sensors (dropping 3 settings and 5 constant sensors with <10 unique values) |
| **Data Split** | 80% Train (160 units) / 20% Test (40 units) |
| **Seeds** | 5 seeds: `[42, 123, 456, 789, 101112]` |
| **Normalization** | Z-score normalization fitted strictly on Train split |
| **Evaluation Metrics** | 1. Harrell's Concordance Index (C-index)<br>2. Integrated Brier Score (IBS) |
| **Comparison Baselines** | 1. SA Init State (`base_cox.py`)<br>2. SA Landmarking (`baseline_cox.py`)<br>3. CoxSig (`baselines/signature_survival`)<br>4. DeepTCSR ($\lambda \in \{0.0, 0.9, 0.95\}$)<br>5. **SurvTD (Ours)** |

---

## 5. Published Target Performance (DeepTCSR Table 1 Reference)

| Method | CI Mean (±Std) ↑ | IBS Mean (±Std) ↓ |
| :--- | :---: | :---: |
| **SA Init State** | $0.461 \pm 0.089$ | $0.306 \pm 0.066$ |
| **SA Landmarking** | $0.638 \pm 0.043$ | $0.135 \pm 0.020$ |
| **DeepTCSR (0.0)** | $0.474 \pm 0.071$ | $0.503 \pm 0.047$ |
| **DeepTCSR (0.9)** | $0.715 \pm 0.097$ | $0.101 \pm 0.024$ |
| **DeepTCSR (0.95)** | $\mathbf{0.730 \pm 0.109}$ | $\mathbf{0.092 \pm 0.032}$ |
| **CoxSig** | $\approx 0.650 \sim 0.710$ | – |
| **SurvTD Target** | **$\ge 0.750$** | **$\le 0.090$** |

---

## 6. Implementation Architecture & Plan

### Component 1: `src/data/nasa_protocol_loader.py`
- Loads `train_FD001.txt` (units 1-100, $E=1$) and `test_FD001.txt` (units 101-200, $E=0$).
- Selects the 16 informative sensor channels.
- Provides a clean split function that accepts `seed` and `test_ratio=0.2`.
- Fits `StandardScaler` strictly on Train, transforms Test.
- Exports padded PyTorch / JAX tensors formatted for SurvTD, DeepTCSR, and CoxSig.

### Component 2: `experiments/benchmark_nasa_parity.py`
- Executes all models across the 5 seeds under identical conditions:
  1. **SurvTD**: Continuous survival TD model.
  2. **DeepTCSR**: Run directly using `baselines/deep_tcsr/`.
  3. **CoxSig**: Run directly using `baselines/signature_survival/`.
- Evaluates C-index and IBS using standard evaluation functions.
- Generates a consolidated results JSON and camera-ready Markdown comparison table.
