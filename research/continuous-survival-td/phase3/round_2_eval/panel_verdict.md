# Phase 3 Round 2: Panel Verdict & Meta-Reviewer Synthesis

**Target Proposal**: `SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation`  
**Candidate Snapshot**: `research/continuous-survival-td/phase3/round_1_revision/candidate_r1.json`  
**Panel Mode**: 5 Independent Concurrent Seats (`chair`, `insider`, `empiricist`, `domain`, `theorist`)  

---

## 1. Scorecard & Progression Summary

| Critic Seat | Role ID | Axis A (Position) | Axis B (Method) | Axis C (Fit) | Axis D (Falsifiability) | Overall Score (0–100) | Round 2 Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Area Chair** | `chair` | 4 | 4 | 5 | 5 | **83** | `ADVANCE` |
| **Subfield Insider** | `insider` | 4 | 4 | 5 | 5 | **83** | `ADVANCE` |
| **Empirical Auditor** | `empiricist` | 4 | 4 | 4 | 5 | **75** | `ADVANCE` |
| **Domain Practitioner** | `domain` | 4 | 4 | 4 | 4 | **75** | `ADVANCE` |
| **Theoretician** | `theorist` | 4 | 3 | 4 | 4 | **67** | `REVISE` |
| **Panel Consensus** | **MEDIAN** | **4** | **4** | **4** | **5** | **75 / 100** | **`REVISE` (Non-Unanimous: 4 ADVANCE / 1 REVISE)** |

### Arithmetic & Gate Calculation:
- **Formula**: $\text{Overall Score} = \text{round}\left(100 \times \frac{4 + 4 + 4 - 3}{12}\right) = \mathbf{75}$
- **Score Band**: **`strong`** ($\ge 67$)
- **Unanimous Advance Gate**: **FAILED (4 ADVANCE, 1 REVISE)**.
  - Per the strict Unanimous Advance standard in `idea-forge`, advancing to Phase 4 requires an unequivocal 5 out of 5 agreement.
  - Because the Theoretician identified 3 valid mathematical/sample-estimator defects, the panel verdict for Round 2 is officially recorded as **`REVISE`**.
  - In-place single-seat hotfixing is strictly prohibited. The author must enter `round_2_revision/` and submit the revised candidate (`candidate_r2.json`) to a full 5-seat parallel panel in `round_3_eval/`.

---

## 2. Mandatory Revision Targets for Round 2 Revision

1. **Step 5 Sample Realization**: Explicitly branch the empirical sample target into continuation updates $(\Pi \Phi p_{\theta^-})$ on living visits and Dirac updates on terminal event visits.
2. **Step 6 Scalar IPCW Weighting**: Move $1/\hat{G}(t_M \mid X) \le 10.0$ to the scalar squared Cramér loss function to preserve unit probability mass.
3. **Step 7 Continuation Discount**: Nest $\gamma_j$ into the recursive continuation term $\lambda_j \gamma_j \Pi \Phi G_{j+1}$.

---

## 3. Final Round 2 Routing

```text
================================================================================
ROUND 2 VERDICT: REVISE -> Increment to round_2_revision / round_3_eval
================================================================================
```
