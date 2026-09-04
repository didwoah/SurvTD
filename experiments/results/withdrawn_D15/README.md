# Withdrawn — defect D15 (2026-09-04)

Every SurvTD and DeepTCSR arm in these directories trained with `has_event = False` on
**100% of trajectories**. The per-visit `events` array is an all-zero placeholder in all
four loaders, and `has_event` was read from it alone in `survtd.py`,
`survtd_operator.py` and `deeptcsr_clamped.py`. The terminal Dirac sits behind
`if event:` and so was never placed: at `alpha = 0` the objective held no ground truth
at all.

| cohort | trajectories | events | visit-level flags set |
|---|---|---|---|
| synthetic_icu | 300 | 86 | **0** |
| cmapss | 156 | **156** | **0** |
| pbc | 187 | 81 | **0** |
| tumor | 240 | 152 | **0** |

C-MAPSS has no censoring at all, so every one of its 156 fully observed failures was
treated as censored.

Correcting the flag, synthetic_icu seed 123, nothing else changed:

| arm | before | after |
|---|---|---|
| SurvTD anchor-only (alpha = 1) | 0.5185 | **0.7229** |
| SurvTD full (alpha = 0) | 0.4350 | **0.6927** |
| Person-Period raw (reference) | 0.7069 | — |
| Dynamic-DeepHit (reference) | 0.7112 | — |

**These results are withdrawn, not superseded.** They cannot serve as evidence in
either direction, in the same sense as `invalidated_2026-09-03/`.

Person-Period, Dynamic-DeepHit and KM-reference cells were never affected — those two
arms read the trajectory-level flag — and remain valid wherever they appear here. The
operator mathematics (D8, D9, D-gamma and their unit tests) is target-level and does
not depend on the flag.

See `research/continuous-survival-td/deviation_log.md` entry **D15**.
