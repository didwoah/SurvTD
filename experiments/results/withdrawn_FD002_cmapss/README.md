# Withdrawn — the FD002 C-MAPSS cells (2026-09-05)

These 16 cells are withdrawn from every NASA/C-MAPSS table. They are kept because
they were really measured, not because they are reportable.

## Why

Two incompatible C-MAPSS protocols existed in the tree at the same time, and this is
the one that is not the project's standard:

| | this directory | the standard |
|---|---|---|
| runner | `experiments/run_tier1.py --cohorts cmapss` | `experiments/benchmark_nasa_tier1_authentic.py` |
| loader | `src/data/cmapss_loader.py` | `src/data/nasa_protocol_loader.py` |
| dataset | **FD002**, Poisson-subsampled | **FD001**, 200 units |
| censoring | synthetic, `admin_censor_at = 100` | canonical: train = 1 (run to failure), test = 0 (truncated) |
| scorer | `src/evaluation/curve_scoring.py` (Antolini, landmark/horizon) | CoxSig's own `score()` (`baselines/signature_survival/src/utils.py`) |
| landmarks | L = 125 / 150 / 175, H = 50 | quantile `pred_times` from `setup_coxsig_horizons` |

Both are defensible evaluations. Neither can be a row in the other's table, and the
project reports NASA on the FD001 CoxSig protocol, so this one is withdrawn.
`run_tier1.py` no longer lists `cmapss` in its default cohorts; the cohort is still
reachable by name for the irregular-sampling study it was built for.

## What is in here

- `tier1_cmapss_fd002.json` — the 16 FD002 cells, alarms re-derived under the current
  rules.
- `tier1_pbc_cmapss_RAW.json` — the untouched file as the runner wrote it, PBC2 and
  C-MAPSS together. The PBC2 half is valid and lives on at
  `experiments/results/tier1/tier1_pbc.json`.

## What they said, for the record

SurvTD lost to a plain landmark Cox on all three landmarks, 0/5 seeds:

| | L=125 | L=150 | L=175 |
|---|---|---|---|
| landmark_cox | 0.7561 | 0.8091 | 0.7846 |
| survtd | 0.5445 | 0.5871 | 0.5974 |
| paired delta | −0.2116 | −0.2220 | −0.1872 |
| t | −8.80 | −7.69 | — |

Nine defect alarms fired. Seed 42 collapsed on both SurvTD arms with near-identical
numbers (`survtd|42` IBS 0.382/0.320/0.256; `survtd_anchor_only|42` IBS
0.380/0.318/0.255), so the collapse tracks the data split, not the TD term, and does
not bias the arm contrast. The other four seeds have in-band IBS and still sit at
0.55–0.65 against Cox's 0.78 — so the missing `apply_hazard_prior_init` repair
(A-16, absent from this runner) does **not** account for the gap.

The run was stopped part-way by decision, not by failure: `survtd_anchor_only` has
seed 42 only, and `ddh`/`tcsr`/`tcsr_landmark`/`coxsig`/`ncde`/`deeptcsr_tcn` never
ran on this cohort. Nothing here is a complete arm comparison.
