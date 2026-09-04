# Partially withdrawn — defect D15

`cohort1_raw_run.log` is the sole record of the 5-seed Cohort 1 run. Its rows split:

**Valid** — these arms read the trajectory-level event flag:
    km              0.5000 +- 0.0000
    person_period   0.6367 +- 0.0759
    dynamic_deephit 0.6461 +- 0.0605

**Withdrawn** — these trained with every event treated as censored:
    survtd          0.5536 +- 0.0573
    deeptcsr        0.5115 +- 0.0610   (IBS 0.506: it was fitting a world with no deaths)

**Kill Criterion 3 is therefore withdrawn**, in both directions. Its falsification of
`C_3` carries no information, and neither does the undetermined verdict against
DeepTCSR. `kc5_verdict.json` here is withdrawn in full — both of its arms are SurvTD.
