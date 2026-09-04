# WITHDRAWN — defect D15

Every SurvTD and DeepTCSR arm in this directory was trained with `has_event = False` on
100% of trajectories: the per-visit `events` array is an all-zero placeholder in every
loader, and `has_event` was read from it alone. The terminal Dirac sits behind
`if event:` and was therefore never placed, so at alpha = 0 the objective held no
ground truth at all.

Correcting the flag moves synthetic_icu seed 123 from 0.5185 to 0.7229 (anchor-only)
and 0.4350 to 0.6927 (full), against Person-Period's 0.7069 and Dynamic-DeepHit's
0.7112.

These numbers are withdrawn, not superseded. Person-Period, Dynamic-DeepHit and KM
reference cells in this directory remain valid. See `deviation_log.md` D15.
