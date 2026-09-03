# Legacy Experiment Scripts (attempt_1)

The scripts in this directory were authored during attempt_1 and are retained for reproducibility / historical audit purposes. They are **not** part of the active pipeline.

## Script Inventory

- `run_lambda_ablation.py`: Divergent operator implementation (hardcoded paths, recursion without projection or mass conservation, self-distillation bug). Superseded by `src.operators.survtd_operator.ARMS` and Track B.
- `battle_*.py`, `demo_survtd.py`, `visualize_survtd.py`: Pre-repair exploratory scripts expecting the legacy 4-tuple data loader return instead of the 6-tuple `CohortData` interface.
