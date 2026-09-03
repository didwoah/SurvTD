"""
DEPRECATED: sepsis_loader.py contains the legacy degenerate generator
where tte = times[-1] + U(0.1, 2.0) and all censored subjects sit at 72.0h.

Use src.data.synthetic_icu_loader (load_synthetic_icu) or src.data.cohorts instead.
"""


def generate_sepsis_icu_cohort(*args, **kwargs):
    raise RuntimeError(
        "generate_sepsis_icu_cohort is DEPRECATED and disabled due to critical data "
        "generation defects (D12: degenerate event timing, uniform censoring, per-subject "
        "feature flattening). Use src.data.synthetic_icu_loader.load_synthetic_icu or "
        "src.data.cohorts.COHORTS['synthetic_icu'].load(seed) instead."
    )
