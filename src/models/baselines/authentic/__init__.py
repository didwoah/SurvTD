"""
Faithful ports of the baseline papers' own architectures.

Everything in `src/models/baselines/` (outside this package) shares this project's
GRU-D backbone and `DiscreteHazardHead` by design -- preregistration §3 promised
identical backbones across the ladder, and `person_period.py:117-120` records the
reasoning. That makes those arms answer "does the TD loss beat the DDH loss on our
backbone?", which is not the same question as "does SurvTD beat Dynamic-DeepHit?".

This package answers the second question. Each module mirrors a vendored upstream
clone as closely as the surrounding harness allows, keeps the authors' own sequence
encoder, head, discretisation and loss, and names the upstream file and line ranges it
follows. Only two things are shared with the rest of the ladder: the data split, and
the evaluation in `src/evaluation/landmark.py`.

Where a port must deviate, the deviation is stated in the module docstring rather than
absorbed silently -- a port that quietly "improves" its source is not a baseline.
"""

from src.models.baselines.authentic.ddh import (
    AuthenticDynamicDeepHit,
    authentic_ddh_total_loss,
)

__all__ = ["AuthenticDynamicDeepHit", "authentic_ddh_total_loss"]
