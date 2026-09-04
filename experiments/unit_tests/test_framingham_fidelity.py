"""
Framingham cohort gate.

Framingham exists in Table 1 for **scale**: PBC2's test split is 63 subjects, which is
where most of this project's between-seed variance comes from, and Framingham's is
~880. That only helps if our copy is the cohort the Deep Survival Machines / Dynamic
DeepHit line of work publishes on, so these tests pin it to the bundled CSV and to the
upstream feature set.

They also pin the two places where this loader deliberately DIVERGES from upstream --
train-only imputation and standardisation -- because "match upstream" is exactly the
argument that would reintroduce defect X-08.
"""
import os
import unittest

import numpy as np
import pandas as pd

from src.data.cohorts import COHORTS
from src.data.framingham_loader import (
    CATEGORICAL_COLS,
    DEFAULT_CSV,
    FEATURE_COLS,
    NUMERIC_COLS,
)


class TestFraminghamFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DEFAULT_CSV):
            raise unittest.SkipTest(f"vendored {DEFAULT_CSV} not present")
        cls.raw = pd.read_csv(DEFAULT_CSV)
        cls.cd = COHORTS["framingham"].load(seed=42)
        cls.spec = COHORTS["framingham"]
        cls.patients = list(cls.cd.train) + list(cls.cd.val) + list(cls.cd.test)

    def test_subject_count(self):
        self.assertEqual(len(self.patients), 4434)
        self.assertEqual(self.raw["RANDID"].nunique(), 4434)

    def test_visit_rows_and_max_three_exams(self):
        self.assertEqual(sum(len(p["times"]) for p in self.patients), len(self.raw))
        self.assertEqual(max(len(p["times"]) for p in self.patients), 3)

    def test_event_rate(self):
        """0.3496 = 1550/4434, read straight off the subject-level DEATH column."""
        rate = float(np.mean([p["event"] for p in self.patients]))
        expect = self.raw.groupby("RANDID")["DEATH"].first().mean()
        self.assertAlmostEqual(rate, float(expect), places=6)
        self.assertAlmostEqual(rate, 0.3496, places=4)

    def test_feature_set_is_upstream_and_get_dummies_is_a_noop(self):
        """`dsm/datasets.py:89` runs `pd.get_dummies(dat_cat)`.

        On this file every one of those columns is numeric dtype, so get_dummies
        passes them through and the design matrix is 18 wide. The loader hard-codes
        18; if a future pandas ever dummified them, upstream's width would change and
        ours would not. This test is what would catch that.
        """
        dummied = pd.get_dummies(self.raw[CATEGORICAL_COLS])
        self.assertEqual(list(dummied.columns), CATEGORICAL_COLS)
        self.assertEqual(len(FEATURE_COLS), 18)
        self.assertEqual(FEATURE_COLS, CATEGORICAL_COLS + NUMERIC_COLS)
        self.assertEqual(self.cd.input_dim, 18)
        self.assertEqual(self.patients[0]["features"].shape[1], 18)

    def test_no_visit_at_or_after_the_event_time(self):
        """A covariate measured at or after the outcome would leak it."""
        for p in self.patients:
            self.assertLess(float(p["times"][-1]), float(p["tte"]))

    def test_mask_is_the_real_missingness_pattern(self):
        """BPMEDS 5.1% and GLUCOSE 12.4% missing -- the mask must not be all ones."""
        m = np.concatenate([p["mask"].numpy() for p in self.patients])
        self.assertLess(m.mean(), 1.0)
        raw_observed = self.raw[FEATURE_COLS].notna().to_numpy(dtype=float)
        self.assertAlmostEqual(m.mean(), raw_observed.mean(), places=6)

    def test_standardisation_is_fit_on_train_only(self):
        """X-08. Train is ~N(0,1); val/test are NOT, because they were not fit on.

        If statistics were fit on the pooled data every split would be centred, and
        the leak would be invisible. Asserting that the *test* split is off-centre is
        what makes this test able to fail.
        """
        def moments(ds):
            f = np.concatenate([p["features"].numpy() for p in ds])
            return float(f.mean()), float(f.std())

        tr_mean, tr_std = moments(self.cd.train)
        self.assertAlmostEqual(tr_mean, 0.0, places=4)
        self.assertAlmostEqual(tr_std, 1.0, places=4)
        te_mean, _ = moments(self.cd.test)
        self.assertNotAlmostEqual(te_mean, 0.0, places=6)

    def test_censoring_is_administrative_at_a_single_time(self):
        """Every censored subject is censored at 8766 days.

        Recorded because it changes how the IPCW IBS column must be read: the
        censoring KM stays at 1.0 throughout follow-up, so the weights are ~1 and the
        IPCW Brier score is close to an unweighted one.
        """
        cens = np.array([p["tte"] for p in self.patients if p["event"] < 0.5])
        self.assertEqual(len(np.unique(cens)), 1)
        self.assertAlmostEqual(float(cens[0]), 8766.0)

    def test_landmarks_are_feasible_and_dynamic(self):
        """Both landmarks must clear min_at_risk / min_events AND follow an exam.

        A landmark before the second exam would make this a baseline-covariate task
        dressed up as a dynamic one.
        """
        lspec = self.spec.landmark_spec
        exam2 = self.raw[self.raw.PERIOD == 2].TIME.median()
        self.assertEqual(len(lspec.landmarks), 2)
        for L in lspec.landmarks:
            self.assertGreater(L, exam2 - 100)
            at_risk = [p for p in self.cd.test if float(p["tte"]) > L]
            events = [p for p in at_risk if float(p["event"]) > 0.5]
            self.assertGreaterEqual(len(at_risk), lspec.min_at_risk)
            self.assertGreaterEqual(len(events), lspec.min_events)

    def test_modelled_horizon_exceeds_the_prediction_window(self):
        self.assertGreater(self.spec.horizon_span(), self.spec.landmark_spec.max_horizon())

    def test_split_is_subject_level_and_disjoint(self):
        ids = [set(p["id"] for p in ds) for ds in (self.cd.train, self.cd.val, self.cd.test)]
        self.assertEqual(sum(len(s) for s in ids), 4434)
        self.assertEqual(len(ids[0] | ids[1] | ids[2]), 4434)


if __name__ == "__main__":
    unittest.main()
