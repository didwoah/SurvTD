"""
Phase D gate, as unit tests.

The expensive half of the gate -- train the Dynamic-DeepHit port and the DDH worker on
the same bundle and check they agree -- lives in `experiments/phase_d_gate.py` because
it takes minutes. What is pinned here is everything that can be checked without
training, plus a red test for each of the three defects the gate turned up (D16, D17,
D18) so none of them can come back silently.
"""
import os
import sys
import unittest

import numpy as np
import torch

from src.data.cohorts import COHORTS
from src.evaluation.curve_scoring import evaluate_curves, predictions_from_curves
from src.evaluation.worker_format import build_worker_bundle

TCSR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "baselines/tcsr")


class TestKMLeakGate(unittest.TestCase):
    """A curve identical for every subject must score exactly 0.500, on every cohort.

    This is the gate that says the scorer reads information from the CURVES and not
    from the labels it was handed alongside them.
    """

    def _km(self, name):
        spec = COHORTS[name]
        cd = spec.load(seed=42)
        wb = build_worker_bundle(cd, spec)
        n = wb.paths_test.shape[0]
        flat = np.tile(np.linspace(1.0, 0.2, len(wb.eval_times)),
                       (n, len(wb.pred_times), 1))
        return evaluate_curves(flat, wb.surv_labels_test, wb.pred_times,
                               wb.eval_times, cd.train, spec.landmark_spec, strict=True)

    def test_pbc(self):
        for _, m in self._km("pbc").items():
            self.assertAlmostEqual(m["c_td"], 0.5, places=10)

    def test_framingham(self):
        for _, m in self._km("framingham").items():
            self.assertAlmostEqual(m["c_td"], 0.5, places=10)

    def test_cmapss(self):
        """The only cohort with `admin_censor_at`, so the only one that exercises the
        capping branch in `predictions_from_curves`."""
        res = self._km("cmapss")
        self.assertEqual(len(res), 3)                  # three landmarks, one horizon
        for _, m in res.items():
            self.assertAlmostEqual(m["c_td"], 0.5, places=10)


class TestCurveScorerContract(unittest.TestCase):
    """Properties every arm's curves are scored under, whoever produced them."""

    @classmethod
    def setUpClass(cls):
        cls.spec = COHORTS["pbc"]
        cls.cd = cls.spec.load(seed=42)
        cls.wb = build_worker_bundle(cls.cd, cls.spec)

    def test_at_risk_filter_is_the_scorers_job_not_the_workers(self):
        """Workers emit rows for ALL test subjects; `tte > L` is applied here.

        If a worker ever pre-filtered, its curve rows would silently mis-align with
        the label rows. Pinning the filter's location is what stops that.
        """
        labels = self.wb.surv_labels_test
        n = labels.shape[0]
        curves = np.tile(np.linspace(1.0, 0.1, 5), (n, 1))
        L = float(self.wb.pred_times[-1])
        preds = predictions_from_curves(curves, labels, L,
                                        np.linspace(1, 5, 5), self.spec.landmark_spec)
        expected = int(np.sum(labels[:, 0] > L))
        self.assertEqual(preds.n, expected)
        self.assertEqual(preds.n_dropped_not_at_risk, n - expected)

    def test_curves_are_forced_monotone_non_increasing(self):
        labels = self.wb.surv_labels_test
        n = labels.shape[0]
        rising = np.tile(np.array([0.2, 0.9, 0.5, 0.95, 0.3]), (n, 1))
        preds = predictions_from_curves(rising, labels, 0.0,
                                        np.linspace(1, 5, 5), self.spec.landmark_spec)
        self.assertTrue(np.all(np.diff(preds.surv, axis=-1) <= 1e-12))


class TestD16NoSequenceLengthLeak(unittest.TestCase):
    """D16. Truncating training sequences at `tte` makes length a proxy for the label.

    Dynamic-DeepHit takes the RNN state at the last non-NaN step, so the count of
    observed steps is an input feature. This test measures the leak directly on the
    bundle -- no training needed -- and asserts the landmark truncation removes it.
    """

    @classmethod
    def setUpClass(cls):
        spec = COHORTS["framingham"]
        cls.wb = build_worker_bundle(spec.load(seed=42), spec)

    @staticmethod
    def _rank_corr(a, b):
        from scipy.stats import spearmanr
        return abs(float(spearmanr(a, b).correlation))

    def test_tte_truncation_leaks_the_label(self):
        """The defect, pinned so the repair below is known to be repairing something."""
        times = self.wb.paths_train[:, :, 0]
        tte = self.wb.surv_labels_train[:, 0]
        n_obs = (times <= tte[:, None]).sum(axis=1)
        self.assertGreater(self._rank_corr(n_obs, tte), 0.99)

    def test_landmark_truncation_removes_the_leak(self):
        times = self.wb.paths_train[:, :, 0]
        tte = self.wb.surv_labels_train[:, 0]
        for L in self.wb.pred_times:
            at_risk = tte > L
            n_obs = (times[at_risk] <= L).sum(axis=1)
            # constant across subjects => carries no label information at all
            self.assertEqual(len(np.unique(n_obs)), 1)

    def test_worker_defaults_to_the_repaired_truncation(self):
        path = os.path.join(os.path.dirname(TCSR_DIR),
                            "dynamic_deephit_pytorch/ddh_worker.py")
        if not os.path.exists(path):                   # pragma: no cover
            self.skipTest("DDH worker not present")
        with open(path) as f:
            src = f.read()
        self.assertIn('"--train_truncation"', src)
        self.assertIn('default="landmark"', src)


class TestD18TCSRResidualGrid(unittest.TestCase):
    """D18. TCSR curve columns are PERIODS ahead; the shared clock is not uniform.

    `build_worker_bundle` inserts an extra early grid point for CoxSig's sake, so
    column `k + 1` is not `eval_times[k]` at landmarks below the insertion. Reading
    the column index directly was off by a whole grid step on PBC2's `L = 0`.
    """

    @classmethod
    def setUpClass(cls):
        if TCSR_DIR not in sys.path:
            sys.path.insert(0, TCSR_DIR)
        try:
            import tcsr_worker
        except Exception as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"tcsr_worker not importable: {exc}")
        cls.W = tcsr_worker

    def test_reads_the_horizon_the_scorer_asked_for(self):
        # Non-uniform grid, exactly the shape build_worker_bundle produces.
        grid = np.array([0.0, 3.0, 30.0, 60.0, 90.0, 120.0])

        class FakeModel:
            """S(c steps) = 1 - c/10, so the value alone identifies the column read."""
            @staticmethod
            def survival_curve(state):
                n = state.shape[0]
                return np.tile(1.0 - np.arange(6) / 10.0, (n, 1))

        seqs = np.zeros((4, len(grid), 2))
        eval_times = np.array([30.0, 60.0, 90.0])

        out = self.W.conditional_curves(FakeModel(), seqs, grid, [0], eval_times)
        # From period 0 (t=0), residual times of the columns are [0,3,30,60,90,120],
        # so 30/60/90 days are columns 2/3/4 -> 0.8, 0.7, 0.6.
        np.testing.assert_allclose(out[0, 0], [0.8, 0.7, 0.6], atol=1e-9)
        # The defect took columns 1/2/3 -> 0.9, 0.8, 0.7. This is the red assertion.
        self.assertNotAlmostEqual(out[0, 0, 0], 0.9, places=6)

    def test_uniform_region_is_unaffected(self):
        grid = np.array([0.0, 3.0, 30.0, 60.0, 90.0, 120.0])

        class FakeModel:
            @staticmethod
            def survival_curve(state):
                n = state.shape[0]
                return np.tile(1.0 - np.arange(6) / 10.0, (n, 1))

        seqs = np.zeros((4, len(grid), 2))
        # From period 2 (t=30) the grid IS uniform, so column k+1 is correct there.
        out = self.W.conditional_curves(FakeModel(), seqs, grid, [2], np.array([30.0, 60.0]))
        np.testing.assert_allclose(out[0, 0], [0.9, 0.8], atol=1e-9)


class TestLandmarkCox(unittest.TestCase):
    """The classical arm. Cheap enough to run in the suite, so it is pinned here."""

    @classmethod
    def setUpClass(cls):
        try:
            import sksurv                              # noqa: F401
        except ImportError:                            # pragma: no cover
            raise unittest.SkipTest("scikit-survival not installed")
        from src.models.baselines.authentic.landmark_cox import landmark_cox_curves
        cls.spec = COHORTS["pbc"]
        cls.cd = cls.spec.load(seed=42)
        cls.eval_times = (np.arange(cls.spec.num_bins) + 1.0) * cls.spec.delta_s
        cls.curves = landmark_cox_curves(cls.cd.train, cls.cd.test,
                                         cls.spec.landmark_spec, cls.eval_times)

    def test_shape_covers_every_test_subject(self):
        self.assertEqual(self.curves.shape,
                         (len(list(self.cd.test)),
                          len(self.spec.landmark_spec.landmarks),
                          len(self.eval_times)))

    def test_curves_are_valid_survival_functions(self):
        self.assertTrue(np.all((self.curves >= 0.0) & (self.curves <= 1.0)))
        self.assertTrue(np.all(np.diff(self.curves, axis=-1) <= 1e-12))

    def test_beats_chance_and_is_not_degenerate(self):
        """Below chance would be a defect alarm; exactly 0.5 would mean no signal."""
        labels = np.array([[float(p["tte"]), float(p["event"])] for p in self.cd.test])
        res = evaluate_curves(self.curves, labels, self.spec.landmark_spec.landmarks,
                              self.eval_times, self.cd.train, self.spec.landmark_spec,
                              strict=True)
        for key, m in res.items():
            self.assertGreater(m["c_td"], 0.6, f"{key} scored {m['c_td']:.4f}")
            self.assertNotAlmostEqual(m["c_td"], 0.5, places=6)

class TestD19ResumeRederivesAlarms(unittest.TestCase):
    """A cached cell must be re-checked against today's rules, not yesterday's.

    `run_tier1.py --resume` used to carry the stored `below_chance_alarms` list
    forward verbatim. The IBS band was added to the runner after the pbc/cmapss
    process had already launched, so 61 cells were scored with no IBS check and every
    later resume preserved that hole -- `cmapss|survtd|42` sat at IBS 0.38/0.32/0.26,
    three landmarks out of band, with nothing in the alarm list.
    """

    @staticmethod
    def _runner():
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "run_tier1.py")
        spec = importlib.util.spec_from_file_location("run_tier1_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_ibs_out_of_band_is_flagged(self):
        cell = {"L=125,H=50": {"c_td": 0.72, "ibs": 0.3818}}
        alarms = self._runner().cell_alarms("cmapss|survtd|42", cell)
        self.assertEqual(len(alarms), 1)
        self.assertIn("ibs=0.3818", alarms[0])

    def test_below_chance_is_flagged(self):
        cell = {"L=180,H=365": {"c_td": 0.3180, "ibs": 0.19}}
        alarms = self._runner().cell_alarms("pbc|coxsig|789", cell)
        self.assertEqual(len(alarms), 1)
        self.assertIn("c_td=0.3180", alarms[0])

    def test_both_rules_fire_on_one_landmark(self):
        cell = {"L=150,H=50": {"c_td": 0.4896, "ibs": 0.3196}}
        alarms = self._runner().cell_alarms("cmapss|survtd|42", cell)
        self.assertEqual(len(alarms), 2)

    def test_healthy_cell_is_silent(self):
        cell = {"L=0,H=365": {"c_td": 0.8428, "ibs": 0.19}}
        self.assertEqual(self._runner().cell_alarms("pbc|survtd|42", cell), [])

    def test_nan_does_not_trip_either_rule(self):
        """A failed arm reports NaN; that is an error, not a defect alarm."""
        cell = {"L=0,H=365": {"c_td": float("nan"), "ibs": float("nan")}}
        self.assertEqual(self._runner().cell_alarms("pbc|coxsig|101112", cell), [])


if __name__ == "__main__":
    unittest.main()
