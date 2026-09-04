"""
The absolute-time -> residual-time bridge, pinned against hand computation.

The adapter is where a wrong answer would be quietest: `evaluate_landmarked` reads
only `survival[-1]`, so a mis-specified conversion produces a plausible curve and a
plausible C-index with nothing obviously broken. These tests therefore check the
arithmetic against values computed by hand, not just the shape of the output.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.data.cohorts import COHORTS
from src.models.baselines.authentic.adapters import (
    ResidualSurvivalAdapter,
    absolute_survival_knots,
)
from src.models.baselines.authentic.ddh import (
    AuthenticDynamicDeepHit,
    discretize_absolute_times,
)


class FixedPMFModel(nn.Module):
    """Stub with a known, constant PMF so the conversion can be checked exactly."""

    def __init__(self, pmf):
        super().__init__()
        self.register_buffer("pmf", torch.as_tensor(pmf, dtype=torch.float32))

    def forward(self, x):
        b = x.shape[0]
        return None, [self.pmf.unsqueeze(0).repeat(b, 1)]


class TestKnots(unittest.TestCase):
    def test_knots_start_at_one_and_follow_the_edges(self):
        """Bin j covers (edge[j], edge[j+1]], so the mass through it is known at
        edge[j+1]; the curve starts at S = 1 on the first edge."""
        edges = np.array([0.0, 10.0, 20.0, 30.0])
        cif = np.array([0.2, 0.5, 0.9])
        times, surv = absolute_survival_knots(cif, edges)
        np.testing.assert_allclose(times, [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_allclose(surv, [1.0, 0.8, 0.5, 0.1])


class TestConditionalRenormalisation(unittest.TestCase):
    """S_resid(r | L) must equal S_abs(L + r) / S_abs(L), not S_abs(L + r)."""

    DELTA_S, K = 10.0, 3
    EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    PMF = np.array([0.1, 0.2, 0.3, 0.4])          # CIF: .1 .3 .6 1.0

    def _adapter(self):
        return ResidualSurvivalAdapter(FixedPMFModel(self.PMF), self.EDGES,
                                       self.DELTA_S, self.K).eval()

    def test_matches_hand_computed_ratio(self):
        """One visit at absolute t = 10, so L = 10 and S_abs(10) = 0.9.

        Knots are (0,10,20,30,40) -> (1, .9, .7, .4, 0). Queries L + {10,20,30} give
        S_abs = .7, .4, 0, so the conditional curve is .7/.9, .4/.9, 0/.9."""
        ad = self._adapter()
        x = torch.zeros(1, 1, 4)
        dts = torch.tensor([[10.0]])
        with torch.no_grad():
            _, surv, _, _ = ad(x, dts)
        expected = np.array([0.7, 0.4, 0.0]) / 0.9
        np.testing.assert_allclose(surv[0, 0].numpy(), expected, atol=1e-6)

    def test_unconditional_would_be_a_different_answer(self):
        """Guards against someone 'simplifying' the adapter to upstream's raw
        `1 - CIF(t)`, which scores a different estimand than every other arm."""
        ad = self._adapter()
        x = torch.zeros(1, 1, 4)
        with torch.no_grad():
            _, surv, _, _ = ad(x, torch.tensor([[10.0]]))
        unconditional = np.array([0.7, 0.4, 0.0])
        self.assertFalse(np.allclose(surv[0, 0].numpy(), unconditional, atol=1e-6))

    def test_later_landmark_gives_a_different_curve(self):
        ad = self._adapter()
        x = torch.zeros(1, 2, 4)
        with torch.no_grad():
            _, surv, _, _ = ad(x, torch.tensor([[10.0, 10.0]]))
        self.assertFalse(torch.allclose(surv[0, 0], surv[0, 1], atol=1e-6))

    def test_clamp_rate_is_reported(self):
        """Once the denominator floor binds, that subject's curve is set by `eps`
        rather than by the model -- and it binds on the highest-risk subjects. The
        rate has to be observable, not silent."""
        ad = ResidualSurvivalAdapter(FixedPMFModel(self.PMF), self.EDGES,
                                     self.DELTA_S, self.K, eps=0.99).eval()
        with torch.no_grad():
            ad(torch.zeros(1, 1, 4), torch.tensor([[10.0]]))
        self.assertGreater(ad.clamp_rate, 0.0)


class TestContractOnRealCohort(unittest.TestCase):
    """The four-tuple must behave like `DiscreteHazardHead`'s, on real PBC2 input."""

    @classmethod
    def setUpClass(cls):
        cls.spec = COHORTS["pbc"]
        cls.cd = cls.spec.load(seed=42)
        tte = [np.array([float(p["tte"])]) for p in cls.cd.train]
        _, split_time = discretize_absolute_times(tte, 50)
        torch.manual_seed(0)
        model = AuthenticDynamicDeepHit(input_dim=cls.cd.input_dim, output_dim=50,
                                        layers_rnn=1, hidden_rnn=10)
        cls.ad = ResidualSurvivalAdapter(model, split_time, cls.spec.delta_s,
                                         cls.spec.num_bins).eval()
        p = cls.cd.test[0]
        with torch.no_grad():
            cls.out = cls.ad(p["features"].unsqueeze(0), p["dts"].unsqueeze(0))
        cls.n_visits = len(p["dts"])

    def test_shapes(self):
        hazard, survival, pmf, cdf = self.out
        K = self.spec.num_bins
        self.assertEqual(tuple(survival.shape), (1, self.n_visits, K))
        self.assertEqual(tuple(pmf.shape), (1, self.n_visits, K + 1))

    def test_survival_is_a_valid_decreasing_curve(self):
        _, survival, _, _ = self.out
        self.assertTrue(bool((survival >= 0).all() and (survival <= 1).all()))
        self.assertTrue(bool((survival[..., 1:] <= survival[..., :-1] + 1e-9).all()))

    def test_mass_sums_to_one_with_the_overflow_symbol(self):
        _, _, pmf, _ = self.out
        torch.testing.assert_close(pmf.sum(-1), torch.ones_like(pmf.sum(-1)),
                                   atol=1e-5, rtol=0)

    def test_runs_through_the_real_evaluation_harness(self):
        from src.evaluation.landmark import evaluate_landmarked
        metrics = evaluate_landmarked(self.ad, self.cd.train, self.cd.test,
                                      self.spec.landmark_spec, self.spec.delta_s,
                                      torch.device("cpu"), strict=False)
        self.assertGreater(len(metrics), 0)
        for value in metrics.values():
            if not np.isnan(value["c_td"]):
                self.assertTrue(0.0 <= value["c_td"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
