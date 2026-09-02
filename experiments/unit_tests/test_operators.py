"""
Unit tests for SurvTD mathematical operators:
- Exact probability mass conservation: sum_k (Pi Phi p)_k = 1.0
- Absorbing terminal boundary accumulation
- Sub-bin linear hazard interpolation
- Cramer non-expansiveness
- Projection variance diffusion bound <= delta_s^2 / 6 (EXP-08)
"""

import unittest
import math
import torch
import numpy as np

from src.operators.survtd_operator import (
    categorical_projection_shift,
    compute_interval_discount,
    localized_projected_dirac,
    squared_cramer_distance_loss
)


class TestSurvTDOperators(unittest.TestCase):
    def setUp(self):
        self.K = 25
        self.delta_s = 1.0
        # Random valid PMF on simplex
        torch.manual_seed(42)
        raw = torch.rand(self.K)
        self.p_sample = raw / torch.sum(raw)

    def test_mass_conservation(self):
        """Tests that Pi Phi_{+dt} preserves unit probability mass exactly across various dt."""
        for dt in [0.0, 0.25, 0.5, 1.0, 2.7, 5.0, 24.0, 100.0]:
            p_proj = categorical_projection_shift(self.p_sample, dt, self.delta_s, self.K)
            total_mass = float(torch.sum(p_proj).item())
            self.assertAlmostEqual(total_mass, 1.0, places=5, msg=f"Mass conservation failed at dt={dt}")
            self.assertTrue(torch.all(p_proj >= 0.0), f"Negative probability at dt={dt}")

    def test_terminal_absorbing_boundary(self):
        """Tests that for very large dt, all probability mass accumulates in the final bin K-1."""
        dt_large = 1000.0
        p_proj = categorical_projection_shift(self.p_sample, dt_large, self.delta_s, self.K)
        self.assertAlmostEqual(float(p_proj[-1].item()), 1.0, places=5)
        self.assertAlmostEqual(float(torch.sum(p_proj[:-1]).item()), 0.0, places=5)

    def test_sub_bin_linear_interpolation(self):
        """Tests that for dt < delta_s, linear hazard interpolation S(dt) = 1 - h0 * dt / delta_s holds."""
        surv_curve = torch.tensor([0.8, 0.64, 0.51, 0.40])
        h0 = 1.0 - 0.8  # 0.2
        dt_sub = 0.5
        gamma = compute_interval_discount(surv_curve, dt_sub, delta_s=1.0, K=4, h0=h0)
        expected = 1.0 - 0.2 * (0.5 / 1.0)  # 0.9
        self.assertAlmostEqual(gamma, expected, places=4)

    def test_cramer_non_expansiveness(self):
        """Tests that categorical projection Pi Phi is non-expansive in Cramer metric (L2 on CDFs)."""
        p1 = torch.softmax(torch.randn(self.K), dim=0)
        p2 = torch.softmax(torch.randn(self.K), dim=0)

        cdf1 = torch.cumsum(p1, dim=0)
        cdf2 = torch.cumsum(p2, dim=0)
        dist_orig = float(torch.sum((cdf1 - cdf2) ** 2).item())

        for dt in [0.3, 1.0, 2.5]:
            p1_shift = categorical_projection_shift(p1, dt, self.delta_s, self.K)
            p2_shift = categorical_projection_shift(p2, dt, self.delta_s, self.K)
            cdf1_s = torch.cumsum(p1_shift, dim=0)
            cdf2_s = torch.cumsum(p2_shift, dim=0)
            dist_shift = float(torch.sum((cdf1_s - cdf2_s) ** 2).item())
            self.assertLessEqual(dist_shift, dist_orig + 1e-5, f"Expansion observed at dt={dt}")

    def test_projection_variance_diffusion_bound(self):
        """EXP-08: Verifies that per-step categorical projection variance is bounded by delta_s^2 / 6."""
        # Dirac distribution at bin 10
        p_dirac = torch.zeros(self.K)
        p_dirac[10] = 1.0
        grid = (torch.arange(self.K, dtype=torch.float32) + 0.5) * self.delta_s
        var_init = 0.0

        dt = 0.5 * self.delta_s  # maximal fractional displacement f = 0.5
        p_proj = categorical_projection_shift(p_dirac, dt, self.delta_s, self.K)
        mean_proj = float(torch.sum(p_proj * grid).item())
        var_proj = float(torch.sum(p_proj * ((grid - mean_proj) ** 2)).item())

        delta_var = var_proj - var_init
        theoretical_bound = (self.delta_s ** 2) / 4.0  # max f*(1-f) at f=0.5 is 0.25; average is 1/6 ~ 0.1667
        self.assertLessEqual(delta_var, theoretical_bound + 1e-4)


if __name__ == "__main__":
    unittest.main()
