"""
RED TESTS: characterization of known operator defects.

Every test in this file is EXPECTED TO FAIL against the pre-repair operator and
to pass once the corresponding fix lands. They are the contract that proves a
fix actually took effect, rather than being masked by a downstream metric.

Defects covered (see plan appendix B):
  D8  - inter-visit durations misaligned by one index, so the "full" model was
        already running the NC-B duration-permutation negative control.
  D9  - the single ground-truth Dirac is placed at the length of the previous
        interval instead of the residual time to event.
  D10 - gamma = S(dt) is interpolated half a bin too early.
  D-gamma - the post-mixture renormalization absorbs gamma into a
        reparameterization of lambda, so gamma is mathematically inert.

Run with:
    python -m unittest experiments.unit_tests.test_operator_defects -v
"""

import unittest

import torch

from src.operators.survtd_operator import (
    categorical_projection_shift,
    compute_interval_discount,
    compute_lambda_returns,
    compute_multistep_lambda_returns,
    interval_gaps,
    localized_projected_dirac,
    one_step_renewal_target,
)


def _bin_of(s: float, delta_s: float, K: int) -> int:
    """Bin carrying the most mass for a projected Dirac at residual time s."""
    return int(localized_projected_dirac(s, delta_s, K).argmax())


def _flat_survival(value: float, L: int, K: int) -> torch.Tensor:
    """(L, K) survival curves that are constant in k, so gamma = `value` for any dt >= delta_s."""
    return torch.full((L, K), float(value))


def _uniform_pmfs(L: int, K: int) -> torch.Tensor:
    return torch.full((L, K), 1.0 / K)


class TestDurationAlignment(unittest.TestCase):
    """D8: the operator must use the interval FOLLOWING visit j for transition j -> j+1."""

    def test_total_shift_uses_forward_gaps(self):
        # Loader convention is a backward difference: dts[j] = times[j] - times[j-1].
        #   times = [1, 11, 16]  ->  dts = [1, 10, 5]
        # Transition 0->1 spans times[1]-times[0] = 10 = dts[1]
        # Transition 1->2 spans times[2]-times[1] =  5 = dts[2]
        # so the total shift from visit 2 back to visit 0 must be 15.
        # The buggy code uses dts[0] + dts[1] = 11 instead.
        K, delta_s = 40, 1.0
        dts = torch.tensor([1.0, 10.0, 5.0])
        tte = 16.0
        events = torch.tensor([0.0, 0.0, 0.0])  # censored: recursion stays on the alive branch

        # lambda = 1 makes the recursion pure propagation, so the argmax of G_0
        # is the argmax of G_2 displaced by exactly the accumulated shift.
        pmfs = torch.zeros(3, K)
        pmfs[:, 0] = 1.0  # terminal target for a censored trajectory is pmfs[-1] -> bin 0

        G, _ = compute_multistep_lambda_returns(
            pmfs, _flat_survival(1.0, 3, K), dts, events,
            tte=tte, tau_event=tte + 100.0, lam=1.0, delta_s=delta_s, K=K,
        )

        observed_shift = int(G[0].argmax()) - int(G[2].argmax())
        self.assertEqual(
            observed_shift, 15,
            msg=(f"D8: accumulated shift was {observed_shift}, expected 15 "
                 f"(times[2]-times[1]=5 plus times[1]-times[0]=10). "
                 f"A value of 11 means dts[j] was used where dts[j+1] is required, "
                 f"which destroys the per-transition duration correspondence."),
        )


class TestTerminalDiracLocation(unittest.TestCase):
    """D9: the terminal ground-truth Dirac must sit at the residual time tte - t_last."""

    def test_terminal_dirac_at_residual_time(self):
        # times = [1, 4, 8], tte = 8.5  ->  residual at the last visit is 0.5
        K, delta_s = 30, 2.5
        dts = torch.tensor([1.0, 3.0, 4.0])
        tte = 8.5
        events = torch.tensor([0.0, 0.0, 1.0])

        G, _ = compute_multistep_lambda_returns(
            _uniform_pmfs(3, K), _flat_survival(0.9, 3, K), dts, events,
            tte=tte, tau_event=tte, lam=0.6, delta_s=delta_s, K=K,
        )

        expected = _bin_of(tte - 8.0, delta_s, K)   # residual 0.5 -> bin 0
        observed = int(G[-1].argmax())
        self.assertEqual(
            observed, expected,
            msg=(f"D9: terminal Dirac landed in bin {observed}, expected {expected}. "
                 f"Placing it at dt_last={float(dts[-1])} instead of the residual "
                 f"time {tte - 8.0} points the only ground-truth signal in the "
                 f"objective at the wrong bin."),
        )


class TestGammaHalfBin(unittest.TestCase):
    """D10: survival[k] = P(R > (k+1)*delta_s), so S(dt) must be read at index dt/delta_s - 1."""

    def test_gamma_at_one_bin_equals_first_entry(self):
        K, delta_s = 8, 1.0
        surv = torch.tensor([0.80, 0.64, 0.51, 0.40, 0.32, 0.26, 0.20, 0.16])

        gamma = compute_interval_discount(surv, dt=delta_s, delta_s=delta_s, K=K)

        self.assertAlmostEqual(
            gamma, float(surv[0]), places=6,
            msg=(f"D10: gamma(dt=delta_s) = {gamma:.6f}, expected S[0] = {float(surv[0]):.6f}. "
                 f"An interpolated value between S[0] and S[1] means step_pos uses "
                 f"dt/delta_s - 0.5 instead of dt/delta_s - 1.0, over-discounting by half a bin."),
        )

    def test_gamma_is_continuous_at_the_sub_bin_boundary(self):
        # The sub-bin branch gives S(delta_s) = 1 - h0 = S[0]. The discrete branch
        # must agree with it in the limit, which is what pins the -1.0 convention.
        K, delta_s = 8, 1.0
        surv = torch.tensor([0.80, 0.64, 0.51, 0.40, 0.32, 0.26, 0.20, 0.16])

        sub_bin = compute_interval_discount(surv, dt=delta_s * 0.999, delta_s=delta_s, K=K)
        at_bin = compute_interval_discount(surv, dt=delta_s, delta_s=delta_s, K=K)

        self.assertAlmostEqual(
            sub_bin, at_bin, places=3,
            msg=(f"D10: gamma is discontinuous at dt = delta_s "
                 f"({sub_bin:.6f} just below vs {at_bin:.6f} at the boundary)."),
        )


class TestGammaIsNotInert(unittest.TestCase):
    """
    D-gamma: under the shipped operator the post-mixture renormalization made gamma
    exactly equivalent to a reparameterization of lambda, so the NC-A1 discount
    ablation was a near-no-op. These tests assert the old identity is now FALSE and
    that gamma controls the near/far split with the correct functional form.

    gamma is no longer supplied via `target_survivals`. It is now derived from the
    mass of the restricted near branch,
        gamma_j = 1 - P_theta-,j(R <= g_j),
    so the mixture sums to 1 by construction instead of relying on two independently
    interpolated quantities agreeing. The tests therefore control gamma by placing
    mass in visit j's own pmf.

    Construction (K=40, delta_s=1, all gaps 1.0, L=2, censored):
        p_0 = (1-gamma) at bin 0, gamma at bin 30   -> restrict_[0,1] keeps bin 0
        p_1 = 1.0 at bin 10                         -> shift by 1 lands at bin 11
      near branch        -> bin 0,  mass (1-lambda)(1-gamma)
      one-step far + recursive branch -> bin 11, mass (1-lambda)*gamma + lambda
    Both branches are well separated, so the mixture weights are read off directly.
    """

    K = 40
    DELTA_S = 1.0
    LAM = 0.6
    NEAR_BIN = 0
    FAR_BIN = 11

    def _returns(self, gamma_value: float) -> torch.Tensor:
        L = 2
        dts = torch.ones(L)
        pmfs = torch.zeros(L, self.K + 1)
        pmfs[0, 0] = 1.0 - gamma_value      # dies inside the interval
        pmfs[0, 30] = gamma_value           # survives it
        pmfs[1, 10] = 1.0

        G, _ = compute_lambda_returns(
            pmfs, torch.full((L, self.K), 0.9), dts, tte=2.0, event=False,
            lam=self.LAM, delta_s=self.DELTA_S, K=self.K, include_overflow=True,
        )
        return G[0]

    def test_near_branch_mass_follows_one_minus_gamma(self):
        for gamma_value in (0.95, 0.70, 0.40):
            G0 = self._returns(gamma_value)
            expected = (1.0 - self.LAM) * (1.0 - gamma_value)
            self.assertAlmostEqual(
                float(G0[self.NEAR_BIN]), expected, places=5,
                msg=(f"gamma={gamma_value}: near-branch mass {float(G0[self.NEAR_BIN]):.6f} "
                     f"!= (1-lam)(1-gamma) = {expected:.6f}. gamma is not controlling "
                     f"the near/far split of the renewal mixture."),
            )

    def test_future_branch_mass_follows_the_new_identity(self):
        for gamma_value in (0.95, 0.70, 0.40):
            G0 = self._returns(gamma_value)
            expected = (1.0 - self.LAM) * gamma_value + self.LAM
            self.assertAlmostEqual(float(G0[self.FAR_BIN]), expected, places=5)

    def test_old_lambda_reparameterization_identity_is_broken(self):
        # The defect made the bootstrap-branch mass exactly
        # lam*gamma / ((1-lam) + lam*gamma), matched to float32 precision.
        for gamma_value in (0.95, 0.70, 0.40):
            G0 = self._returns(gamma_value)
            lam_eff = (self.LAM * gamma_value) / ((1.0 - self.LAM) + self.LAM * gamma_value)
            self.assertNotAlmostEqual(
                float(G0[self.FAR_BIN]), lam_eff, places=4,
                msg=(f"gamma={gamma_value}: bootstrap-branch mass still equals "
                     f"lam*gamma/((1-lam)+lam*gamma) = {lam_eff:.6f}; gamma is still "
                     f"absorbed into lambda."),
            )

    def test_gamma_changes_the_target_shape(self):
        near_one = self._returns(0.95)
        small = self._returns(0.40)
        cramer = float(torch.sum((torch.cumsum(near_one[:self.K], 0)
                                  - torch.cumsum(small[:self.K], 0)) ** 2))
        self.assertGreater(cramer, 1e-2)

    def test_mass_is_conserved_without_renormalization(self):
        for gamma_value in (1.0, 0.95, 0.70, 0.40, 0.0):
            self.assertAlmostEqual(float(self._returns(gamma_value).sum()), 1.0, places=5)


class TestLambdaEndpoints(unittest.TestCase):
    """
    lambda = 1 must be the exact projected Monte-Carlo target and lambda = 0 the
    exact one-step renewal target. Those two endpoints are what make the lambda
    spectrum a bias-variance axis at all, and Kill Criterion 2 is stated in terms
    of them.
    """

    K = 40
    DELTA_S = 1.0

    def _setup(self):
        L = 4
        dts = torch.tensor([1.0, 2.0, 1.0, 3.0])
        torch.manual_seed(3)
        raw = torch.rand(L, self.K + 1)
        pmfs = raw / raw.sum(-1, keepdim=True)
        surv = torch.full((L, self.K), 0.9)
        return pmfs, surv, dts

    def test_lambda_one_is_projected_monte_carlo(self):
        pmfs, surv, dts = self._setup()
        tte = 8.0                       # r = 8 - [1,3,4,7] = [7,5,4,1], all > 0
        G, _ = compute_lambda_returns(
            pmfs, surv, dts, tte=tte, event=True,
            lam=1.0, delta_s=self.DELTA_S, K=self.K, include_overflow=True,
        )
        # Build the Monte-Carlo target independently: a Dirac at the true residual
        # time, carried back one gap at a time.
        expected = localized_projected_dirac(1.0, self.DELTA_S, self.K, include_overflow=True)
        gaps = interval_gaps(dts)
        for j in range(2, -1, -1):
            expected = categorical_projection_shift(
                expected, float(gaps[j]), self.DELTA_S, self.K, include_overflow=True
            )
            torch.testing.assert_close(G[j], expected, atol=1e-6, rtol=0)

    def test_lambda_zero_is_the_one_step_renewal_target(self):
        pmfs, surv, dts = self._setup()
        G, _ = compute_lambda_returns(
            pmfs, surv, dts, tte=8.0, event=True,
            lam=0.0, delta_s=self.DELTA_S, K=self.K, include_overflow=True,
        )
        gaps = interval_gaps(dts)
        for j in range(3):
            expected, _ = one_step_renewal_target(
                pmfs[j], pmfs[j + 1], float(gaps[j]), float(gaps[j]),
                self.DELTA_S, self.K, include_overflow=True,
            )
            torch.testing.assert_close(G[j], expected, atol=1e-6, rtol=0)

    def test_compounded_placement_breaks_the_monte_carlo_endpoint(self):
        """Documents the tension recorded in deviation log A-15."""
        pmfs, surv, dts = self._setup()
        kw = dict(delta_s=self.DELTA_S, K=self.K, include_overflow=True)
        boot, _ = compute_lambda_returns(pmfs, surv, dts, 8.0, True, lam=1.0,
                                         gamma_placement="bootstrap", **kw)
        comp, _ = compute_lambda_returns(pmfs, surv, dts, 8.0, True, lam=1.0,
                                         gamma_placement="compounded", **kw)
        self.assertFalse(torch.allclose(boot[0], comp[0], atol=1e-4))
        self.assertAlmostEqual(float(comp[0].sum()), 1.0, places=5)


class TestArmsAreDistinct(unittest.TestCase):
    """
    Each preregistered arm must actually change the target. If an arm produces the
    same target as `full`, a null ablation result says nothing about the mechanism.
    """

    K = 30
    DELTA_S = 1.0

    def test_each_arm_differs_from_full(self):
        L = 4
        # Gaps far from delta_s so the unit-step arms are genuinely different.
        dts = torch.tensor([1.0, 4.0, 0.5, 6.0])
        torch.manual_seed(5)
        raw = torch.rand(L, self.K + 1)
        pmfs = raw / raw.sum(-1, keepdim=True)
        surv = torch.full((L, self.K), 0.9)
        kw = dict(delta_s=self.DELTA_S, K=self.K, include_overflow=True, lam=0.6)

        full, _ = compute_lambda_returns(pmfs, surv, dts, 12.0, False, arm="full", **kw)
        for arm in ("arm_a1_discount", "arm_a2_shift", "count_geometric"):
            other, _ = compute_lambda_returns(pmfs, surv, dts, 12.0, False, arm=arm, **kw)
            diff = float(torch.sum((torch.cumsum(full[:, :self.K], -1)
                                    - torch.cumsum(other[:, :self.K], -1)) ** 2))
            self.assertGreater(diff, 1e-6, msg=f"arm {arm} produces the same target as full")
            self.assertAlmostEqual(float(other[0].sum()), 1.0, places=5)

    def test_unknown_arm_raises(self):
        with self.assertRaises(ValueError):
            compute_lambda_returns(
                torch.rand(2, self.K + 1), torch.full((2, self.K), 0.9),
                torch.ones(2), 2.0, False, arm="nope",
                delta_s=self.DELTA_S, K=self.K, include_overflow=True,
            )


class TestMassConservationWithoutRenormalization(unittest.TestCase):
    """
    The repaired recursion must conserve mass by construction. This currently passes
    only because of the explicit `G_j / sum(G_j)`; after that line is deleted it
    becomes a real invariant.
    """

    def test_returns_sum_to_one_across_lambda_and_gamma(self):
        K, delta_s = 30, 2.5
        dts = torch.tensor([1.0, 3.5, 2.0, 4.0])
        L = 4

        for gamma_value in (1.0, 0.7, 0.3):
            for lam in (0.0, 0.3, 0.6, 1.0):
                G, _ = compute_multistep_lambda_returns(
                    _uniform_pmfs(L, K), _flat_survival(gamma_value, L, K), dts,
                    torch.zeros(L), tte=10.5, tau_event=110.5,
                    lam=lam, delta_s=delta_s, K=K,
                )
                for j in range(L):
                    self.assertAlmostEqual(
                        float(G[j].sum()), 1.0, places=5,
                        msg=f"mass leak at j={j}, gamma={gamma_value}, lam={lam}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
