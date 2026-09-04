"""
Numerical fidelity of the Dynamic-DeepHit port against its vendored upstream.

A port that cannot reproduce its source is not a port, so this loads
`baselines/dynamic_deephit_pytorch/ddh/` and asserts bit-level agreement on the
forward pass, the objective and the discretisation. If someone "improves" the port
later, these fail.

Loading upstream is fiddly and the mechanics are the point of the helper below:
`ddh/__init__.py` imports `ddh_api`, which pulls the whole vendored
`DeepSurvivalMachines` tree, and `ddh/utils.py` imports `dsm.utilities`. So the
package is stubbed with an explicit `__path__` and the three modules are loaded in
dependency order (losses -> utils -> ddh_torch) with the DSM directory on `sys.path`.
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np
import torch

from src.models.baselines.authentic.ddh import (
    AuthenticDynamicDeepHit,
    authentic_ddh_total_loss,
    discretize_absolute_times,
)

# Module-level, not class attributes: a plain function stored on a class becomes a
# bound method, which would silently pass the TestCase as upstream's `model` argument.
_UPSTREAM_MODEL = None
_UPSTREAM_TOTAL_LOSS = None

DDH_DIR = "baselines/dynamic_deephit_pytorch/ddh"
DSM_DIR = "baselines/dynamic_deephit_pytorch/DeepSurvivalMachines"


def load_upstream():
    """Import the vendored upstream modules without triggering `ddh/__init__.py`."""
    sys.path.insert(0, DSM_DIR)
    try:
        pkg = types.ModuleType("ddh")
        pkg.__path__ = [DDH_DIR]
        sys.modules["ddh"] = pkg

        def _load(name, filename):
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(DDH_DIR, filename))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(pkg, name.split(".")[-1], module)
            return module

        losses = _load("ddh.losses", "losses.py")
        _load("ddh.utils", "utils.py")
        ddh_torch = _load("ddh.ddh_torch", "ddh_torch.py")
        return ddh_torch.DynamicDeepHitTorch, losses.total_loss
    finally:
        sys.path.remove(DSM_DIR)


@unittest.skipUnless(os.path.isdir(DDH_DIR), f"vendored {DDH_DIR} not present")
class TestAuthenticDDHFidelity(unittest.TestCase):
    B, L, D, K = 6, 9, 15, 12
    KW = dict(input_dim=15, output_dim=12, layers_rnn=1, hidden_rnn=10,
              typ="LSTM", risks=1)

    @classmethod
    def setUpClass(cls):
        global _UPSTREAM_MODEL, _UPSTREAM_TOTAL_LOSS
        _UPSTREAM_MODEL, _UPSTREAM_TOTAL_LOSS = load_upstream()

    def _pair(self):
        torch.manual_seed(7)
        up = _UPSTREAM_MODEL(**self.KW)
        torch.manual_seed(7)
        port = AuthenticDynamicDeepHit(**self.KW)
        port.load_state_dict(up.state_dict())
        return up, port

    def _batch(self):
        """Includes NaN padding, which is upstream's own last-observation mechanism."""
        torch.manual_seed(0)
        x = torch.randn(self.B, self.L, self.D)
        x[0, 6:, :] = float("nan")
        x[3, 4:, :] = float("nan")
        return x

    def test_parameter_count_matches(self):
        up, port = self._pair()
        self.assertEqual(sum(p.numel() for p in up.parameters()),
                         sum(p.numel() for p in port.parameters()))

    def test_forward_is_bit_identical(self):
        up, port = self._pair()
        up.eval(); port.eval()          # upstream dropout defaults to 0.6
        x = self._batch()
        with torch.no_grad():
            long_up, out_up = up(x)
            long_pt, out_pt = port(x)
        self.assertEqual(float((long_up - long_pt).abs().max()), 0.0)
        for a, b in zip(out_up, out_pt):
            self.assertEqual(float((a - b).abs().max()), 0.0)

    def test_head_emits_a_normalised_pmf(self):
        _, port = self._pair()
        port.eval()
        with torch.no_grad():
            _, outcomes = port(self._batch())
        torch.testing.assert_close(outcomes[0].sum(-1), torch.ones(self.B), atol=1e-6,
                                   rtol=0)

    def test_total_loss_is_bit_identical(self):
        up, port = self._pair()
        up.train(); port.train()
        x = self._batch()
        t = torch.randint(0, self.K, (self.B,))
        e = torch.randint(0, 2, (self.B,))
        torch.manual_seed(11)
        a = _UPSTREAM_TOTAL_LOSS(up, x, t, e, alpha=0.1, beta=0.5, sigma=0.1)
        torch.manual_seed(11)
        b = authentic_ddh_total_loss(port, x, t, e, alpha=0.1, beta=0.5, sigma=0.1)
        self.assertEqual(float(a), float(b))

    def test_discretisation_matches_upstream(self):
        """`ddh_api.discretize`: histogram into `split - 1`, then digitize right=True."""
        raw = [np.array([10., 55., 120.]), np.array([5., 200.]), np.array([333.])]
        got, edges = discretize_absolute_times(raw, 8)
        _, expected_edges = np.histogram(np.concatenate(raw), 7)
        expected = [np.digitize(t, expected_edges, right=True) - 1 for t in raw]
        np.testing.assert_allclose(edges, expected_edges)
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)

    def test_port_ignores_dts_like_upstream(self):
        """Dynamic-DeepHit has no notion of inter-visit duration. Pinned deliberately:
        it is the gap SurvTD claims to close, so a later 'fix' here would quietly
        remove the comparison."""
        import inspect
        sig = inspect.signature(AuthenticDynamicDeepHit.forward)
        self.assertEqual(list(sig.parameters), ["self", "x"])


if __name__ == "__main__":
    unittest.main()
