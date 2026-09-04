"""
TCSR subprocess worker: JAX actually computes, and the contract holds.

Runs out of process on purpose. `tdsurv` needs JAX, and jaxlib 0.10.2 expects
numpy >= 2.0 while this project pins 1.26.4, so the worker installs a
`numpy.dtypes.StringDType` shim before importing jax. Importing that into the test
interpreter would contaminate every other test, and the shim is exactly the thing most
likely to be silently wrong -- so this exercises the real subprocess path.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

TCSR_DIR = "baselines/tcsr"
WORKER = os.path.join(TCSR_DIR, "tcsr_worker.py")
AUTHOR_PKL = "baselines/deep_tcsr/data/pbc-seqs.pkl"


@unittest.skipUnless(os.path.exists(WORKER) and os.path.exists(AUTHOR_PKL),
                     "vendored tcsr tree or authors' PBC2 pickle not present")
class TestTCSRWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build a bundle from the authors' own preprocessed PBC2, so a wrong number is
        caught against a known reference rather than against nothing."""
        with open(AUTHOR_PKL, "rb") as f:
            data = pickle.load(f)
        seqs, ts, cs = data["seqs"], data["ts"], data["cs"]
        n, horizon, _ = seqs.shape
        grid = np.arange(horizon, dtype=float)
        paths = np.concatenate(
            [np.tile(grid, (n, 1))[:, :, None], seqs], axis=-1)
        labels = np.stack([ts.astype(float), (~cs).astype(float)], axis=1)
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = os.path.join(cls.tmp, "bundle.pt")
        torch.save({
            "paths_train": paths[:250], "surv_labels_train": labels[:250],
            "paths_test": paths[250:], "surv_labels_test": labels[250:],
            "sampling_times": grid,
            "pred_times": np.array([0.0, 2.0, 5.0]),
            "eval_times": np.array([1.0, 2.0, 4.0]),
        }, cls.bundle)
        cls.n_test = n - 250

    def _run(self, arm):
        out = os.path.join(self.tmp, f"{arm}.json")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(WORKER),
             "--data", os.path.abspath(self.bundle), "--out", os.path.abspath(out),
             "--arm", arm],
            cwd=TCSR_DIR, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"worker failed:\n{proc.stderr[-2000:]}")
        with open(out) as f:
            return json.load(f)

    def test_all_three_author_arms_run_and_produce_valid_curves(self):
        for arm in ("initial_state", "landmark", "tcsr"):
            with self.subTest(arm=arm):
                res = self._run(arm)
                curves = np.array(res["surv_curves"])
                self.assertEqual(curves.shape, (self.n_test, 3, 3))
                self.assertTrue(bool((curves >= 0).all() and (curves <= 1).all()))
                self.assertTrue(bool(np.all(np.diff(curves, axis=-1) <= 1e-9)),
                                "survival must not increase with the horizon")
                self.assertTrue(res["params_finite"],
                                "JAX produced non-finite parameters -- the numpy shim "
                                "imports but does not compute")

    def test_lambda_convention_is_recorded(self):
        """tdsurv's lambda_ is inverted relative to SurvTD's. Reporting both without
        saying so would invert the reader's understanding of every ablation."""
        self.assertEqual(self._run("tcsr")["lambda_"], 0.0)
        self.assertEqual(self._run("landmark")["lambda_"], 1.0)
        self.assertIn("inverted", self._run("tcsr")["lambda_convention"])

    def test_arms_are_actually_different_models(self):
        """TD regularises toward smaller parameters; if the arms agreed exactly the
        config plumbing would be dead."""
        absmax = {a: self._run(a)["params_absmax"]
                  for a in ("initial_state", "landmark", "tcsr")}
        self.assertGreater(len(set(round(v, 6) for v in absmax.values())), 1)


if __name__ == "__main__":
    unittest.main()
