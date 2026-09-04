"""
Phase A gate: our PBC2 cohort must be the one the baseline papers published on.

`src/data/pbc_loader.py` reads the Dynamic-DeepHit authors' own file
(`baselines/signature_survival/.../data/pbc2_cleaned.csv`), and the TCSR / DeepTCSR
authors' preprocessed copy (`baselines/deep_tcsr/data/pbc-seqs.pkl`) is vendored too.
These tests pin our loader against both so a preprocessing change cannot silently
move us off the literature cohort.
"""
import os
import pickle
import unittest

import numpy as np

from src.data.cohorts import COHORTS

AUTHOR_PKL = "baselines/deep_tcsr/data/pbc-seqs.pkl"
AUTHOR_CSV = ("baselines/signature_survival/competing_methods/"
              "Dynamic_DeepHit/data/pbc2_cleaned.csv")


class TestPBC2LiteratureFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(AUTHOR_PKL):
            raise unittest.SkipTest(f"vendored {AUTHOR_PKL} not present")
        with open(AUTHOR_PKL, "rb") as f:
            cls.author = pickle.load(f)
        cd = COHORTS["pbc"].load(seed=42)
        cls.patients = list(cd.train) + list(cd.val) + list(cd.test)
        cls.cd = cd

    def test_subject_count_matches(self):
        self.assertEqual(len(self.patients), self.author["seqs"].shape[0], 312)

    def test_event_rate_matches(self):
        """0.449 = 140/312. Both we and the authors count only label == 1 as an
        event; transplant (label == 2, 29 subjects) is censoring. That is defect
        X-04's declared policy, and this test is its independent confirmation."""
        ours = float(np.mean([float(p["event"]) for p in self.patients]))
        theirs = float(1.0 - self.author["cs"].mean())
        self.assertAlmostEqual(ours, theirs, places=3)
        self.assertAlmostEqual(ours, 140.0 / 312.0, places=3)

    def test_visit_counts_match_the_source_csv_exactly(self):
        """Our per-subject visit counts reproduce the raw CSV row-per-visit layout."""
        import pandas as pd
        ours = np.array(sorted(len(p["dts"]) for p in self.patients))
        raw = np.array(sorted(pd.read_csv(AUTHOR_CSV).groupby("id").size()))
        np.testing.assert_array_equal(ours, raw)
        self.assertEqual(int(ours.sum()), 1945)

    def test_author_terminal_index_convention(self):
        """The authors' `ts` is the index of the period in which the trajectory ends:
        the last observed visit (0-based) for a censored subject, and one step past it
        for an event subject. Pins the identity

            sum(ts) == total_visits - n_censored

        which is order-free, so it holds without aligning subject ids. Measured:
        ours - (ts + 1) is -1 for exactly 140 subjects (the events) and 0 for exactly
        172 (the censored). This is a convention difference, not a disagreement."""
        ts = self.author["ts"]
        n_censored = int(self.author["cs"].sum())
        total_visits = int(sum(len(p["dts"]) for p in self.patients))
        self.assertEqual(int(ts.sum()), total_visits - n_censored)
        self.assertEqual(n_censored, 172)

    def test_feature_dimension_matches(self):
        self.assertEqual(self.cd.input_dim, self.author["seqs"].shape[2], 15)

    def test_loader_reads_the_authors_csv(self):
        from src.data.pbc_loader import DEFAULT_CSV
        self.assertEqual(DEFAULT_CSV, AUTHOR_CSV)
        self.assertTrue(os.path.exists(DEFAULT_CSV))

    def test_our_clock_is_continuous_not_their_period_grid(self):
        """Recorded as a difference, not a defect. The authors discretise PBC2 onto a
        fixed 16-period grid; we keep the day clock (tte median ~328, max 744) and
        irregular inter-visit gaps, which is the setting SurvTD's claim is about. Any
        comparison must state which clock it is on."""
        tte = np.array([float(p["tte"]) for p in self.patients])
        self.assertGreater(tte.max(), 700)
        self.assertEqual(self.author["seqs"].shape[1], 16)


if __name__ == "__main__":
    unittest.main()
