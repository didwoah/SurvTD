"""
D15 regression: the per-visit `events` array is an all-zero placeholder in every
cohort, so any arm that derives `has_event` from it alone treats 100% of events as
censored. This is what silently crippled SurvTD and DeepTCSR while leaving
Person-Period and Dynamic-DeepHit correct.
"""
import unittest
import torch

from src.data.cohorts import COHORTS
from src.models.survtd import SurvTDModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel


class TestEventFlagPropagation(unittest.TestCase):
    def test_loaders_leave_the_per_visit_events_array_empty(self):
        """Characterizes the convention the defect depended on. If a loader ever starts
        populating `events`, this fails and the has_event rule can be simplified."""
        for name in ("synthetic_icu",):
            cd = COHORTS[name].load(seed=42)
            any_visit_flag = any(bool(torch.any(p["events"] > 0.5).item()) for p in cd.train)
            n_event_traj = sum(1 for p in cd.train if float(p["event"]) > 0.5)
            self.assertGreater(n_event_traj, 0, f"{name}: no event trajectories to test")
            self.assertFalse(any_visit_flag,
                             f"{name}: `events` is documented as an all-zero placeholder")

    def _has_event_seen_by(self, model, patient):
        """Runs the arm's own loss and reports whether it placed the event branch."""
        tte = float(patient["tte"])
        tau = tte if float(patient["event"]) > 0.5 else tte + 100.0
        loss_event = model.compute_loss_trajectory(
            patient["features"], patient["dts"], patient["events"], tte, tau,
            mask=patient["mask"], event=float(patient["event"]) > 0.5)
        # An event trajectory scored as censored produces a different anchor value than
        # one scored as an event; compare against the deliberately censored call.
        loss_as_censored = model.compute_loss_trajectory(
            patient["features"], patient["dts"], patient["events"], tte, tte + 100.0,
            mask=patient["mask"], event=False)
        return abs(float(loss_event) - float(loss_as_censored)) > 1e-9

    def test_survtd_sees_events_despite_the_empty_placeholder(self):
        cd = COHORTS["synthetic_icu"].load(seed=42)
        spec = COHORTS["synthetic_icu"]
        ev = next(p for p in cd.train if float(p["event"]) > 0.5)
        m = SurvTDModel(input_dim=cd.input_dim, num_bins=spec.num_bins,
                        delta_s=spec.delta_s, alpha_anchor=1.0, dropout=0.0)
        m.backbone.set_empirical_mean(cd.x_mean); m.eval()
        self.assertTrue(self._has_event_seen_by(m, ev),
                        "SurvTD scored an event trajectory identically to a censored one")

    def test_deeptcsr_sees_events_despite_the_empty_placeholder(self):
        cd = COHORTS["synthetic_icu"].load(seed=42)
        spec = COHORTS["synthetic_icu"]
        ev = next(p for p in cd.train if float(p["event"]) > 0.5)
        m = DeepTCSRClampedModel(input_dim=cd.input_dim, num_bins=spec.num_bins,
                                 delta_s=spec.delta_s, alpha_anchor=1.0, dropout=0.0)
        m.backbone.set_empirical_mean(cd.x_mean); m.eval()
        self.assertTrue(self._has_event_seen_by(m, ev),
                        "DeepTCSR scored an event trajectory identically to a censored one")


if __name__ == "__main__":
    unittest.main()


class TestBothCallingConventions(unittest.TestCase):
    """The two live callers set the flag differently and both must work.

    - The main pipeline (`trainer.py`) leaves `events` all-zero and passes
      `event=bool(p['event'] > 0.5)` explicitly.
    - The NASA benchmark (`benchmark_nasa_parity.py`) builds its own `events_seq`
      with the last visit set, and passes `tau_event=tte` unconditionally -- so any
      rule that infers the flag from `tau_event` marks its 50% censored units as
      events. That inference is why it must not exist.
    """

    def _model(self, cd, spec):
        m = SurvTDModel(input_dim=cd.input_dim, num_bins=spec.num_bins,
                        delta_s=spec.delta_s, alpha_anchor=1.0, dropout=0.0)
        m.backbone.set_empirical_mean(cd.x_mean)
        m.eval()   # dropout would otherwise make two identical calls differ
        return m

    def test_explicit_flag_beats_an_empty_events_array(self):
        spec = COHORTS["synthetic_icu"]; cd = spec.load(seed=42)
        m = self._model(cd, spec)
        p = next(q for q in cd.train if float(q["event"]) > 0.5)
        tte = float(p["tte"])
        as_event = m.compute_loss_trajectory(p["features"], p["dts"], p["events"], tte,
                                             tte, mask=p["mask"], event=True)
        as_censored = m.compute_loss_trajectory(p["features"], p["dts"], p["events"], tte,
                                                tte, mask=p["mask"], event=False)
        self.assertNotAlmostEqual(float(as_event), float(as_censored), places=6)

    def test_tau_event_alone_never_decides(self):
        """NASA passes tau_event == tte for censored units too."""
        spec = COHORTS["synthetic_icu"]; cd = spec.load(seed=42)
        m = self._model(cd, spec)
        p = next(q for q in cd.train if float(q["event"]) < 0.5)
        tte = float(p["tte"])
        censored_seq = torch.zeros(len(p["dts"]))
        with_bad_tau = m.compute_loss_trajectory(p["features"], p["dts"], censored_seq,
                                                 tte, tte, mask=p["mask"])
        with_good_tau = m.compute_loss_trajectory(p["features"], p["dts"], censored_seq,
                                                  tte, tte + 100.0, mask=p["mask"])
        self.assertAlmostEqual(float(with_bad_tau), float(with_good_tau), places=6,
                               msg="tau_event is being used to infer the event flag")

    def test_nasa_style_visit_flag_is_honoured(self):
        spec = COHORTS["synthetic_icu"]; cd = spec.load(seed=42)
        m = self._model(cd, spec)
        p = next(q for q in cd.train if float(q["event"]) > 0.5)
        tte = float(p["tte"])
        seq = torch.zeros(len(p["dts"])); seq[-1] = 1.0
        flagged = m.compute_loss_trajectory(p["features"], p["dts"], seq, tte, tte,
                                            mask=p["mask"])
        censored = m.compute_loss_trajectory(p["features"], p["dts"],
                                             torch.zeros(len(p["dts"])), tte, tte,
                                             mask=p["mask"])
        self.assertNotAlmostEqual(float(flagged), float(censored), places=6)
