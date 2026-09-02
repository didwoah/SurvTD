"""
Unit tests for all comparative survival models:
- SurvTDModel
- DeepTCSRClampedModel
- DynamicDeepHitModel
- PersonPeriodModel
"""

import unittest
import torch

from src.models.survtd import SurvTDModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel
from src.models.baselines.dynamic_deephit import DynamicDeepHitModel
from src.models.baselines.person_period import PersonPeriodModel


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.L = 5
        self.D = 4
        self.K = 20
        self.H = 16
        self.delta_s = 1.0

        self.sample_patient = {
            'features': torch.randn(self.L, self.D),
            'dts': torch.tensor([1.0, 2.5, 0.8, 3.2, 1.5]),
            'times': torch.tensor([1.0, 3.5, 4.3, 7.5, 9.0]),
            'events': torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            'tte': 9.0,
            'event': 1.0,
            'mask': torch.ones(self.L, self.D)
        }

    def test_survtd_step(self):
        model = SurvTDModel(input_dim=self.D, hidden_dim=self.H, num_bins=self.K, delta_s=self.delta_s)
        p = self.sample_patient
        loss = model.compute_loss_trajectory(
            p['features'], p['dts'], p['events'], p['tte'], p['tte'], mask=p['mask']
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.update_target_network()

    def test_deeptcsr_step(self):
        model = DeepTCSRClampedModel(input_dim=self.D, hidden_dim=self.H, num_bins=self.K, delta_s=self.delta_s)
        p = self.sample_patient
        loss = model.compute_loss_trajectory(
            p['features'], p['dts'], p['events'], p['tte'], p['tte'], mask=p['mask']
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.update_target_network()

    def test_dynamic_deephit_step(self):
        model = DynamicDeepHitModel(input_dim=self.D, hidden_dim=self.H, num_bins=self.K, delta_s=self.delta_s)
        batch = [self.sample_patient, {
            'features': torch.randn(self.L, self.D),
            'dts': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            'times': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            'events': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            'tte': 12.0,
            'event': 0.0,
            'mask': torch.ones(self.L, self.D)
        }]
        loss = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_person_period_step(self):
        model = PersonPeriodModel(input_dim=self.D, hidden_dim=self.H, num_bins=self.K, delta_s=self.delta_s)
        p = self.sample_patient
        loss = model.compute_loss(p['features'], p['dts'], p['events'], p['tte'], mask=p['mask'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
