"""
Unit tests for continuous sequence backbones:
- GRU-D forward/backward on irregular observation tensors with decay
- ContinuousLSTM forward/backward
"""

import unittest
import torch

from src.models.backbones import GRUD, ContinuousLSTM, build_backbone


class TestBackbones(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.L = 8
        self.D = 6
        self.H = 32
        self.x = torch.randn(self.B, self.L, self.D, requires_grad=True)
        self.dts = torch.abs(torch.randn(self.B, self.L)) + 0.1
        self.mask = (torch.rand(self.B, self.L, self.D) > 0.3).float()

    def test_grud_forward_backward(self):
        grud = GRUD(input_dim=self.D, hidden_dim=self.H, num_layers=2)
        out = grud(self.x, self.dts, self.mask)
        self.assertEqual(out.shape, (self.B, self.L, self.H))
        loss = torch.sum(out)
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertFalse(torch.isnan(self.x.grad).any())

    def test_continuous_lstm_forward_backward(self):
        x = torch.randn(self.B, self.L, self.D, requires_grad=True)
        lstm = ContinuousLSTM(input_dim=self.D, hidden_dim=self.H, num_layers=2)
        out = lstm(x, self.dts, self.mask)
        self.assertEqual(out.shape, (self.B, self.L, self.H))
        loss = torch.sum(out)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


if __name__ == "__main__":
    unittest.main()
