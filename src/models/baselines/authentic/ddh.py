"""
Dynamic-DeepHit (Lee, Yoon, Zame, van der Schaar; IEEE TBME 2020) — faithful port.

Mirrors the vendored upstream clone at `baselines/dynamic_deephit_pytorch/`
(MIT, https://github.com/Jeanselme/DynamicDeepHit):

    ddh/ddh_torch.py:6-98    DynamicDeepHitTorch  -> AuthenticDynamicDeepHit
    ddh/utils.py:create_nn   MLP builder          -> create_nn (verbatim)
    ddh/losses.py:1-75       total_loss           -> authentic_ddh_total_loss
    ddh/ddh_api.py:88-103    discretize           -> discretize_absolute_times

What is preserved
-----------------
* `nn.LSTM(..., bias=False)` — the upstream constructor really does disable bias.
* The longitudinal / attention / cause-specific MLPs, all built by `create_nn` with
  its upstream defaults `dropout=0.6, layers=[100, 100], activation='ReLU'`.
* The attention masking: future positions set to `-1e10` before the softmax, and
  attention forced to 0 for subjects with a single observation.
* The final `Softmax(dim=-1)` over `output_dim` bins, so the head emits a PMF over the
  **absolute** event-time axis.
* `total_loss = (1-alpha-beta) * longitudinal + alpha * ranking + beta * NLL`, with the
  same `+1e-10` log guards and the same `O(N^2)` ranking loop.

Two properties worth stating plainly, because they are the comparison
--------------------------------------------------------------------
1. **Dynamic-DeepHit never sees `dts`.** `forward` takes only `x`; there is no notion
   of inter-visit duration anywhere in the model. Irregular sampling reaches it only
   through whatever the features happen to encode. That gap is exactly what SurvTD
   claims to close, so this port keeps it rather than quietly repairing it.
2. **Its bins live on the absolute time axis**, not on residual time: `discretize`
   histograms the raw event/censoring times into `split` bins and the head predicts a
   PMF over those. Converting to this project's residual-time contract is the
   adapter's job (see `adapters.py`), not the model's.

Deviations from upstream, and why
---------------------------------
* Upstream pads short sequences with `NaN` and recovers the last observation with
  `(~isnan(x[:,:,0])).sum(axis=1) - 1`. This port keeps that mechanism exactly, so a
  caller may pass NaN-padded batches; when a single unpadded trajectory is passed
  (this project's convention) the mask is all-False and the last index is `L-1`,
  which is the same answer.
* Upstream's `ddh_api.DynamicDeepHit` wraps the module with fit/predict, an internal
  validation split and `DeepSurvivalMachines` padding helpers. That wrapper is not
  ported: this project supplies its own splits, and importing it would pull in the
  vendored `DeepSurvivalMachines` tree. The architecture and the objective — the parts
  that decide the number — are ported verbatim.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def create_nn(inputdim, outputdim, dropout=0.6, layers=[100, 100],
              activation="ReLU", no_activation_last=False):
    """Verbatim from `baselines/dynamic_deephit_pytorch/ddh/utils.py`.

    Kept including the mutable default `layers=[100, 100]` and the dropout-first
    ordering, so the parameter count and the initialisation match upstream exactly.
    """
    modules = []
    if dropout > 0:
        modules.append(nn.Dropout(p=dropout))

    if activation == "ReLU6":
        act = nn.ReLU6()
    elif activation == "ReLU":
        act = nn.ReLU()
    elif activation == "SeLU":
        act = nn.SELU()
    elif activation == "Tanh":
        act = nn.Tanh()

    prevdim = inputdim
    for hidden in layers + [outputdim]:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        modules.append(act)
        prevdim = hidden

    if no_activation_last:
        modules = modules[:-1]

    return nn.Sequential(*modules)


def discretize_absolute_times(t, split, split_time=None):
    """Port of `ddh_api.DynamicDeepHit.discretize` (`ddh_api.py:88-103`).

    Bins are learned from the training event/censoring times by
    `np.histogram(concat(t), split - 1)` and applied with
    `np.digitize(..., right=True) - 1`. They are **absolute-time** bins, shared across
    subjects, and must be fit on the training split only.

    Args:
        t: sequence of arrays (or an array) of times
        split: number of bins
        split_time: bin edges from a previous call; fit fresh when None
    Returns:
        (discretised, split_time)
    """
    if split_time is None:
        _, split_time = np.histogram(np.concatenate([np.atleast_1d(t_) for t_ in t]),
                                     split - 1)
    disc = np.array([np.digitize(np.atleast_1d(t_), split_time, right=True) - 1
                     for t_ in t], dtype=object)
    return disc, split_time


class AuthenticDynamicDeepHit(nn.Module):
    """Port of `ddh/ddh_torch.py::DynamicDeepHitTorch`.

    Args mirror upstream. `output_dim` is upstream's `split` (number of absolute-time
    bins). `forward` returns `(longitudinal_prediction, outcomes)` where `outcomes` is
    a list of per-risk PMFs over the absolute-time bins — the upstream contract, not
    this project's 4-tuple. `adapters.py` bridges the two.
    """

    def __init__(self, input_dim, output_dim, layers_rnn=1, hidden_rnn=10,
                 long_param={}, att_param={}, cs_param={}, typ="LSTM", risks=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.risks = risks
        self.typ = typ

        # RNN over the longitudinal record. bias=False is upstream's choice.
        if self.typ == "LSTM":
            self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn,
                                     bias=False, batch_first=True)
        elif self.typ == "RNN":
            self.embedding = nn.RNN(input_dim, hidden_rnn, layers_rnn,
                                    bias=False, batch_first=True, nonlinearity="relu")
        elif self.typ == "GRU":
            self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn,
                                    bias=False, batch_first=True)
        else:
            raise ValueError(f"unknown rnn type {typ!r}; upstream allows LSTM/RNN/GRU")

        self.longitudinal = create_nn(hidden_rnn, input_dim,
                                      no_activation_last=True, **long_param)
        self.attention = create_nn(input_dim + hidden_rnn, 1,
                                   no_activation_last=True, **att_param)
        self.attention_soft = nn.Softmax(1)      # over the temporal dimension

        self.cause_specific = nn.ModuleList([
            create_nn(input_dim + hidden_rnn, output_dim,
                      no_activation_last=True, **cs_param)
            for _ in range(self.risks)
        ])

        self.soft = nn.Softmax(dim=-1)           # over all observed output bins

    def forward(self, x):
        device = x.device

        # NaN marks unobserved padding upstream; an unpadded trajectory yields an
        # all-False mask and last index L-1, which is the same answer.
        x = x.clone()
        inputmask = torch.isnan(x[:, :, 0])
        x[inputmask] = 0
        hidden, _ = self.embedding(x)

        longitudinal_prediction = self.longitudinal(hidden)

        last_observations = ((~inputmask).sum(axis=1) - 1)
        last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
        index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

        last = index == last_observations_idx
        x_last = x[last]

        concatenation = torch.cat(
            [hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1)

        attention = self.attention(concatenation).squeeze(-1)
        attention[index >= last_observations_idx] = -1e10
        attention[last_observations > 0] = self.attention_soft(
            attention[last_observations > 0])
        attention[last_observations == 0] = 0

        attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
        hidden_attentive = torch.sum(attention * hidden, axis=1)
        hidden_attentive = torch.cat([hidden_attentive, x_last], 1)

        outcomes = [cs_nn(hidden_attentive) for cs_nn in self.cause_specific]
        outcomes = torch.cat(outcomes, dim=1)
        outcomes = self.soft(outcomes)
        outcomes = [outcomes[:, i * self.output_dim:(i + 1) * self.output_dim]
                    for i in range(self.risks)]
        return longitudinal_prediction, outcomes


# --------------------------------------------------------------------------------
# Losses — verbatim from `baselines/dynamic_deephit_pytorch/ddh/losses.py`
# --------------------------------------------------------------------------------

def negative_log_likelihood(outcomes, cif, t, e):
    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        censored_cif += cif[k][e == 0][torch.arange((e == 0).sum()), t[e == 0]]
        selection = e == (k + 1)
        loss += torch.sum(torch.log(
            ok[selection][torch.arange((selection).sum()), t[selection]] + 1e-10))
    loss += torch.sum(torch.log(1 - censored_cif + 1e-10))
    return - loss / len(outcomes)


def ranking_loss(cif, t, e, sigma):
    loss = 0
    for k, cifk in enumerate(cif):
        for ci, ti in zip(cifk[e - 1 == k], t[e - 1 == k]):
            if torch.sum(t > ti) > 0:
                loss += torch.mean(torch.exp(
                    (cifk[t > ti][torch.arange((t > ti).sum()), ti] - ci[ti])) / sigma)
    return loss / len(cif)


def longitudinal_loss(longitudinal_prediction, x):
    length = (~torch.isnan(x[:, :, 0])).sum(axis=1) - 1
    device = x.device

    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False

    return torch.nn.MSELoss(reduction="mean")(
        longitudinal_prediction[prediction_mask], x[observation_mask])


def authentic_ddh_total_loss(model, x, t, e, alpha, beta, sigma):
    """Port of `ddh/losses.py::total_loss`.

    `t` is the **discretised absolute** event/censoring bin index and `e` the risk
    label (0 = censored). Upstream weights: `(1 - alpha - beta)` longitudinal,
    `alpha` ranking, `beta` NLL.
    """
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    return ((1 - alpha - beta) * longitudinal_loss(longitudinal_prediction, x)
            + alpha * ranking_loss(cif, t, e, sigma)
            + beta * negative_log_likelihood(outcomes, cif, t, e))
