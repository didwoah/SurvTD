"""
Continuous Sequence Backbones for Dynamic Survival Analysis:
1. GRU-D (Che et al., 2018): Continuous-time decay recurrent cell for irregular time-series.
2. ContinuousLSTM: Time-augmented LSTM with elapsed duration and observation timestamps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDCell(nn.Module):
    """
    Single-layer GRU-D cell implementing continuous-time decays for irregular observations.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Decay parameter networks for input and hidden state
        self.w_gamma_x = nn.Linear(1, input_dim)
        self.w_gamma_h = nn.Linear(1, hidden_dim)

        # Standard GRU gates taking [x_hat, m, delta_t]
        gru_input_dim = input_dim * 2 + 1
        self.w_z = nn.Linear(gru_input_dim + hidden_dim, hidden_dim)
        self.w_r = nn.Linear(gru_input_dim + hidden_dim, hidden_dim)
        self.w_h = nn.Linear(gru_input_dim + hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize decay weights with positive bias so decays start moderately
        nn.init.constant_(self.w_gamma_x.bias, 0.1)
        nn.init.constant_(self.w_gamma_h.bias, 0.1)
        for layer in [self.w_z, self.w_r, self.w_h]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, dt, m, h_prev, x_last, x_mean):
        """
        Args:
            x: (B, input_dim) current observation
            dt: (B, 1) elapsed duration since previous observation
            m: (B, input_dim) missingness mask (1 = observed, 0 = missing)
            h_prev: (B, hidden_dim) previous hidden state
            x_last: (B, input_dim) previous observed feature
            x_mean: (input_dim,) empirical dataset mean for imputation
        """
        # Continuous-time decay factors: gamma in (0, 1]
        gamma_x = torch.exp(-F.relu(self.w_gamma_x(dt)))
        gamma_h = torch.exp(-F.relu(self.w_gamma_h(dt)))

        # Decayed feature imputation toward empirical mean
        x_decayed = gamma_x * x_last + (1.0 - gamma_x) * x_mean.unsqueeze(0)
        x_hat = m * x + (1.0 - m) * x_decayed

        # Decayed hidden state
        h_hat = gamma_h * h_prev

        # Concatenate inputs for GRU gates: [x_hat, m, dt]
        inputs = torch.cat([x_hat, m, dt], dim=-1)
        combined = torch.cat([inputs, h_hat], dim=-1)

        z = torch.sigmoid(self.w_z(combined))
        r = torch.sigmoid(self.w_r(combined))

        candidate_input = torch.cat([inputs, r * h_hat], dim=-1)
        h_tilde = torch.tanh(self.w_h(candidate_input))

        h_new = (1.0 - z) * h_hat + z * h_tilde
        return h_new, x_hat


class GRUD(nn.Module):
    """
    Multi-layer GRU-D sequence encoder handling variable-length irregular time series.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.cell_layer0 = GRUDCell(input_dim, hidden_dim)
        self.higher_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        # Register empirical feature mean buffer
        self.register_buffer("x_mean", torch.zeros(input_dim))

    def set_empirical_mean(self, mean_tensor: torch.Tensor):
        self.x_mean.copy_(mean_tensor)

    def forward(self, x, dts, mask=None):
        """
        Args:
            x: (B, L, input_dim) time series features
            dts: (B, L) elapsed duration between visits (dt_0 can be 0.0 or 1.0)
            mask: (B, L, input_dim) optional missingness mask; defaults to all ones
        Returns:
            hidden_seq: (B, L, hidden_dim) hidden states at every step
        """
        B, L, D = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones_like(x)

        dts_expanded = dts.unsqueeze(-1)  # (B, L, 1)

        h_states = [torch.zeros(B, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        x_last = x[:, 0, :].clone()

        hidden_out = []

        for t in range(L):
            x_t = x[:, t, :]
            dt_t = dts_expanded[:, t, :]
            m_t = mask[:, t, :]

            # Layer 0 GRU-D cell
            h_0, x_hat_t = self.cell_layer0(x_t, dt_t, m_t, h_states[0], x_last, self.x_mean)
            h_states[0] = h_0

            # Update x_last for observed components
            x_last = m_t * x_t + (1.0 - m_t) * x_last

            # Propagate through higher standard GRU cells if num_layers > 1
            cur_input = self.dropout(h_0)
            for l_idx in range(1, self.num_layers):
                h_l = self.higher_cells[l_idx - 1](cur_input, h_states[l_idx])
                h_states[l_idx] = h_l
                cur_input = self.dropout(h_l)

            hidden_out.append(h_states[-1].unsqueeze(1))

        return torch.cat(hidden_out, dim=1)  # (B, L, hidden_dim)


class ContinuousLSTM(nn.Module):
    """
    Continuous-time LSTM concatenating elapsed duration and cumulative observation time.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Augmented input: [x, dt, t_cumulative]
        augmented_dim = input_dim + 2
        self.input_proj = nn.Linear(augmented_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, dts, mask=None, times=None):
        """
        Args:
            x: (B, L, input_dim)
            dts: (B, L)
            mask: optional mask
            times: (B, L) optional cumulative timestamps; computed from cumsum(dts) if None
        """
        B, L, _ = x.shape
        if times is None:
            times = torch.cumsum(dts, dim=1)

        dt_expanded = dts.unsqueeze(-1)
        t_expanded = times.unsqueeze(-1)

        if mask is not None:
            x = x * mask

        x_aug = torch.cat([x, dt_expanded, t_expanded], dim=-1)
        proj = F.relu(self.input_proj(x_aug))
        out, _ = self.lstm(proj)
        return self.layer_norm(out)


def build_backbone(backbone_type: str, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
    """
    Unified factory method for sequence backbones.
    """
    b_type = backbone_type.lower().strip()
    if b_type in ["grud", "gru-d"]:
        return GRUD(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif b_type in ["lstm", "continuous_lstm", "continuous-lstm"]:
        return ContinuousLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Choose 'grud' or 'continuous_lstm'.")
