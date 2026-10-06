"""Encoder GRU para sequência tabular (CSV / clínico)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularTemporalEncoder(nn.Module):
    """Fluxo tabular: (B, T, F_tab) -> (B, H)."""

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        out, _ = self.gru(h)
        return self.norm(out[:, -1, :])
