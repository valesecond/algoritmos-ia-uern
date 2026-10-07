# -*- coding: utf-8 -*-
"""
Redes temporais para predição do próximo perímetro cefálico (HC) fetal.

Famílias: Vanilla RNN, LSTM, GRU.
Uso didático / demos — treinado sobre séries construídas a partir do HC18.
"""
from __future__ import annotations

import torch.nn as nn


class TemporalHC(nn.Module):
    """Preditor sequencial: histórico (features + HC + ΔHC) → próximo HC."""

    def __init__(
        self,
        kind: str = "lstm",
        input_dim: int = 34,
        hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        kind = kind.lower().strip()
        self.kind = kind
        if kind == "rnn":
            self.rnn = nn.RNN(input_dim, hidden, num_layers=num_layers, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        elif kind == "gru":
            self.rnn = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"kind deve ser rnn|lstm|gru, recebido: {kind}")
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(kind: str, **kwargs) -> TemporalHC:
    return TemporalHC(kind=kind, **kwargs)
