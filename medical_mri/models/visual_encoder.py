"""Encoder 3D por semana — fluxo visual MRI."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeekVisualEncoder3D(nn.Module):
    """(B, C, D, H, W) -> (B, embed_dim)."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.proj = nn.Linear(48 * 4 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.proj(h)
