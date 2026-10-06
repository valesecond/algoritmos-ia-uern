"""Refinamento voxel 3D — melhora SSIM sobre predição Stacked + contexto multimodal."""

from __future__ import annotations

import torch
import torch.nn as nn


class VoxelResidualRefiner(nn.Module):
    """Prediz residual Δ sobre volume base: out = base + Δ."""

    def __init__(self, in_channels: int = 5, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
