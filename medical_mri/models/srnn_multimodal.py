"""
SRNN Multimodal — fusão visual + tabular + radiomics com contexto temporal GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_rnn.models.visual_encoder import WeekVisualEncoder3D


class SRNNMultimodalCore(nn.Module):
    """
    Fluxo visual 3D + fluxo tabular GRU + radiomics -> fusão SRNN -> embedding global.
    """

    def __init__(
        self,
        visual_channels: int,
        tabular_dim: int,
        radiomics_dim: int,
        visual_embed: int = 64,
        tabular_hidden: int = 64,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.visual_enc = WeekVisualEncoder3D(visual_channels, visual_embed)
        self.tab_gru = nn.GRU(tabular_dim, tabular_hidden, num_layers=1, batch_first=True)
        self.rad_proj = nn.Linear(radiomics_dim, tabular_hidden)
        fuse_in = visual_embed + tabular_hidden + tabular_hidden
        self.temporal_gru = nn.GRU(fuse_in, fusion_hidden, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(fusion_hidden)
        self.context_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_hidden),
        )

    def encode_sequence(
        self,
        visual_volumes: torch.Tensor,
        tabular_seq: torch.Tensor,
        radiomics_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        visual_volumes: (B, T, C, D, H, W)
        tabular_seq: (B, T, F_tab)
        radiomics_seq: (B, T, F_rad)
        """
        b, t, c, d, h, w = visual_volumes.shape
        vis_steps = [self.visual_enc(visual_volumes[:, i]) for i in range(t)]
        vis_seq = torch.stack(vis_steps, dim=1)

        tab_out, _ = self.tab_gru(tabular_seq)
        rad_seq = torch.relu(self.rad_proj(radiomics_seq))

        fused = torch.cat([vis_seq, tab_out, rad_seq], dim=-1)
        out, _ = self.temporal_gru(fused)
        ctx = self.norm(out[:, -1, :])
        return self.context_head(ctx)
