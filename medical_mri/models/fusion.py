"""
Fusão multimodal: embeddings visuais + tabulares + radiomics.

Base conceitual: Stacked Refiner (melhor modelo atual) como meta-camada final.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_rnn.models.tabular_encoder import TabularTemporalEncoder


class VisualTemporalEncoder(nn.Module):
    """Fluxo visual resumido por semana (estatísticas / features derivadas)."""

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(out[:, -1, :])


class MultimodalFusionGRU(nn.Module):
    """
    Arquitetura em dois fluxos + fusão:
      visual (MRI) + tabular (CSV) + radiomics -> concat -> GRU -> cabeça FLAIR
    """

    def __init__(
        self,
        tabular_dim: int,
        visual_dim: int,
        radiomics_dim: int,
        hidden_size: int = 128,
        tabular_hidden: int = 64,
        visual_hidden: int = 64,
    ):
        super().__init__()
        self.tabular_enc = TabularTemporalEncoder(tabular_dim, hidden_size=tabular_hidden)
        self.visual_enc = VisualTemporalEncoder(visual_dim, hidden_size=visual_hidden)
        self.rad_proj = nn.Linear(radiomics_dim, visual_hidden)
        fuse_in = tabular_hidden + visual_hidden + visual_hidden
        self.fusion_gru = nn.GRU(fuse_in, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        tabular_seq: torch.Tensor,
        visual_seq: torch.Tensor,
        radiomics_seq: torch.Tensor,
    ) -> torch.Tensor:
        tab_emb = self.tabular_enc(tabular_seq)
        vis_emb = self.visual_enc(visual_seq)
        rad_emb = torch.relu(self.rad_proj(radiomics_seq[:, -1, :]))
        fused_step = torch.cat([tab_emb, vis_emb, rad_emb], dim=-1).unsqueeze(1)
        out, _ = self.fusion_gru(fused_step)
        return self.head(out[:, -1, :]).squeeze(-1)
