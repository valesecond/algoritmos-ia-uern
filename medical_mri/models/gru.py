"""GRU/RNN tumor-guided — Stacked Refiner original (substitui NARX)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from patient067.config import Patient067Config
from patient067.tumor_guided.data import TumorGuidedWeekData
from train_patient037_improved_pipeline import train_gru_predictor


def train_tumor_guided_gru(
    weeks: list[TumorGuidedWeekData],
    target_idx: int,
    seq_len: int,
    cfg: Patient067Config,
    device: str = "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """
    GRU voxel-wise sobre FLAIR longitudinal (registrado + skull-stripped).
    Mesma logica do Stacked Refiner original Patient-037.
    """
    base_weeks = [w.base for w in weeks]
    pred, info = train_gru_predictor(
        base_weeks,
        target_idx,
        seq_len,
        max_samples=cfg.gru_max_samples,
        epochs=cfg.gru_epochs,
        device=device,
    )
    info = dict(info)
    info["model_name"] = "gru_tumor_guided"
    info["tumor_guided"] = True
    info["uses_registered_data"] = bool(any(w.uses_registered for w in weeks))
    return pred, info
