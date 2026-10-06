"""
Cabeça Stacked Refiner multimodal.

Estende a ideia do Ridge sobre [GRU, SVR, RF] adicionando canal tabular:
  [pred_gru, pred_svr, pred_rf, pred_tabular_context, bias]
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MultimodalStackedRefinerHead:
    """
    Meta-regressor leve (Ridge) — mesma filosofia de train_stacked_linear_refiner,
    porém com feature tabular agregada por semana (não substitui o refiner legado).
    """

    def __init__(self, alpha: float = 1e-2, blend_with_mean: float = 0.75):
        self.alpha = float(alpha)
        self.blend_with_mean = float(blend_with_mean)
        self.coefficients: np.ndarray | None = None
        self.source_channels: list[str] = []

    def fit(
        self,
        channels: dict[str, np.ndarray],
        target: np.ndarray,
        tabular_context: np.ndarray,
        mask: np.ndarray | None = None,
        sample_size: int = 250_000,
        seed: int = 42,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        keys = ["rnn_gru", "svr", "random_forest"]
        for k in keys:
            if k not in channels:
                raise KeyError(f"Canal obrigatório ausente: {k}")

        shapes = [channels[k].shape for k in keys]
        if len(set(shapes)) != 1 or shapes[0] != target.shape:
            raise ValueError(f"Incompatibilidade de shapes: {shapes} vs target {target.shape}")

        flat = {k: np.nan_to_num(channels[k].reshape(-1), nan=0.0) for k in keys}
        flat_target = np.nan_to_num(target.reshape(-1), nan=0.0)
        n = flat_target.size

        tab_scalar = float(np.mean(tabular_context))
        tab_channel = np.full(n, tab_scalar, dtype=np.float32)

        weights = np.ones(n, dtype=np.float32)
        p10 = float(np.percentile(flat_target, 10.0))
        weights += 0.35 * (flat_target > p10).astype(np.float32)
        if mask is not None:
            weights += 8.0 * (mask.reshape(-1) > 0.5).astype(np.float32)

        rng = np.random.default_rng(seed)
        sample_n = min(int(sample_size), n)
        prob = weights / max(float(weights.sum()), 1e-8)
        idx = rng.choice(n, size=sample_n, replace=False, p=prob)

        x = np.stack([
            flat["rnn_gru"][idx],
            flat["svr"][idx],
            flat["random_forest"][idx],
            tab_channel[idx],
            np.ones(sample_n, dtype=np.float32),
        ], axis=1).astype(np.float32)
        y = flat_target[idx].astype(np.float32)
        w = np.sqrt(weights[idx])

        mu = x[:, :4].mean(axis=0, keepdims=True)
        sd = x[:, :4].std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        y_mu, y_sd = float(y.mean()), float(y.std())
        if y_sd < 1e-6:
            y_sd = 1.0

        xs = (x.copy())
        xs[:, :4] = (xs[:, :4] - mu) / sd
        ys = (y - y_mu) / y_sd
        xw, yw = xs * w[:, None], ys * w
        eye = np.eye(5, dtype=np.float32)
        eye[-1, -1] = 0.0
        coef = np.linalg.solve((xw.T @ xw) + (self.alpha * eye), xw.T @ yw).astype(np.float32)

        full_x = np.stack([
            flat["rnn_gru"],
            flat["svr"],
            flat["random_forest"],
            tab_channel,
            np.ones(n, dtype=np.float32),
        ], axis=1).astype(np.float32)
        full_x[:, :4] = (full_x[:, :4] - mu) / sd
        pred = ((full_x @ coef) * y_sd + y_mu).reshape(target.shape).astype(np.float32)

        mean_img = np.mean([channels[k] for k in keys], axis=0).astype(np.float32)
        pred = (self.blend_with_mean * pred + (1.0 - self.blend_with_mean) * mean_img).astype(np.float32)

        self.coefficients = coef
        self.source_channels = keys + ["tabular_context", "bias"]
        info = {
            "coefficients": coef[:4].tolist(),
            "intercept": float(coef[4]),
            "tabular_context_mean": tab_scalar,
            "alpha": self.alpha,
            "blend_with_mean": self.blend_with_mean,
            "sample_size": int(sample_n),
        }
        return pred, info
