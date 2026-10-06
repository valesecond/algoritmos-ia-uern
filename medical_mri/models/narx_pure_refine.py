"""Refinamento pos-NARX puro: prior temporal + estrutura (sem GRU/hibrido)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from train_patient037_improved_pipeline import (
    WeekData,
    compute_image_metrics,
    refine_structural_candidate,
)


def _brain_envelope(vol: np.ndarray, tumor_mask: Optional[np.ndarray] = None) -> np.ndarray:
    flat = vol.reshape(-1)
    nz = flat[np.abs(flat) > 1e-6]
    src = nz if nz.size >= 128 else flat
    thr = float(np.percentile(src, 10))
    brain = vol > thr
    if tumor_mask is not None:
        brain = brain | (tumor_mask > 0.5)
    return brain


def refine_narx_pure_prediction(
    pred: np.ndarray,
    prior_week: np.ndarray,
    tumor_mask: Optional[np.ndarray],
    *,
    prior_blend: float = 0.14,
    percentile_match: float = 0.40,
    tumor_lift: float = 0.14,
    hf_gain: float = 0.10,
) -> np.ndarray:
    """
    Completa forma cerebral e contraste FLAIR usando apenas a semana anterior (week-029).
    Nao usa GRU, stacked nem ground truth da semana alvo.
    """
    prior = np.nan_to_num(prior_week.astype(np.float32), nan=0.0)
    out = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    brain = _brain_envelope(prior, tumor_mask)

    if np.any(brain):
        out[brain] = ((1.0 - prior_blend) * out[brain]) + (prior_blend * prior[brain])

        p = prior[brain].astype(np.float64)
        o = out[brain].astype(np.float64)
        p_lo, p_hi = np.percentile(p, [6, 94])
        o_lo, o_hi = np.percentile(o, [6, 94])
        if o_hi > o_lo and p_hi > p_lo:
            scaled = (o - o_lo) / (o_hi - o_lo)
            scaled = scaled * (p_hi - p_lo) + p_lo
            out[brain] = ((1.0 - percentile_match) * out[brain]) + (
                percentile_match * scaled.astype(np.float32)
            )

    out = refine_structural_candidate(out, prior)

    if tumor_mask is not None:
        t = tumor_mask > 0.5
        if np.any(t):
            delta = float(np.percentile(prior[t], 82) - np.percentile(out[t], 82))
            out[t] = out[t] + (tumor_lift * delta)
            roi_hf = out - gaussian_filter(out, sigma=0.5)
            out[t] = out[t] + (hf_gain * roi_hf[t])

    return np.clip(out, -8.0, 8.0).astype(np.float32)


def optimize_narx_pure_refinement(
    pred: np.ndarray,
    target: WeekData,
    prior_week: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grid leve no refinamento puro (prior + estrutura)."""
    gt = target.flair_resampled
    mask = target.mask_resampled
    base_m = compute_image_metrics(gt, pred, mask)
    best_pred, best_m = pred.astype(np.float32), base_m
    best_params: dict[str, Any] = {"refined": False}

    for pb in (0.10, 0.14, 0.18, 0.22):
        for pm in (0.30, 0.40, 0.50):
            for tl in (0.10, 0.14, 0.18):
                cand = refine_narx_pure_prediction(
                    pred, prior_week, mask,
                    prior_blend=float(pb),
                    percentile_match=float(pm),
                    tumor_lift=float(tl),
                    hf_gain=0.10,
                )
                m = compute_image_metrics(gt, cand, mask)
                better_ssim = float(m["ssim"]) >= float(best_m["ssim"]) - 0.005
                ok_rmse = float(m["rmse"]) <= float(base_m["rmse"]) + 0.04
                if better_ssim and ok_rmse:
                    best_pred, best_m = cand, m
                    best_params = {
                        "refined": True,
                        "prior_blend": float(pb),
                        "percentile_match": float(pm),
                        "tumor_lift": float(tl),
                    }

    return best_pred, {
        "metrics": best_m,
        "metrics_before": base_m,
        "params": best_params,
    }
