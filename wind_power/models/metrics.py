"""Extended evaluation metrics for wind-power forecasting (publication set).

Ten core metrics for scientific comparison (plus helpers).
MAPE is optional and discouraged near zero generation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Canonical ordered list for wind-power forecasting papers (industry + research).
# MAPE is intentionally excluded (unstable near zero generation).
PUBLICATION_METRIC_KEYS = [
    "mae",  # 1 absolute scale
    "rmse",  # 2 large-error sensitive
    "nmae_capacity",  # 3 MAE / installed capacity (industry standard)
    "nrmse",  # 4 RMSE / std(obs)
    "skill_rmse_vs_persistence",  # 5 key utility metric
    "r2",  # 6 variance explained
    "bias",  # 7 mean bias (MBE)
    "corr",  # 8 Pearson correlation
    "medae",  # 9 robust absolute error
    "skill_mae_vs_persistence",  # 10 MAE skill vs persistence
]


def regression_metrics(
    y_true,
    y_pred,
    *,
    include_mape: bool = False,
    mape_epsilon: float = 1e-6,
    capacity: Optional[np.ndarray] = None,
    y_persistence: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Core regression metrics.

    Always returns MAE, RMSE, R², n, smape.
    Prefer :func:`publication_metrics` for the full 10-metric panel.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
        "rmse": (
            float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan
        ),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n": int(len(y_true)),
    }

    if include_mape:
        denom = np.maximum(np.abs(y_true), mape_epsilon)
        metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
        metrics["mape_note"] = (
            "MAPE uses epsilon flooring; interpret cautiously near zero."
        )
    denom_s = np.abs(y_true) + np.abs(y_pred) + mape_epsilon
    metrics["smape"] = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom_s) * 100.0)
    return metrics


def publication_metrics(
    y_true,
    y_pred,
    *,
    y_persistence: Optional[np.ndarray] = None,
    capacity_mw: Optional[np.ndarray] = None,
    mape_epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Ten publication-oriented metrics for wind-power forecasting (MWmed).

    1. mae — Mean Absolute Error (MWmed)
    2. rmse — Root Mean Squared Error (MWmed)
    3. nmae_capacity — mean(|err|/P_installed) when capacity provided; else NaN
    4. nrmse — RMSE / std(y)
    5. skill_rmse_vs_persistence — 1 − RMSE/RMSE_pers
    6. r2 — Coefficient of determination
    7. bias — Mean error (pred − true); positive = overforecast
    8. corr — Pearson correlation
    9. medae — Median Absolute Error
    10. skill_mae_vs_persistence — 1 − MAE/MAE_pers

    Also returns smape and nmae (= MAE/mean|y|) as extras when useful.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if y_persistence is not None:
        yp = np.asarray(y_persistence, dtype=float).ravel()
        mask = mask & np.isfinite(yp)
        yp = yp[mask]
    else:
        yp = None
    if capacity_mw is not None:
        cap = np.asarray(capacity_mw, dtype=float).ravel()
        mask = mask & np.isfinite(cap) & (cap > 0)
        cap = cap[mask]
    else:
        cap = None

    yt = y_true[mask]
    yh = y_pred[mask]
    n = int(len(yt))
    out: Dict[str, float] = {"n": float(n)}
    if n == 0:
        for k in PUBLICATION_METRIC_KEYS:
            out[k] = float("nan")
        return out

    err = yh - yt
    abs_err = np.abs(err)
    out["mae"] = float(np.mean(abs_err))
    out["rmse"] = float(np.sqrt(np.mean(err**2)))
    out["r2"] = float(r2_score(yt, yh)) if n > 1 else float("nan")
    out["bias"] = float(np.mean(err))
    out["medae"] = float(np.median(abs_err))
    if n > 2 and float(np.std(yt)) > 1e-12 and float(np.std(yh)) > 1e-12:
        out["corr"] = float(np.corrcoef(yt, yh)[0, 1])
    else:
        out["corr"] = float("nan")

    denom_s = np.abs(yt) + np.abs(yh) + mape_epsilon
    out["smape"] = float(np.mean(2.0 * abs_err / denom_s) * 100.0)

    mean_abs = float(np.mean(np.abs(yt)))
    out["nmae"] = float(out["mae"] / mean_abs) if mean_abs > 1e-12 else float("nan")
    std_y = float(np.std(yt))
    out["nrmse"] = float(out["rmse"] / std_y) if std_y > 1e-12 else float("nan")

    if yp is not None and len(yp) == n:
        rmse_p = float(np.sqrt(np.mean((yp - yt) ** 2)))
        out["rmse_persistence"] = rmse_p
        out["skill_rmse_vs_persistence"] = (
            float(1.0 - out["rmse"] / rmse_p) if rmse_p > 1e-12 else float("nan")
        )
        mae_p = float(np.mean(np.abs(yp - yt)))
        out["mae_persistence"] = mae_p
        out["skill_mae_vs_persistence"] = (
            float(1.0 - out["mae"] / mae_p) if mae_p > 1e-12 else float("nan")
        )
    else:
        out["skill_rmse_vs_persistence"] = float("nan")
        out["skill_mae_vs_persistence"] = float("nan")

    if cap is not None and len(cap) == n:
        out["nmae_capacity"] = float(np.mean(abs_err / cap))
    else:
        out["nmae_capacity"] = float("nan")
    return out


def metrics_to_frame(
    metrics_by_model: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Convert {model: metrics_dict} to a tidy DataFrame."""
    rows = []
    for name, metrics in metrics_by_model.items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if not isinstance(v, str)})
        rows.append(row)
    return pd.DataFrame(rows)


def publication_metrics_legend() -> str:
    return "\n".join(
        [
            "1. MAE (MWmed): erro absoluto médio",
            "2. RMSE (MWmed): erro quadrático médio (penaliza outliers/rampas)",
            "3. nMAE_cap: MAE / capacidade instalada (padrão indústria eólica)",
            "4. nRMSE: RMSE / desvio-padrão(obs)",
            "5. Skill RMSE vs persistência: 1 − RMSE/RMSE_pers (>0 = melhor)",
            "6. R²: fração da variância explicada",
            "7. Bias/MBE (MWmed): média(pred−obs); >0 = superestimativa",
            "8. Corr: correlação de Pearson",
            "9. MedAE (MWmed): mediana dos |erros| (robusto)",
            "10. Skill MAE vs persistência: 1 − MAE/MAE_pers (>0 = melhor)",
            "Evitar MAPE clássico (instável com geração ≈ 0).",
        ]
    )
