"""Target-independent spatial-map baselines (OncoChronos-style hold-out)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def _as_2d(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values, dtype=float)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {values.shape}")
    return values


def persistence_map(
    cube: xr.Dataset,
    variable: str,
    target_year: int,
    source_year: Optional[int] = None,
) -> xr.DataArray:
    """Copy the previous annual map (default: target_year - 1)."""
    src = source_year if source_year is not None else target_year - 1
    if src not in np.asarray(cube["year"].values):
        raise KeyError(f"Persistence source year {src} not in cube.")
    out = cube[variable].sel(year=src).drop_vars("year", errors="ignore")
    out.attrs["method"] = "persistence_copy_last_year"
    out.attrs["source_year"] = int(src)
    out.attrs["target_year"] = int(target_year)
    return out


def climatology_map(
    cube: xr.Dataset,
    variable: str,
    train_years: Sequence[int],
    target_year: int,
) -> xr.DataArray:
    """Mean map over training years (historical climatology)."""
    years = [y for y in train_years if y in np.asarray(cube["year"].values)]
    if not years:
        raise ValueError("No training years available in cube.")
    out = cube[variable].sel(year=years).mean(dim="year")
    out.attrs["method"] = "historical_climatology"
    out.attrs["train_years"] = list(years)
    out.attrs["target_year"] = int(target_year)
    return out


def linear_trend_map(
    cube: xr.Dataset,
    variable: str,
    train_years: Sequence[int],
    target_year: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Per-cell linear regression on annual means.

    Returns (predicted map at target_year, slope map per year).
    Cells with fewer than 3 finite years are NaN.
    """
    years = [y for y in train_years if y in np.asarray(cube["year"].values)]
    if len(years) < 3:
        raise ValueError("Linear map forecast needs at least 3 training years.")
    da = cube[variable].sel(year=years)
    y_vals = np.asarray(da["year"].values, dtype=float)
    data = np.asarray(da.values, dtype=float)
    n_year, n_lat, n_lon = data.shape
    pred = np.full((n_lat, n_lon), np.nan)
    slope = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            series = data[:, i, j]
            mask = np.isfinite(series)
            if mask.sum() < 3:
                continue
            sl, intercept, _, _, _ = stats.linregress(y_vals[mask], series[mask])
            slope[i, j] = sl
            pred[i, j] = intercept + sl * float(target_year)

    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    coords = {lat_name: da[lat_name], lon_name: da[lon_name]}
    pred_da = xr.DataArray(pred, coords=coords, dims=(lat_name, lon_name), name=variable)
    slope_da = xr.DataArray(
        slope, coords=coords, dims=(lat_name, lon_name), name=f"{variable}_slope"
    )
    pred_da.attrs["method"] = "per_cell_linear_regression"
    pred_da.attrs["train_years"] = list(years)
    pred_da.attrs["target_year"] = int(target_year)
    slope_da.attrs["units"] = "variable units per year"
    return pred_da, slope_da


def map_error_metrics(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    method: str,
) -> Dict[str, float]:
    """MAE, RMSE, spatial Pearson r, bias on overlapping finite cells."""
    a = _as_2d(y_true)
    b = _as_2d(y_pred)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: true {a.shape} vs pred {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n == 0:
        return {"method": method, "mae": np.nan, "rmse": np.nan, "r": np.nan, "bias": np.nan, "n": 0}
    err = b[mask] - a[mask]
    r = float(np.corrcoef(a[mask], b[mask])[0, 1]) if n > 1 else np.nan
    return {
        "method": method,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r": r,
        "bias": float(np.mean(err)),
        "n": n,
    }


def delta_correlation(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    y_prev: xr.DataArray,
) -> float:
    """
    Correlation of predicted change vs observed change (OncoChronos temporal-delta).

    delta_true = true - previous; delta_pred = pred - previous.
    """
    a = _as_2d(y_true) - _as_2d(y_prev)
    b = _as_2d(y_pred) - _as_2d(y_prev)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def metrics_frame(rows: Iterable[Dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))
