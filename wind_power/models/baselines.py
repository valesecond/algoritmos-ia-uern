"""Baseline forecasting models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PersistenceBaseline:
    """
    Persistence: prediction at t equals observed target at t-lag.

    Typically lag=1 for hourly series.
    """

    def __init__(self, lag: int = 1):
        self.lag = lag
        self.name = f"persistence_lag{lag}"

    def predict_from_series(self, y: pd.Series) -> pd.Series:
        return y.shift(self.lag)


class HistoricalMeanBaseline:
    """Predict the training-set mean for all timestamps."""

    def __init__(self):
        self.mean_: Optional[float] = None
        self.name = "historical_mean"

    def fit(self, y_train) -> "HistoricalMeanBaseline":
        self.mean_ = float(np.nanmean(np.asarray(y_train, dtype=float)))
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model not fitted.")
        return np.full(n, self.mean_, dtype=float)


def train_linear_regression(X_train, y_train) -> Pipeline:
    """Linear regression with standardized features."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_baseline(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: Optional[int] = 12,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Random Forest baseline (shallower / fewer trees than full training)."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model
