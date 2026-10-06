"""Baseline and Machine Learning model utilities."""

from .metrics import regression_metrics, metrics_to_frame
from .baselines import (
    PersistenceBaseline,
    HistoricalMeanBaseline,
    train_linear_regression,
    train_random_forest_baseline,
)
from .spatial_forecast import (
    persistence_map,
    climatology_map,
    linear_trend_map,
    map_error_metrics,
    delta_correlation,
)
from .trainers import (
    train_random_forest,
    train_gradient_boosting,
    train_mlp,
    save_model_artifact,
    load_model_artifact,
)
from .deep_temporal import (
    GRUForecaster,
    LSTMForecaster,
    TCNForecaster,
    train_temporal_model,
    build_model,
)

__all__ = [
    "regression_metrics",
    "metrics_to_frame",
    "PersistenceBaseline",
    "HistoricalMeanBaseline",
    "train_linear_regression",
    "train_random_forest_baseline",
    "train_random_forest",
    "train_gradient_boosting",
    "train_mlp",
    "save_model_artifact",
    "load_model_artifact",
    "GRUForecaster",
    "LSTMForecaster",
    "TCNForecaster",
    "train_temporal_model",
    "build_model",
    "persistence_map",
    "climatology_map",
    "linear_trend_map",
    "map_error_metrics",
    "delta_correlation",
]
