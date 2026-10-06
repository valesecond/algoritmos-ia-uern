"""Model training helpers and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train,
    y_train,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
):
    """
    Train Gradient Boosting; try XGBoost if installed, else sklearn GBR.
    """
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        model._era5_backend = "xgboost"
        return model
    except ImportError:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        model._era5_backend = "sklearn_gbr"
        return model


def train_mlp(
    X_train,
    y_train,
    hidden_layer_sizes=(128, 64),
    max_iter: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """MLP with feature scaling to reduce leakage-safe preprocessing issues."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def save_model_artifact(
    model,
    output_dir: Path,
    model_name: str,
    features: List[str],
    params: Dict[str, Any],
    dataset_version: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save model binary + metadata JSON.

    Returns paths of written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.joblib"
    meta_path = output_dir / f"{model_name}_meta.json"

    joblib.dump(model, model_path)
    meta = {
        "model_name": model_name,
        "features": features,
        "params": params,
        "dataset_version": dataset_version,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return {"model": str(model_path), "meta": str(meta_path)}


def load_model_artifact(model_path: Path):
    """Load a joblib model artifact."""
    return joblib.load(Path(model_path))
