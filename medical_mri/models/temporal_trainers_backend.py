from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml.temporal.continual import get_continual_state_dir, update_incremental_memory
from ml.temporal.dataset import build_temporal_dataset


def _compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _split_dataset(
    X_seq: np.ndarray,
    X_flat: np.ndarray,
    y: np.ndarray,
    task: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    n_samples = len(y)
    if n_samples < 1:
        raise RuntimeError("Amostras insuficientes para treino. Necessário ao menos 1 registro.")

    if n_samples == 1:
        idx = np.array([0], dtype=np.int64)
        return {
            "X_seq_train": X_seq[idx],
            "X_seq_val": X_seq[idx],
            "X_seq_test": X_seq[:0],
            "X_flat_train": X_flat[idx],
            "X_flat_val": X_flat[idx],
            "X_flat_test": X_flat[:0],
            "y_train": y[idx],
            "y_val": y[idx],
            "y_test": y[:0],
        }

    if n_samples < 6:
        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_test = 1 if n_samples >= 3 else 0
        remaining = n_samples - n_test
        n_val = 1 if remaining >= 2 else 0
        n_train = n_samples - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return {
            "X_seq_train": X_seq[train_idx],
            "X_seq_val": X_seq[val_idx],
            "X_seq_test": X_seq[test_idx],
            "X_flat_train": X_flat[train_idx],
            "X_flat_val": X_flat[val_idx],
            "X_flat_test": X_flat[test_idx],
            "y_train": y[train_idx],
            "y_val": y[val_idx],
            "y_test": y[test_idx],
        }

    stratify = None
    if task == "classification" and len(np.unique(y)) > 1:
        _, counts = np.unique(y, return_counts=True)
        if np.min(counts) >= 2:
            stratify = y

    X_seq_train_val, X_seq_test, X_flat_train_val, X_flat_test, y_train_val, y_test = train_test_split(
        X_seq,
        X_flat,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_ratio_adjusted = val_size / (1.0 - test_size)
    stratify_train = None
    if task == "classification" and len(np.unique(y_train_val)) > 1:
        _, counts = np.unique(y_train_val, return_counts=True)
        if np.min(counts) >= 2:
            stratify_train = y_train_val

    X_seq_train, X_seq_val, X_flat_train, X_flat_val, y_train, y_val = train_test_split(
        X_seq_train_val,
        X_flat_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=stratify_train,
    )

    return {
        "X_seq_train": X_seq_train,
        "X_seq_val": X_seq_val,
        "X_seq_test": X_seq_test,
        "X_flat_train": X_flat_train,
        "X_flat_val": X_flat_val,
        "X_flat_test": X_flat_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _train_classical_models(
    task: str,
    split_data: dict[str, np.ndarray],
    output_dir: Path,
    random_state: int,
    selected_models: set[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    X_train = split_data["X_flat_train"]
    X_val = split_data["X_flat_val"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]

    if "decision_tree" in selected_models and task == "classification":
        tree_model = DecisionTreeClassifier(max_depth=8, random_state=random_state)
        tree_model.fit(X_train, y_train)
        y_pred = tree_model.predict(X_val)
        tree_metrics = _compute_metrics(task, y_val, y_pred)
        tree_path = output_dir / "decision_tree_classifier.joblib"
        joblib.dump(tree_model, tree_path)

        results["decision_tree"] = {
            "status": "trained",
            "artifact": str(tree_path),
            "metrics": tree_metrics,
        }

    if "decision_tree" in selected_models and task != "classification":
        tree_model = DecisionTreeRegressor(max_depth=8, random_state=random_state)
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_val)
        tree_metrics = _compute_metrics(task, y_val, y_pred_tree)
        tree_path = output_dir / "decision_tree_regressor.joblib"
        joblib.dump(tree_model, tree_path)

        results["decision_tree"] = {
            "status": "trained",
            "artifact": str(tree_path),
            "metrics": tree_metrics,
        }

    if "svr" in selected_models and task != "classification":
        svr_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
            ]
        )
        svr_pipeline.fit(X_train, y_train)
        y_pred_svr = svr_pipeline.predict(X_val)
        svr_metrics = _compute_metrics(task, y_val, y_pred_svr)
        svr_path = output_dir / "svr_regressor.joblib"
        joblib.dump(svr_pipeline, svr_path)

        results["svr"] = {
            "status": "trained",
            "artifact": str(svr_path),
            "metrics": svr_metrics,
        }

    if "decision_tree" in selected_models and "decision_tree" not in results:
        results["decision_tree"] = {"status": "skipped", "reason": "Modelo não selecionado para esta task."}

    if "svr" in selected_models and "svr" not in results:
        if task == "classification":
            results["svr"] = {
                "status": "skipped",
                "reason": "SVR é regressivo; use task=regression para treinar SVR.",
            }
        else:
            results["svr"] = {"status": "skipped", "reason": "Modelo não selecionado para esta task."}

    for model_name in ["decision_tree", "svr"]:
        if model_name not in selected_models:
            results[model_name] = {"status": "skipped", "reason": "Não selecionado na requisição."}

    return results


def _train_deep_models(
    task: str,
    split_data: dict[str, np.ndarray],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
    selected_models: set[str],
    continual_state_dir: Path | None = None,
) -> dict[str, Any]:
    deep_model_map = {
        "cnn_1d": "cnn_1d",
        "gru": "gru",
        "transformer": "transformer",
    }

    if not any(name in selected_models for name in deep_model_map):
        return {
            "cnn_1d": {"status": "skipped", "reason": "Não selecionado na requisição."},
            "gru": {"status": "skipped", "reason": "Não selecionado na requisição."},
            "transformer": {"status": "skipped", "reason": "Não selecionado na requisição."},
        }

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        torch_utils_data = importlib.import_module("torch.utils.data")
        DataLoader = getattr(torch_utils_data, "DataLoader")
        TensorDataset = getattr(torch_utils_data, "TensorDataset")
    except Exception as error:
        return {
            "cnn_1d": {"status": "skipped", "reason": f"PyTorch indisponível: {error}"},
            "gru": {"status": "skipped", "reason": f"PyTorch indisponível: {error}"},
            "transformer": {"status": "skipped", "reason": f"PyTorch indisponível: {error}"},
        }

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(split_data["X_seq_train"], dtype=torch.float32)
    X_val = torch.tensor(split_data["X_seq_val"], dtype=torch.float32)

    if task == "classification":
        y_train = torch.tensor(split_data["y_train"], dtype=torch.long)
        y_val = torch.tensor(split_data["y_val"], dtype=torch.long)
        output_dim = int(np.max(split_data["y_train"])) + 1
        criterion = nn.CrossEntropyLoss()
    else:
        y_train = torch.tensor(split_data["y_train"], dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(split_data["y_val"], dtype=torch.float32).unsqueeze(1)
        output_dim = 1
        criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    class CNNTemporal(nn.Module):
        def __init__(self, input_dim: int, output_size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(32, output_size)

        def forward(self, features):
            features = features.transpose(1, 2)
            features = self.net(features).squeeze(-1)
            return self.head(features)

    class GRUTemporal(nn.Module):
        def __init__(self, input_dim: int, output_size: int):
            super().__init__()
            self.rnn = nn.GRU(input_dim, 64, batch_first=True)
            self.head = nn.Linear(64, output_size)

        def forward(self, features):
            _, hidden_state = self.rnn(features)
            return self.head(hidden_state[-1])

    class TransformerTemporal(nn.Module):
        def __init__(self, input_dim: int, output_size: int):
            super().__init__()
            self.proj = nn.Linear(input_dim, 64)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(64, output_size)

        def forward(self, features):
            encoded = self.proj(features)
            encoded = self.encoder(encoded)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)

    def fit_and_eval(
        model_name: str,
        model: Any,
        warm_start_checkpoint: Path | None = None,
        incremental_save_checkpoint: Path | None = None,
    ) -> dict[str, Any]:
        warm_start_status: dict[str, Any] = {"enabled": bool(warm_start_checkpoint), "loaded": False}

        if warm_start_checkpoint and warm_start_checkpoint.exists():
            try:
                state_dict = torch.load(warm_start_checkpoint, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                warm_start_status = {
                    "enabled": True,
                    "loaded": True,
                    "source": str(warm_start_checkpoint),
                }
            except Exception as error:
                warm_start_status = {
                    "enabled": True,
                    "loaded": False,
                    "source": str(warm_start_checkpoint),
                    "reason": str(error),
                }

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                predictions = model(xb)
                loss = criterion(predictions, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            if task == "classification":
                y_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
                y_true = split_data["y_val"]
            else:
                y_pred = val_logits.squeeze(1).cpu().numpy()
                y_true = split_data["y_val"]

        metrics = _compute_metrics(task, y_true, y_pred)
        model_path = output_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        if incremental_save_checkpoint:
            incremental_save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), incremental_save_checkpoint)

        return {
            "status": "trained",
            "artifact": str(model_path),
            "metrics": metrics,
            "device": str(device),
            "warm_start": warm_start_status,
        }

    input_dim = split_data["X_seq_train"].shape[-1]
    results: dict[str, Any] = {}

    if "cnn_1d" in selected_models:
        results["cnn_1d"] = fit_and_eval("cnn_1d_temporal", CNNTemporal(input_dim, output_dim))
    else:
        results["cnn_1d"] = {"status": "skipped", "reason": "Não selecionado na requisição."}

    if "gru" in selected_models:
        gru_checkpoint = continual_state_dir / "gru_incremental.pt" if continual_state_dir else None
        results["gru"] = fit_and_eval(
            "gru_temporal",
            GRUTemporal(input_dim, output_dim),
            warm_start_checkpoint=gru_checkpoint,
            incremental_save_checkpoint=gru_checkpoint,
        )
    else:
        results["gru"] = {"status": "skipped", "reason": "Não selecionado na requisição."}

    if "transformer" in selected_models:
        results["transformer"] = fit_and_eval("transformer_temporal", TransformerTemporal(input_dim, output_dim))
    else:
        results["transformer"] = {"status": "skipped", "reason": "Não selecionado na requisição."}

    return results


def run_temporal_training(
    target_key: str,
    task: str,
    output_base_dir: str,
    models: list[str] | None = None,
    segmentation_sources: list[str] | None = None,
    sequence_length: int = 5,
    min_records_per_patient: int = 2,
    test_size: float = 0.15,
    val_size: float = 0.15,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    enable_continual_learning: bool = True,
    continual_replay_size: int = 2000,
) -> dict[str, Any]:
    if task not in {"classification", "regression"}:
        raise ValueError("task deve ser 'classification' ou 'regression'.")

    if not (0.0 < test_size < 0.5 and 0.0 < val_size < 0.5 and test_size + val_size < 0.8):
        raise ValueError("test_size/val_size inválidos. Use frações pequenas (ex.: 0.15).")

    selected_models = set(models or ["decision_tree", "svr", "cnn_1d", "gru", "transformer"])

    dataset = build_temporal_dataset(
        target_key=target_key,
        task=task,
        sequence_length=sequence_length,
        min_records_per_patient=min_records_per_patient,
        segmentation_sources=segmentation_sources,
    )

    X_seq = dataset["X_seq"]
    X_flat = dataset["X_flat"]
    y = dataset["y"]

    if len(y) < 1:
        raise RuntimeError(
            "Amostras insuficientes para treino. Gere mais segmentações concluídas com target em acquisition_metadata."
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"temporal_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data = _split_dataset(
        X_seq=X_seq,
        X_flat=X_flat,
        y=y,
        task=task,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    classical_results = _train_classical_models(
        task=task,
        split_data=split_data,
        output_dir=output_dir,
        random_state=random_state,
        selected_models=selected_models,
    )

    continual_state_dir = None
    if enable_continual_learning:
        continual_state_dir = get_continual_state_dir(
            base_dir=output_base_dir,
            task=task,
            target_key=target_key,
            segmentation_sources=dataset.get("segmentation_sources", ["automatic", "manual"]),
            sequence_length=sequence_length,
        )

    deep_results = _train_deep_models(
        task=task,
        split_data=split_data,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        selected_models=selected_models,
        continual_state_dir=continual_state_dir,
    )

    continual_result: dict[str, Any]
    if enable_continual_learning:
        try:
            continual_result = update_incremental_memory(
                base_dir=output_base_dir,
                task=task,
                target_key=target_key,
                segmentation_sources=dataset.get("segmentation_sources", ["automatic", "manual"]),
                sequence_length=sequence_length,
                feature_names=dataset["feature_names"],
                X_flat=X_flat,
                y=y,
                random_state=random_state,
                replay_max_size=continual_replay_size,
            )
        except Exception as error:
            continual_result = {
                "status": "failed",
                "reason": f"Falha na atualização incremental: {error}",
            }
    else:
        continual_result = {
            "status": "skipped",
            "reason": "Aprendizado incremental desativado na requisição.",
        }

    manifest = {
        "task": task,
        "target_key": target_key,
        "num_samples": int(len(y)),
        "sequence_length": int(sequence_length),
        "feature_count": int(X_seq.shape[-1]),
        "feature_names": dataset["feature_names"],
        "label_mapping": dataset["label_mapping"],
        "segmentation_sources": dataset.get("segmentation_sources", ["automatic", "manual"]),
        "selected_models": sorted(list(selected_models)),
        "split": {
            "train": int(len(split_data["y_train"])),
            "val": int(len(split_data["y_val"])),
            "test": int(len(split_data["y_test"])),
        },
        "models": {
            "decision_tree": classical_results.get("decision_tree"),
            "svr": classical_results.get("svr"),
            "cnn_1d": deep_results.get("cnn_1d"),
            "gru": deep_results.get("gru"),
            "transformer": deep_results.get("transformer"),
        },
        "continual_learning": continual_result,
    }

    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(output_dir / "X_seq.npy", X_seq)
    np.save(output_dir / "y.npy", y)

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
