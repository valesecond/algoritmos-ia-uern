"""PyTorch temporal models for park-level actual_power (CPU-first).

Not a foundation model. Does not train ConvLSTM or Transformer here;
those decisions are documented in the deep-learning audit.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.paths import ProjectPaths
from .metrics import regression_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GRUForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding:
            y = y[:, :, : -self.padding]
        return y


class TCNForecaster(nn.Module):
    """Small causal TCN; channels-last input (N, T, F) like the RNNs."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
        kernel_size: int = 3,
    ):
        super().__init__()
        dilations = (1, 2, 4)
        layers: List[nn.Module] = []
        in_ch = n_features
        for d in dilations:
            layers.append(CausalConv1d(in_ch, hidden_size, kernel_size, d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_size
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, T, F) -> (N, F, T)
        h = self.net(x.transpose(1, 2))
        return self.head(h[:, :, -1]).squeeze(-1)


def build_model(
    name: str,
    n_features: int,
    hidden_size: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUForecaster(n_features, hidden_size=hidden_size, dropout=dropout)
    if name == "lstm":
        return LSTMForecaster(n_features, hidden_size=hidden_size, dropout=dropout)
    if name == "tcn":
        return TCNForecaster(n_features, hidden_size=hidden_size, dropout=dropout)
    raise ValueError(f"Unknown temporal model: {name}")


@dataclass
class TrainConfig:
    model_name: str = "gru"
    hidden_size: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 10
    seed: int = 42
    sequence_length: int = 24
    horizon: int = 1
    include_power_history: bool = False


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_temporal_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[TrainConfig] = None,
    paths: Optional[ProjectPaths] = None,
    experiment_dir: Optional[Path] = None,
    weights_dir: Optional[Path] = None,
    curve_path: Optional[Path] = None,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train on train split; early stop on validation. Never sees test y."""
    cfg = config or TrainConfig(model_name=model_name)
    cfg.model_name = model_name
    set_seed(cfg.seed)
    device = get_device()
    paths = paths or ProjectPaths.from_root()

    n_features = int(X_train.shape[-1])
    model = build_model(model_name, n_features, cfg.hidden_size, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = _loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(yb)
            n_seen += len(yb)
        train_loss = running / max(n_seen, 1)

        model.eval()
        v_run, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vloss = loss_fn(pred, yb)
                v_run += float(vloss.item()) * len(yb)
                v_n += len(yb)
        val_loss = v_run / max(v_n, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger = logging.getLogger(__name__)
        logger.info(
            "%s epoch %s/%s train_mse=%.4f val_mse=%.4f stale=%s",
            cfg.model_name,
            epoch,
            cfg.max_epochs,
            train_loss,
            val_loss,
            stale,
        )

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    y_val_hat = predict_numpy(model, X_val, device)
    val_metrics = regression_metrics(y_val, y_val_hat, include_mape=True)

    tag = (
        f"{model_name}_L{cfg.sequence_length}_h{cfg.horizon}_"
        f"{'pwr' if cfg.include_power_history else 'era5'}_hid{cfg.hidden_size}"
    )
    exp_dir = experiment_dir or (paths.root / "experiments" / model_name / tag)
    exp_dir.mkdir(parents=True, exist_ok=True)
    wdir = Path(weights_dir) if weights_dir is not None else (paths.models / "deep_learning")
    wdir.mkdir(parents=True, exist_ok=True)
    ckpt = wdir / (checkpoint_name or f"{tag}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "config": asdict(cfg),
            "model_name": model_name,
        },
        ckpt,
    )
    meta = {
        "model_name": model_name,
        "tag": tag,
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "best_val_loss": best_val,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": elapsed,
        "val_metrics": val_metrics,
        "checkpoint": str(ckpt),
        "python": __import__("sys").version,
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savetxt(exp_dir / "train_loss.csv", np.asarray(history["train_loss"]), delimiter=",")
    np.savetxt(exp_dir / "val_loss.csv", np.asarray(history["val_loss"]), delimiter=",")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["train_loss"], label="train MSE")
        ax.plot(history["val_loss"], label="val MSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (actual_power, MWmed²)")
        ax.set_title(f"{model_name} training ({tag})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_dir = paths.figures / "deep_learning"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        default_curve = fig_dir / f"{model_name}_training_curve.png"
        dest_curve = Path(curve_path) if curve_path is not None else default_curve
        dest_curve.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest_curve, dpi=120)
        fig.savefig(exp_dir / "training_curve.png", dpi=120)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass

    return {
        "model": model,
        "meta": meta,
        "history": history,
        "tag": tag,
        "checkpoint": ckpt,
        "device": device,
    }


def predict_numpy(model: nn.Module, X: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    preds = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> nn.Module:
    device = device or get_device()
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = build_model(
        blob["model_name"],
        int(blob["n_features"]),
        hidden_size=int(cfg.get("hidden_size", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model
