"""
NARX (Nonlinear AutoRegressive with eXogenous inputs) voxel-wise temporal predictor.

Integrated alongside legacy GRU/LSTM trainers without modifying the main pipeline file.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Reuse proven training loop from legacy pipeline
from train_patient037_improved_pipeline import (
    WeekData,
    build_temporal_windows,
    _latest_sequence_indices,
)


class VoxelNARXPredictor(nn.Module):
    """
    Predict next FLAIR intensity from:
    - exogenous: seq_len past FLAIR values (same as GRU input)
    - feedback: n_feedback lagged values from the same window (AR part)
    - optional exogenous context: tumor flag + local temporal gradient
    """

    def __init__(
        self,
        seq_len: int = 3,
        n_feedback: int = 2,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_feedback = n_feedback
        # features: seq_len + n_feedback + tumor + delta_last
        in_dim = seq_len + n_feedback + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_narx_features(
    seq_stack: np.ndarray,
    tumor_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    seq_stack: (n_voxels, seq_len) FLAIR history ending at t
    Returns X (n, in_dim), y not included
    """
    seq_len = seq_stack.shape[1]
    n_fb = min(2, seq_len - 1)
    n = seq_stack.shape[0]

    fb_cols = []
    for lag in range(1, n_fb + 1):
        fb_cols.append(seq_stack[:, -(lag + 1)])
    if len(fb_cols) < 2:
        fb_cols.append(seq_stack[:, -1])
        if len(fb_cols) < 2:
            fb_cols.append(seq_stack[:, -1])
    fb = np.stack(fb_cols[:2], axis=1)

    delta = (seq_stack[:, -1] - seq_stack[:, -2]).reshape(-1, 1)
    tumor = tumor_flat.reshape(-1, 1).astype(np.float32)

    X = np.concatenate([seq_stack, fb, tumor, delta], axis=1).astype(np.float32)
    return X, tumor_flat


def train_narx_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    *,
    max_samples: int = 50000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 7e-4,
    hidden_size: int = 128,
    n_feedback: int = 2,
    seed_offset: int = 17,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """Train NARX and predict target week FLAIR volume."""
    info: dict[str, Any] = {
        "narx_samples": 0,
        "narx_epochs": 0,
        "narx_final_loss": None,
        "model_name": "narx",
        "n_feedback": n_feedback,
    }

    windows = build_temporal_windows(weeks_data, seq_len, target_idx)
    if not windows:
        return None, info

    rng = np.random.default_rng(1337 + target_idx + seed_offset)
    X_chunks, y_chunks = [], []
    per_window = max(1200, max_samples // len(windows))

    for seq_indices, tgt_idx in windows:
        seq_stack = np.stack([weeks_data[i].flair_resampled for i in seq_indices], axis=-1)
        X_seq = seq_stack.reshape(-1, seq_len)
        y_full = weeks_data[tgt_idx].flair_resampled.reshape(-1)
        tumor_flat = (weeks_data[tgt_idx].mask_resampled.reshape(-1) > 0.5)

        X_full, _ = _build_narx_features(X_seq, tumor_flat)

        n = X_full.shape[0]
        if per_window < n:
            weights = np.ones(n, dtype=np.float64)
            if tumor_flat.any():
                weights[tumor_flat] = 5.0
            weights /= weights.sum()
            idx = rng.choice(n, size=per_window, replace=False, p=weights)
            X_full = X_full[idx]
            y_full = y_full[idx]

        X_chunks.append(X_full)
        y_chunks.append(y_full)

    X_train = np.concatenate(X_chunks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_chunks, axis=0).astype(np.float32)
    info["narx_samples"] = int(X_train.shape[0])

    x_mean = float(X_train.mean())
    x_std = float(max(X_train.std(), 1e-6))
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))
    X_train = (X_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std

    in_dim = X_train.shape[1]
    model = VoxelNARXPredictor(seq_len=seq_len, n_feedback=n_feedback, hidden_size=hidden_size)
    model.net = nn.Sequential(
        nn.Linear(in_dim, hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, 1),
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss(beta=0.2)

    dataset_size = X_train.shape[0]
    val_size = max(1024, int(0.1 * dataset_size))
    val_idx = rng.choice(dataset_size, size=min(val_size, dataset_size // 4), replace=False)
    train_mask = np.ones(dataset_size, dtype=bool)
    train_mask[val_idx] = False
    train_idx_all = np.where(train_mask)[0]

    X_val, y_val = X_train[val_idx], y_train[val_idx]

    model.train()
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        perm = rng.permutation(train_idx_all)
        epoch_losses = []
        for start in range(0, train_idx_all.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(
                criterion(
                    model(torch.from_numpy(X_val).to(device)),
                    torch.from_numpy(y_val).to(device),
                ).item()
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                break
        model.train()
        info["narx_final_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None
        info["narx_epochs"] = int(epoch + 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    latest_seq = _latest_sequence_indices(len(weeks_data), target_idx, seq_len)
    X_test_seq = (
        np.stack([weeks_data[i].flair_resampled for i in latest_seq], axis=-1)
        .reshape(-1, seq_len)
        .astype(np.float32)
    )
    tumor_test = (weeks_data[target_idx - 1].mask_resampled.reshape(-1) > 0.5)
    X_test, _ = _build_narx_features(X_test_seq, tumor_test)
    X_test = (X_test - x_mean) / x_std
    out_shape = weeks_data[target_idx].flair_resampled.shape

    model.eval()
    preds = np.zeros((X_test.shape[0],), dtype=np.float32)
    chunk = 200000
    with torch.no_grad():
        for start in range(0, X_test.shape[0], chunk):
            xb = torch.from_numpy(X_test[start : start + chunk]).to(device)
            preds[start : start + chunk] = model(xb).cpu().numpy().astype(np.float32)

    preds = (preds * y_std) + y_mean
    return preds.reshape(out_shape).astype(np.float32), info
