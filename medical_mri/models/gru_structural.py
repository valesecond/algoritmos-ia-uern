"""GRU estrutural: features locais + T1/T2 + losses estruturais no treino."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from patient067.config import Patient067Config
from patient067.tumor_guided.data import TumorGuidedWeekData
from patient067.tumor_guided.structural_loss import (
    StructuralVoxelLoss,
    gradient_loss_3d,
    local_ssim_loss_3d,
    local_texture_loss_3d,
    saturation_penalty,
    tumor_heterogeneity_loss,
)
import torch.nn.functional as F
from patient067.tumor_guided.voxel_features import build_gru_structural_features
from train_patient037_improved_pipeline import (
    _latest_sequence_indices,
    build_temporal_windows,
)


class StructuralGRUPredictor(nn.Module):
    """GRU sobre FLAIR + MLP estatico (local 3x3x3, T1/T2, contexto tumoral)."""

    def __init__(self, seq_len: int, static_dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T), x_static: (B, F)
        gru_out, _ = self.gru(x_seq.unsqueeze(-1))
        h = gru_out[:, -1, :]
        s = self.static_proj(x_static)
        return self.head(torch.cat([h, s], dim=1)).squeeze(-1)


def _sample_patch_indices(
    shape: tuple[int, int, int],
    brain_flat: np.ndarray,
    rng: np.random.Generator,
    *,
    patch: tuple[int, int, int] = (8, 8, 6),
    n_patches: int = 12,
) -> list[np.ndarray]:
    """Indices lineares de patches 3D no cerebro."""
    sx, sy, sz = shape
    px, py, pz = patch
    coords = np.argwhere(brain_flat.reshape(shape))
    if coords.shape[0] == 0:
        return []
    out: list[np.ndarray] = []
    for _ in range(n_patches):
        z, y, x = coords[rng.integers(0, coords.shape[0])]
        z0, y0, x0 = max(0, z - pz // 2), max(0, y - py // 2), max(0, x - px // 2)
        z1, y1, x1 = min(sz, z0 + pz), min(sy, y0 + py), min(sx, x0 + px)
        z0, y0, x0 = z1 - pz, y1 - py, x1 - px
        z0, y0, x0 = max(0, z0), max(0, y0), max(0, x0)
        block = np.zeros(shape, dtype=bool)
        block[z0:z1, y0:y1, x0:x1] = True
        idx = np.where((block & brain_flat.reshape(shape)).ravel())[0]
        if idx.size >= 64:
            out.append(idx)
    return out


def train_tumor_guided_gru_structural(
    weeks: list[TumorGuidedWeekData],
    target_idx: int,
    seq_len: int,
    cfg: Patient067Config,
    device: str = "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """GRU + features locais; treino com gradient/local-SSIM em patches 3D."""
    info: dict[str, Any] = {
        "model_name": "gru_structural_tumor_guided",
        "structural_training": True,
        "tumor_guided": True,
    }
    windows = build_temporal_windows([w.base for w in weeks], seq_len, target_idx)
    if not windows:
        return None, info

    rng = np.random.default_rng(cfg.seed + target_idx + 7001)
    seq_chunks, static_chunks, y_chunks = [], [], []
    per_window = max(2500, cfg.gru_max_samples // max(len(windows), 1))

    for seq_indices, tgt_idx in windows:
        x_seq, x_static, _, fd = build_gru_structural_features(weeks, seq_indices, seq_len, cfg)
        info["feature_dims"] = fd
        y_full = weeks[tgt_idx].flair_resampled.reshape(-1)
        tumor_flat = weeks[tgt_idx].mask_resampled.reshape(-1) > 0.5
        brain_flat = weeks[tgt_idx].brain_mask_resampled.reshape(-1) > 0.5
        n = x_seq.shape[0]
        if per_window < n:
            weights = np.ones(n, dtype=np.float64)
            weights[brain_flat] += 0.6
            if tumor_flat.any():
                weights[tumor_flat] = 4.0  # menos que 10 — evita colapso hiperintenso
            weights /= weights.sum()
            idx = rng.choice(n, size=per_window, replace=False, p=weights)
            x_seq, x_static, y_full = x_seq[idx], x_static[idx], y_full[idx]
        seq_chunks.append(x_seq)
        static_chunks.append(x_static)
        y_chunks.append(y_full)

    x_seq_train = np.concatenate(seq_chunks, axis=0).astype(np.float32)
    x_static_train = np.concatenate(static_chunks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_chunks, axis=0).astype(np.float32)
    info["gru_samples"] = int(x_seq_train.shape[0])

    xs_mu, xs_sd = x_seq_train.mean(), max(x_seq_train.std(), 1e-6)
    xf_mu = x_static_train.mean(axis=0, keepdims=True)
    xf_sd = np.maximum(x_static_train.std(axis=0, keepdims=True), 1e-6)
    y_mu, y_sd = float(y_train.mean()), float(max(y_train.std(), 1e-6))

    x_seq_n = (x_seq_train - xs_mu) / xs_sd
    x_static_n = (x_static_train - xf_mu) / xf_sd
    y_n = (y_train - y_mu) / y_sd

    hidden = int(cfg.tg("gru_hidden_size", 128))
    model = StructuralGRUPredictor(seq_len, x_static_train.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    voxel_criterion = nn.SmoothL1Loss(beta=0.15)
    patch_criterion = StructuralVoxelLoss(
        w_l1=float(cfg.tg("loss_w_l1", 1.0)),
        w_grad=float(cfg.tg("loss_w_grad", 0.35)),
        w_texture=float(cfg.tg("loss_w_texture", 0.25)),
        w_ssim=float(cfg.tg("loss_w_ssim", 0.20)),
        w_tumor_hetero=float(cfg.tg("loss_w_tumor_hetero", 0.30)),
        w_saturation=float(cfg.tg("loss_w_saturation", 0.15)),
    )
    batch_size = 512
    patch_every = int(cfg.tg("structural_patch_every", 4))
    n_patches = int(cfg.tg("structural_n_patches", 10))
    patch_shape = tuple(int(x) for x in cfg.tg("structural_patch_shape", [8, 8, 6]))

    val_n = min(2048, x_seq_n.shape[0] // 5)
    val_idx = rng.choice(x_seq_n.shape[0], size=val_n, replace=False)
    train_mask = np.ones(x_seq_n.shape[0], dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    best_state, best_val, bad = None, float("inf"), 0
    epochs = cfg.gru_epochs

    for epoch in range(epochs):
        perm = rng.permutation(train_idx)
        epoch_losses = []
        for start in range(0, perm.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb_seq = torch.from_numpy(x_seq_n[idx]).to(device)
            xb_static = torch.from_numpy(x_static_n[idx]).to(device)
            yb = torch.from_numpy(y_n[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_seq, xb_static)
            loss = voxel_criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        if (epoch + 1) % patch_every == 0 and windows:
            win_i = int(rng.integers(0, len(windows)))
            seq_indices, tgt_idx = windows[win_i]
            x_seq_f, x_static_f, shape, _ = build_gru_structural_features(
                weeks, seq_indices, seq_len, cfg,
            )
            y_f = weeks[tgt_idx].flair_resampled.reshape(-1)
            x_seq_fn = (x_seq_f - xs_mu) / xs_sd
            x_static_fn = (x_static_f - xf_mu) / xf_sd
            y_fn = (y_f - y_mu) / y_sd
            brain = weeks[tgt_idx].brain_mask_resampled > 0.5
            patch_idx_list = _sample_patch_indices(
                shape, brain, rng, patch=patch_shape, n_patches=n_patches,
            )
            for pidx in patch_idx_list:
                side = int(round(pidx.size ** (1 / 3)))
                side = max(4, min(side, 10))
                n_use = side ** 3
                if pidx.size < n_use:
                    continue
                sl = slice(0, n_use)
                xb_seq = torch.from_numpy(x_seq_fn[pidx[sl]]).to(device)
                xb_static = torch.from_numpy(x_static_fn[pidx[sl]]).to(device)
                yb = torch.from_numpy(y_fn[pidx[sl]]).to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb_seq, xb_static)
                p3 = pred.view(1, 1, side, side, side)
                t3 = yb.view(1, 1, side, side, side)
                tumor_vol = weeks[tgt_idx].mask_resampled.ravel()[pidx[sl]].reshape(1, 1, side, side, side)
                tmask = torch.from_numpy(tumor_vol.astype(np.float32)).to(device)
                win = 3 if side < 8 else 5
                patch_crit = StructuralVoxelLoss(
                    w_l1=patch_criterion.w_l1,
                    w_grad=patch_criterion.w_grad,
                    w_texture=patch_criterion.w_texture,
                    w_ssim=patch_criterion.w_ssim if side >= 7 else 0.0,
                    w_tumor_hetero=patch_criterion.w_tumor_hetero,
                    w_saturation=patch_criterion.w_saturation,
                )
                pl = patch_crit.w_l1 * F.smooth_l1_loss(p3, t3)
                pl = pl + patch_crit.w_grad * gradient_loss_3d(p3, t3)
                pl = pl + patch_crit.w_texture * local_texture_loss_3d(p3, t3, win=win)
                if side >= 7:
                    pl = pl + patch_crit.w_ssim * local_ssim_loss_3d(p3, t3, window=win)
                pl = pl + patch_crit.w_tumor_hetero * tumor_heterogeneity_loss(p3, t3, tmask)
                pl = pl + patch_crit.w_saturation * saturation_penalty(p3, t3, tmask)
                pl.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = float(
                voxel_criterion(
                    model(
                        torch.from_numpy(x_seq_n[val_idx]).to(device),
                        torch.from_numpy(x_static_n[val_idx]).to(device),
                    ),
                    torch.from_numpy(y_n[val_idx]).to(device),
                ).item()
            )
        if vl < best_val:
            best_val, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
        model.train()
        info["gru_epochs"] = epoch + 1
        info["gru_final_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None

    if best_state:
        model.load_state_dict(best_state)

    latest_seq = _latest_sequence_indices(len(weeks), target_idx, seq_len)
    x_seq_t, x_static_t, out_shape, _ = build_gru_structural_features(weeks, latest_seq, seq_len, cfg)
    x_seq_t = (x_seq_t - xs_mu) / xs_sd
    x_static_t = (x_static_t - xf_mu) / xf_sd

    model.eval()
    preds = np.zeros(x_seq_t.shape[0], dtype=np.float32)
    chunk = 120000
    with torch.no_grad():
        for start in range(0, x_seq_t.shape[0], chunk):
            end = min(start + chunk, x_seq_t.shape[0])
            out = model(
                torch.from_numpy(x_seq_t[start:end]).to(device),
                torch.from_numpy(x_static_t[start:end]).to(device),
            )
            preds[start:end] = out.cpu().numpy()
    preds = (preds * y_sd) + y_mu
    return preds.reshape(out_shape).astype(np.float32), info
