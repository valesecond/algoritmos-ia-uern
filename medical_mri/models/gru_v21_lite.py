"""GRU v2.1-LITE — nucleo v1 exato + regularizacao temporal minima."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from patient067.anatomic_realism.voxel_features import build_gru_anatomic_features
from patient067.anatomic_realism_v21_lite.losses_v21_lite import AnatomicVoxelLossV21Lite
from patient067.config import Patient067Config
from patient067.tumor_guided.data import TumorGuidedWeekData
from patient067.tumor_guided.gru_structural import StructuralGRUPredictor, _sample_patch_indices
from train_patient037_improved_pipeline import _latest_sequence_indices, build_temporal_windows


def train_anatomic_gru_v21_lite(
    weeks: list[TumorGuidedWeekData],
    target_idx: int,
    seq_len: int,
    cfg: Patient067Config,
    device: str = "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    info: dict[str, Any] = {
        "model_name": "gru_anatomic_realism_v21_lite",
        "base": "anatomic_realism_v1",
        "residual_reconstruction": False,
        "temporal_regularization": "minimal_auxiliary",
    }
    windows = build_temporal_windows([w.base for w in weeks], seq_len, target_idx)
    if not windows:
        return None, info

    rng = np.random.default_rng(cfg.seed + target_idx + 9211)
    seq_chunks, static_chunks, y_chunks = [], [], []
    per_window = max(3000, cfg.gru_max_samples // max(len(windows), 1))

    for seq_indices, tgt_idx in windows:
        x_seq, x_static, _, fd = build_gru_anatomic_features(weeks, seq_indices, seq_len, cfg)
        info["feature_dims"] = fd
        y_full = weeks[tgt_idx].flair_resampled.reshape(-1)
        tumor_flat = weeks[tgt_idx].mask_resampled.reshape(-1) > 0.5
        brain_flat = weeks[tgt_idx].brain_mask_resampled.reshape(-1) > 0.5
        n = x_seq.shape[0]
        if per_window < n:
            weights = np.ones(n, dtype=np.float64)
            weights[brain_flat] += 0.8
            if tumor_flat.any():
                weights[tumor_flat] = 3.5
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

    hidden = int(cfg.tg("gru_hidden_size", 160))
    model = StructuralGRUPredictor(seq_len, x_static_train.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-5)
    voxel_criterion = nn.SmoothL1Loss(beta=0.12)
    patch_criterion = AnatomicVoxelLossV21Lite(
        w_l1=float(cfg.tg("loss_w_l1", 1.0)),
        w_grad=float(cfg.tg("loss_w_grad", 0.42)),
        w_texture=float(cfg.tg("loss_w_texture", 0.32)),
        w_ssim=float(cfg.tg("loss_w_ssim", 0.26)),
        w_hf=float(cfg.tg("loss_w_hf", 0.28)),
        w_edge=float(cfg.tg("loss_w_edge", 0.18)),
        w_ms_texture=float(cfg.tg("loss_w_ms_texture", 0.22)),
        w_tumor_hetero=float(cfg.tg("loss_w_tumor_hetero", 0.45)),
        w_saturation=float(cfg.tg("loss_w_saturation", 0.24)),
        w_tumor_dynamic=float(cfg.tg("loss_w_tumor_dynamic", 0.10)),
        w_entropy=float(cfg.tg("loss_w_entropy", 0.05)),
        w_microcontrast=float(cfg.tg("loss_w_microcontrast", 0.05)),
        w_vol_continuity=float(cfg.tg("loss_w_vol_continuity", 0.04)),
        w_temporal_delta=float(cfg.tg("loss_w_temporal_delta", 0.04)),
        w_temporal_stability=float(cfg.tg("loss_w_temporal_stability", 0.03)),
    )
    batch_size = 512
    patch_every = int(cfg.tg("structural_patch_every", 3))
    n_patches = int(cfg.tg("structural_n_patches", 12))
    patch_shape = tuple(int(x) for x in cfg.tg("structural_patch_shape", [8, 8, 6]))

    val_n = min(2048, x_seq_n.shape[0] // 5)
    val_idx = rng.choice(x_seq_n.shape[0], size=val_n, replace=False)
    train_mask = np.ones(x_seq_n.shape[0], dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    best_state, best_val, bad = None, float("inf"), 0
    for epoch in range(cfg.gru_epochs):
        perm = rng.permutation(train_idx)
        epoch_losses = []
        for start in range(0, perm.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                torch.from_numpy(x_seq_n[idx]).to(device),
                torch.from_numpy(x_static_n[idx]).to(device),
            )
            loss = voxel_criterion(pred, torch.from_numpy(y_n[idx]).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        if (epoch + 1) % patch_every == 0 and windows:
            win_i = int(rng.integers(0, len(windows)))
            seq_indices, tgt_idx = windows[win_i]
            if tgt_idx < 1:
                continue
            x_seq_f, x_static_f, shape, _ = build_gru_anatomic_features(
                weeks, seq_indices, seq_len, cfg,
            )
            y_f = weeks[tgt_idx].flair_resampled.reshape(-1)
            prior_f = weeks[tgt_idx - 1].flair_resampled.reshape(-1).astype(np.float32)
            brain_f = weeks[tgt_idx].brain_mask_resampled.reshape(-1).astype(np.float32)
            x_seq_fn = (x_seq_f - xs_mu) / xs_sd
            x_static_fn = (x_static_f - xf_mu) / xf_sd
            y_fn = (y_f - y_mu) / y_sd
            brain = weeks[tgt_idx].brain_mask_resampled > 0.5
            for pidx in _sample_patch_indices(shape, brain, rng, patch=patch_shape, n_patches=n_patches):
                side = max(4, min(int(round(pidx.size ** (1 / 3))), 10))
                n_use = side ** 3
                if pidx.size < n_use:
                    continue
                sl = slice(0, n_use)
                optimizer.zero_grad(set_to_none=True)
                pred = model(
                    torch.from_numpy(x_seq_fn[pidx[sl]]).to(device),
                    torch.from_numpy(x_static_fn[pidx[sl]]).to(device),
                )
                p3 = pred.view(1, 1, side, side, side)
                t3 = torch.from_numpy(y_fn[pidx[sl]]).to(device).view(1, 1, side, side, side)
                prior_b = torch.from_numpy(prior_f[pidx[sl]]).to(device).view(1, 1, side, side, side)
                brain_b = torch.from_numpy(brain_f[pidx[sl]]).to(device).view(1, 1, side, side, side)
                tmask = torch.from_numpy(
                    weeks[tgt_idx].mask_resampled.ravel()[pidx[sl]].astype(np.float32),
                ).to(device).view(1, 1, side, side, side)
                pl = patch_criterion(p3, t3, tmask, prior=prior_b, brain_mask=brain_b)
                pl.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = float(voxel_criterion(
                model(
                    torch.from_numpy(x_seq_n[val_idx]).to(device),
                    torch.from_numpy(x_static_n[val_idx]).to(device),
                ),
                torch.from_numpy(y_n[val_idx]).to(device),
            ).item())
        if vl < best_val:
            best_val, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
        model.train()
        info["gru_epochs"] = epoch + 1
        info["gru_final_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None

    if best_state:
        model.load_state_dict(best_state)

    latest_seq = _latest_sequence_indices(len(weeks), target_idx, seq_len)
    x_seq_t, x_static_t, out_shape, _ = build_gru_anatomic_features(weeks, latest_seq, seq_len, cfg)
    x_seq_t = (x_seq_t - xs_mu) / xs_sd
    x_static_t = (x_static_t - xf_mu) / xf_sd
    model.eval()
    preds = np.zeros(x_seq_t.shape[0], dtype=np.float32)
    chunk = 100000
    with torch.no_grad():
        for start in range(0, x_seq_t.shape[0], chunk):
            end = min(start + chunk, x_seq_t.shape[0])
            preds[start:end] = model(
                torch.from_numpy(x_seq_t[start:end]).to(device),
                torch.from_numpy(x_static_t[start:end]).to(device),
            ).cpu().numpy()
    preds = (preds * y_sd) + y_mu
    return preds.reshape(out_shape).astype(np.float32), info
