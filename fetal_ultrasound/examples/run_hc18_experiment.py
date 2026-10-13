# -*- coding: utf-8 -*-
"""
Treino didático no subset HC18 já versionado neste repositório.

Lê fetal_ultrasound/data/hc18_subset/ e treina RNN, LSTM e GRU
para predizer o próximo perímetro cefálico (mm).

Uso (na raiz do repositório):
    python fetal_ultrasound/examples/run_hc18_experiment.py
    python fetal_ultrasound/examples/run_hc18_experiment.py --demo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fetal_ultrasound.models import TemporalHC

FETAL = REPO / "fetal_ultrasound"
DATA = FETAL / "data" / "hc18_subset"
RESULTS = FETAL / "results"

DEVICE = torch.device("cpu")
IMG_SIZE = 64
FEAT_DIM = 32
SEQ_LEN = 3
SEED = 42


def load_manifest() -> list[dict]:
    rows = []
    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["subject_id"] = int(r["subject_id"])
            r["hc_mm"] = float(r["hc_mm"])
            r["ga_weeks_proxy"] = float(r["ga_weeks_proxy"])
            rows.append(r)
    return sorted(rows, key=lambda x: x["hc_mm"])


def image_vector(path: Path, proj: np.ndarray) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feat = proj @ arr.reshape(-1)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat.astype(np.float32)


def build_sequences(rows: list[dict], proj: np.ndarray, demo: bool):
    """Janelas deslizantes ordenadas por HC (proxy longitudinal entre pacientes)."""
    feats, hcs = [], []
    print("extraindo features de imagem...")
    for r in rows:
        ip = DATA / "images" / r["filename"]
        if not ip.exists():
            raise FileNotFoundError(ip)
        feats.append(image_vector(ip, proj))
        hcs.append(r["hc_mm"])

    X, y = [], []
    strides = (1,) if demo else (1, 2)
    for step in strides:
        for i in range(0, len(feats) - SEQ_LEN, step):
            hist_rows = []
            for j in range(SEQ_LEN):
                hc_n = hcs[i + j] / 300.0
                dhc = 0.0 if j == 0 else (hcs[i + j] - hcs[i + j - 1]) / 50.0
                hist_rows.append(np.concatenate([feats[i + j], [hc_n, dhc]]))
            X.append(np.stack(hist_rows, axis=0))
            y.append(hcs[i + SEQ_LEN] / 300.0)
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def train_model(kind: str, Xtr, ytr, Xte, yte, epochs: int, verbose: bool = False):
    model = TemporalHC(kind).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    hist = []
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    bs = min(16, len(Xtr))
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        hist.append(ep_loss / max(1, n_batches))
        if verbose and (ep + 1) % max(1, epochs // 6) == 0:
            print(f"    {kind.upper()} epoca {ep + 1}/{epochs}  loss={hist[-1]:.5f}")
    model.eval()
    with torch.no_grad():
        ptr = model(torch.from_numpy(Xtr)).numpy()
        pte = model(torch.from_numpy(Xte)).numpy()
    mae_tr = float(np.mean(np.abs(ptr * 300 - ytr * 300)))
    mae_te = float(np.mean(np.abs(pte * 300 - yte * 300)))
    return model, hist, mae_tr, mae_te, pte


def save_gallery(rows: list[dict], out: Path):
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, 8, dtype=int)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), dpi=120)
    for ax, r in zip(axes.ravel(), picks):
        img = Image.open(DATA / "images" / r["filename"]).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"HC={r['hc_mm']:.0f} mm\n~{r['ga_weeks_proxy']:.1f} sem.", fontsize=10)
        ax.axis("off")
        ann = DATA / "annotations" / r["filename"].replace(".png", "_Annotation.png")
        if ann.exists():
            mask = np.asarray(Image.open(ann).convert("L").resize(img.size))
            ax.contour(mask > 0, levels=[0.5], colors="#E07A3D", linewidths=0.8)
    fig.suptitle("HC18 subset — planos de cabeça fetal ordenados por HC", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results_fig(results: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    for kind, color in [("rnn", "#5C6B7A"), ("lstm", "#0E7C86"), ("gru", "#E07A3D")]:
        axes[0].plot(results[kind]["loss"], label=kind.upper(), color=color, lw=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Smooth L1 (treino)")
    axes[0].set_title("Curvas de treino")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    names = ["RNN", "LSTM", "GRU"]
    maes = [results[k]["mae_test_mm"] for k in ("rnn", "lstm", "gru")]
    colors = ["#5C6B7A", "#0E7C86", "#E07A3D"]
    bars = axes[1].bar(names, maes, color=colors)
    axes[1].set_ylabel("MAE no teste (mm)")
    axes[1].set_title("Erro — predição do próximo HC")
    for b, v in zip(bars, maes):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(maes) * 0.02, f"{v:.1f}", ha="center", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="mais épocas, imprime o loss")
    args = ap.parse_args()

    if not (DATA / "manifest.csv").exists():
        raise SystemExit(f"Subset não encontrado em {DATA}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("Experimento fetal HC18 · RNN / LSTM / GRU")
    print(f"dados: {DATA}")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    proj = rng.normal(0, 1 / np.sqrt(IMG_SIZE * IMG_SIZE), size=(FEAT_DIM, IMG_SIZE * IMG_SIZE)).astype(np.float32)

    rows = load_manifest()
    X, y = build_sequences(rows, proj, demo=args.demo)
    print(f"imagens no subset: {len(rows)}")
    print(f"amostras sequenciais: {len(X)}  shape={tuple(X.shape)}")

    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = max(5, int(0.22 * n))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    epochs = 900 if args.demo else 80
    results, preds = {}, {}
    for kind in ("rnn", "lstm", "gru"):
        print(f"treinando {kind.upper()} ({epochs} epocas)...")
        _model, hist, mae_tr, mae_te, pte = train_model(kind, Xtr, ytr, Xte, yte, epochs=epochs, verbose=args.demo)
        results[kind] = {
            "loss": hist,
            "mae_train_mm": round(mae_tr, 2),
            "mae_test_mm": round(mae_te, 2),
            "epochs": epochs,
        }
        preds[kind] = pte
        print(f"  MAE treino={mae_tr:.2f} mm | teste={mae_te:.2f} mm")

    best = min(results, key=lambda k: results[k]["mae_test_mm"])
    save_gallery(rows, RESULTS / "fetal_gallery.png")
    save_results_fig(results, RESULTS / "fetal_rnn_results.png")

    elapsed = time.perf_counter() - t0
    summary = {
        "dataset": "HC18 subset (Zenodo 10.5281/zenodo.1322001)",
        "license": "CC BY 4.0",
        "subset_images": len(rows),
        "n_sequences": int(n),
        "elapsed_sec": round(elapsed, 1),
        "best_model": best,
        "models": {
            k: {kk: vv for kk, vv in v.items() if kk != "loss"} | {"final_loss": v["loss"][-1]}
            for k, v in results.items()
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"melhor modelo: {best.upper()} (MAE teste={results[best]['mae_test_mm']:.1f} mm)")
    print(f"tempo total: {elapsed:.1f}s")
    print(f"figuras -> {RESULTS}")


if __name__ == "__main__":
    main()
