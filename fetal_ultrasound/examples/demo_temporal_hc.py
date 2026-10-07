# -*- coding: utf-8 -*-
"""
Demo mínima dos modelos F-RNN / F-LSTM / F-GRU (sem dataset).

Mostra shapes e um passo de forward — útil para o professor inspecionar a API.
Para o experimento completo com HC18, use Vescanis/fetal_us_experiment/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetal_ultrasound.models import build_model


def main() -> None:
    batch, seq_len, feat = 4, 3, 34
    x = torch.randn(batch, seq_len, feat)
    for kind in ("rnn", "lstm", "gru"):
        model = build_model(kind, input_dim=feat, hidden=64)
        y = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"{kind.upper():4s}  out={tuple(y.shape)}  params={n}")


if __name__ == "__main__":
    main()
