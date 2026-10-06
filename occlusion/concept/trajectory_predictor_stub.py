# -*- coding: utf-8 -*-
"""
Stub pedagógico — predição simples de trajetória sob oclusão.

AVISO: código educacional. Não representa um experimento publicado,
não reporta métricas e não deve ser citado como resultado científico.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass
class TrajectoryPredictorStub:
    """Extrapolação linear do deslocamento médio recente (baseline didática)."""

    history: List[Point]

    def predict(self, steps: int = 10) -> List[Point]:
        if len(self.history) < 2:
            raise ValueError("É preciso ao menos 2 pontos observados.")
        arr = np.asarray(self.history, dtype=float)
        delta = arr[-1] - arr[-2]
        # média dos últimos deslocamentos (suavização leve)
        if len(arr) >= 4:
            deltas = np.diff(arr[-4:], axis=0)
            delta = deltas.mean(axis=0)
        preds: List[Point] = []
        cur = arr[-1].copy()
        for _ in range(steps):
            cur = cur + delta
            preds.append((float(cur[0]), float(cur[1])))
        return preds


def demo() -> None:
    observed = [(0.0, 0.0), (1.0, 0.2), (2.0, 0.3), (3.0, 0.25)]
    model = TrajectoryPredictorStub(history=observed)
    hidden = model.predict(steps=5)
    print("Observado:", observed)
    print("Previsto durante oclusão (stub):", hidden)


if __name__ == "__main__":
    demo()
