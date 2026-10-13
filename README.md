# Algoritmos de IA — Imagem Médica · Energia · Oclusão · Ultrassom fetal

Repositório curado para disponibilizar ao professor e à disciplina os **algoritmos efetivamente implementados** nos trabalhos de pesquisa/aplicação de:

| Domínio | Projetos de origem |
|---------|-------------------|
| **Imagem médica (MRI / glioma)** | Vescanis / OncoChronos / SegTimeNet |
| **Energia eólica (RN)** | ERA5-RN Wind Power |
| **Oclusão / trajetória em vídeo** | Conceito pedagógico |
| **Ultrassom fetal (HC)** | Experimento didático HC18 (RNN / LSTM / GRU) |

**Autor:** Francisco Fernandes do Vale Segundo  
**Instituição:** UERN  
**Orientação:** Harold Ivan Angulo Bustos  
**Ano:** 2026

---

## Objetivo

Centralizar código de modelos (GNN, CNN, RNN/GRU/LSTM, TCN, baselines clássicos) usados em:

1. previsão longitudinal de FLAIR / modelagem temporal em RM;
2. previsão de geração eólica (ONS) com séries + reanálise ERA5;
3. material conceitual de predição sob oclusão (visão computacional);
4. predição temporal de perímetro cefálico fetal (HC18).

> **Honestidade metodológica:** este repositório **não inventa** algoritmos. O que não tinha implementação executável no workspace aparece marcado como *conceito / paper only / wrapper externo*.

---

## Mapa rápido dos algoritmos

### Imagem médica (`medical_mri/`)

| Família | Algoritmo | Onde está |
|---------|-----------|-----------|
| **CNN 3D** | `ImprovedSpatialTemporalPredictor` (U-Net-like 3D) | `examples/train_patient037_improved_pipeline.py` |
| **CNN 3D** | `WeekVisualEncoder3D` / `VoxelResidualRefiner` | `models/visual_encoder.py`, `models/voxel_refiner.py` |
| **CNN 1D** | `CNNTemporal` | `models/backend_temporal_trainers.py` |
| **RNN / GRU** | `VoxelGRUPredictor`, GRUs estruturais/anatômicos, fusão multimodal | `examples/...`, `models/gru*.py`, `models/fusion.py`, `models/srnn_multimodal.py` |
| **RNN / LSTM** | `VoxelLSTMPredictor` | `examples/train_patient037_improved_pipeline.py` |
| **Transformer** | `VoxelTransformerPredictor`, `TransformerTemporal` | `examples/...`, `models/backend_temporal_trainers.py` |
| **GNN** | `TemporalEntropyGraphPredictor` (grafo temporal de regiões) | `examples/train_patient037_improved_pipeline.py` |
| **NARX** | `VoxelNARXPredictor` e híbridos | `models/narx*.py` |
| **Clássicos** | Random Forest, LinearSVR, stacking Ridge | no pipeline `examples/...` e heads multimodais |

### Energia eólica (`wind_power/`)

| Família | Algoritmo | Onde está |
|---------|-----------|-----------|
| **GRU** | `GRUForecaster` | `models/deep_temporal.py` |
| **LSTM** | `LSTMForecaster` | `models/deep_temporal.py` |
| **TCN** | `TCNForecaster` | `models/deep_temporal.py` |
| **Clássicos** | Persistência, LR, RF, GBM/MLP | `models/baselines.py`, `models/trainers.py` |

### Oclusão (`occlusion/`)

| Status | Conteúdo |
|--------|----------|
| Conceito + stub pedagógico | `concept/` |

### Ultrassom fetal (`fetal_ultrasound/`)

| Família | Algoritmo | Onde está |
|---------|-----------|-----------|
| **RNN** | `TemporalHC(kind="rnn")` | `models/temporal_hc.py` |
| **LSTM** | `TemporalHC(kind="lstm")` | `models/temporal_hc.py` |
| **GRU** | `TemporalHC(kind="gru")` | `models/temporal_hc.py` |

Catálogo completo: [`docs/CATALOGO_ALGORITMOS.md`](docs/CATALOGO_ALGORITMOS.md)

---

## Estrutura

```text
algoritmos-ia-uern/
├── README.md
├── requirements.txt
├── docs/
│   ├── CATALOGO_ALGORITMOS.md
│   └── ORIGEM_E_LICENCA.md
├── medical_mri/
├── wind_power/
├── occlusion/
└── fetal_ultrasound/
    ├── data/hc18_subset/       ← 80 imagens HC18 (CC BY 4.0)
    ├── models/temporal_hc.py   ← RNN / LSTM / GRU
    ├── examples/run_hc18_experiment.py
    └── README.md
```

---

## Como usar

```bash
cd algoritmos-ia-uern
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# treino no subset HC18 já incluído no clone (~1 min na CPU)
python fetal_ultrasound/examples/run_hc18_experiment.py
```

```python
from fetal_ultrasound.models import build_model
model = build_model("lstm", input_dim=34, hidden=64)
```

Imagens: `fetal_ultrasound/data/hc18_subset/` (80 planos HC18, licença CC BY 4.0, citação em `fetal_ultrasound/data/README.md`).

---

## Relação com os trabalhos

| Trabalho | Papel |
|----------|--------|
| **OncoChronos / Iberamia** | Framework longitudinal FLAIR — GRU multimodal + realismo anatômico |
| **ReCiC** | Segmentação mamária (nnU-Net + triagem) — paper |
| **ERA5-RN** | Recurso ERA5-Land ≠ geração ONS; GRU/LSTM/TCN + baselines |
| **Oclusão** | Capítulo conceitual de visão |
| **Fetal / HC18** | Experimento didático de predição temporal (RNN/LSTM/GRU) |

---

## Contato acadêmico

Francisco Fernandes do Vale Segundo — UERN  
Orientação: Harold Ivan Angulo Bustos
