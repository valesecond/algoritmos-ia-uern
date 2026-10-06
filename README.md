# Algoritmos de IA — Imagem Médica · Energia Eólica · Oclusão (conceito)

Repositório curado para disponibilizar ao professor e à disciplina os **algoritmos efetivamente implementados** nos trabalhos de pesquisa/aplicação de:

| Domínio | Projetos de origem |
|---------|-------------------|
| **Imagem médica (MRI / glioma)** | Vescanis / OncoChronos / SegTimeNet |
| **Energia eólica (RN)** | ERA5-RN Wind Power |
| **Oclusão / trajetória em vídeo** | Conceito pedagógico (sem experimento publicado neste workspace) |

**Autor:** Francisco Fernandes do Vale Segundo  
**Instituição:** UERN  
**Orientação:** Harold Ivan Angulo Bustos  
**Ano:** 2026

---

## Objetivo

Centralizar código de modelos (GNN, CNN, RNN/GRU/LSTM, TCN, baselines clássicos) usados em:

1. previsão longitudinal de FLAIR / modelagem temporal em RM;
2. previsão de geração eólica (ONS) com séries + reanálise ERA5;
3. material conceitual de predição sob oclusão (visão computacional).

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
| **nnU-Net** | apenas *wrapper* de inferência externa (não treina a arquitetura aqui) | histórico Vescanis / paper ReCiC |

### Energia eólica (`wind_power/`)

| Família | Algoritmo | Onde está |
|---------|-----------|-----------|
| **GRU** | `GRUForecaster` | `models/deep_temporal.py` |
| **LSTM** | `LSTMForecaster` | `models/deep_temporal.py` |
| **TCN** | `TCNForecaster` (convolução causal dilatada) | `models/deep_temporal.py` |
| **Clássicos** | Persistência, média histórica, LR, RF, GBM/MLP | `models/baselines.py`, `models/trainers.py` |
| **Espacial** | Persistência / climatologia de mapas | `models/spatial_forecast.py` |

### Oclusão (`occlusion/`)

| Status | Conteúdo |
|--------|----------|
| Conceito + stub pedagógico | `concept/` + figuras em `assets/` |
| **Sem** métricas/experimentos inventados | Ver `occlusion/README.md` |

---

## Estrutura

```text
algoritmos-ia-uern/
├── README.md                 ← este arquivo
├── requirements.txt
├── .gitignore
├── docs/
│   ├── CATALOGO_ALGORITMOS.md
│   └── ORIGEM_E_LICENCA.md
├── medical_mri/
│   ├── models/               ← módulos PyTorch / sklearn
│   └── examples/             ← pipeline Patient-037 (CNN/GRU/LSTM/GNN/Transformer)
├── wind_power/
│   └── models/               ← GRU/LSTM/TCN + baselines
├── occlusion/
│   ├── README.md
│   └── concept/              ← stub educacional de trajetória
└── assets/                   ← GIFs/figuras de apoio (oclusão)
```

---

## Como usar (visão geral)

```bash
cd algoritmos-ia-uern
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

- **Eólica:** importe `wind_power.models.deep_temporal` (ajuste `PYTHONPATH` ou instale em modo editável conforme sua prática).
- **MRI:** os modelos em `medical_mri/models` são módulos; o pipeline completo está em `medical_mri/examples/train_patient037_improved_pipeline.py` (depende de dados LUMIERE / Patient-* **não incluídos** neste repo por tamanho/privacidade).

---

## O que NÃO está neste repositório

- Volumes NIfTI / datasets clínicos brutos  
- Checkpoints de redes treinadas (pesados)  
- Código de treino completo do nnU-Net (externo)  
- Código experimental do artigo ReCiC (apenas descrito no paper)  
- “IA” do MAMO (é mock de UI, não modelo)

---

## Relação com os trabalhos

| Trabalho | Papel |
|----------|--------|
| **OncoChronos / Iberamia** | Framework longitudinal FLAIR — GRU multimodal + realismo anatômico |
| **ReCiC** | Segmentação mamária (nnU-Net + triagem) — paper; código de modelos não estava no workspace |
| **ERA5-RN** | Recurso ERA5-Land ≠ geração ONS; GRU/LSTM/TCN + baselines |
| **Portfólio / oclusão** | Capítulo conceitual de visão |

---

## Contato acadêmico

Francisco Fernandes do Vale Segundo — UERN  
Orientação: Harold Ivan Angulo Bustos
