# Catálogo de algoritmos (implementações)

## 1. Imagem médica — MRI longitudinal (glioma / FLAIR)

### 1.1 Deep learning

| ID | Nome | Tipo | Arquivo | Tarefa |
|----|------|------|---------|--------|
| M-CNN3D | ImprovedSpatialTemporalPredictor | CNN 3D encoder–decoder | `medical_mri/examples/train_patient037_improved_pipeline.py` | Prever próximo FLAIR (+ máscara aux.) |
| M-GRU | VoxelGRUPredictor | RNN/GRU voxel | mesmo | Séries FLAIR → próximo voxel |
| M-LSTM | VoxelLSTMPredictor | RNN/LSTM voxel | mesmo | idem |
| M-TRF | VoxelTransformerPredictor | Transformer temporal | mesmo | idem |
| M-GNN | TemporalEntropyGraphPredictor | GNN (grafo de regiões) | mesmo | Predição por nós/regiões + entropia |
| M-SRNN | SRNN multimodal | CNN 3D semanal + GRU | `medical_mri/models/srnn_multimodal.py` | Fusão multimodal |
| M-FUS | Multimodal fusion GRU | GRU | `medical_mri/models/fusion.py` | Visual + tabular → FLAIR |
| M-VIS | WeekVisualEncoder3D | CNN 3D | `medical_mri/models/visual_encoder.py` | Encoder por semana |
| M-REF | VoxelResidualRefiner | CNN 3D residual | `medical_mri/models/voxel_refiner.py` | Refino espacial |
| M-SGRU | StructuralGRUPredictor | GRU + contexto | `medical_mri/models/gru_structural.py` | Predição estrutural |
| M-AGRU | GRU anatômico / v2.1 | GRU + perdas anatômicas | `gru_anatomic.py`, `gru_v21*.py` | Realismo anatômico |
| M-NARX | VoxelNARXPredictor | MLP NARX | `medical_mri/models/narx.py` | ARX não linear |
| M-CNN1D | CNNTemporal | CNN 1D | `backend_temporal_trainers.py` | Séries genéricas |
| M-GRU-B | GRUTemporal | GRU | mesmo | API backend |
| M-TRF-B | TransformerTemporal | Transformer | mesmo | API backend |

### 1.2 Clássicos / híbridos

| ID | Nome | Tipo | Onde |
|----|------|------|------|
| M-RF | RandomForestRegressor | Ensemble | pipeline Patient-037 / stacked heads |
| M-SVR | LinearSVR / SVR | Kernel / linear | pipeline + backend |
| M-STACK | Ridge / stacked refiner | Meta-modelo | `multimodal_stacked_head.py` |

### 1.3 Externos / paper only

| Item | Status |
|------|--------|
| nnU-Net (treino da arquitetura) | **Não incluído** — CLI externo / paper ReCiC |
| Regressão logística ReCiC (direção do erro) | **Paper only** no workspace |

---

## 2. Energia eólica — ERA5-RN

| ID | Nome | Tipo | Arquivo | Tarefa |
|----|------|------|---------|--------|
| W-GRU | GRUForecaster | RNN/GRU | `wind_power/models/deep_temporal.py` | Prever `actual_power` (MWmed) por parque |
| W-LSTM | LSTMForecaster | RNN/LSTM | mesmo | idem |
| W-TCN | TCNForecaster | Conv causal dilatada | mesmo | idem |
| W-PERS | Persistência / média | Baseline | `baselines.py` | Referência honesta |
| W-LR | LinearRegression | Linear | `baselines.py` / `trainers.py` | Baseline tabular |
| W-RF | RandomForest | Ensemble | `trainers.py` | Baseline tabular |
| W-GBM | GradientBoosting / XGB | Boosting | `trainers.py` | Baseline tabular |
| W-MLP | MLPRegressor | MLP sklearn | `trainers.py` | Baseline tabular |
| W-MAP | Spatial forecast | Climatologia/persistência | `spatial_forecast.py` | Mapas de recurso |

**Importante:** ERA5-Land = recurso meteo; ONS = geração elétrica. Não misturar unidades.

---

## 3. Oclusão / visão

| ID | Nome | Status |
|----|------|--------|
| O-CONCEPT | Predição de trajetória sob oclusão | Conceito + stub pedagógico em `occlusion/concept/` |
| — | Tracking / SORT / Kalman / Transformer video SOTA | **Não implementados** neste material acadêmico local |

---

## Como o professor pode explorar

1. Abrir `docs/CATALOGO_ALGORITMOS.md` (este arquivo).  
2. Para **GNN / CNN 3D / GRU / LSTM / Transformer** médicos: começar por `medical_mri/examples/train_patient037_improved_pipeline.py` e buscar as classes da tabela.  
3. Para **eólica**: `wind_power/models/deep_temporal.py`.  
4. Para **oclusão**: `occlusion/README.md` (sem métricas inventadas).
