# medical_mri — algoritmos para imagem médica

Modelos usados em previsão longitudinal de FLAIR / fusão multimodal (glioma).

## Principais entradas

- `examples/train_patient037_improved_pipeline.py` — **CNN 3D, GRU, LSTM, Transformer, GNN**, RF, SVR
- `models/srnn_multimodal.py`, `fusion.py`, `visual_encoder.py` — multimodal
- `models/gru_*.py` — variantes estruturais / anatômicas
- `models/narx*.py` — NARX
- `models/backend_temporal_trainers.py` — CNN1D / GRU / Transformer (API)

## Dados

Não incluídos. Requer coortes locais (ex.: LUMIERE / Patient-*).
