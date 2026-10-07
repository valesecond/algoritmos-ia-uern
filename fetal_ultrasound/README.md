# Ultrassom fetal — predição temporal de HC (HC18)

## Origem

- **Dataset:** HC18 (Zenodo DOI [10.5281/zenodo.1322001](https://doi.org/10.5281/zenodo.1322001))
- **Projeto:** experimento didático do portfólio UERN (`Vescanis/fetal_us_experiment`)
- **Tarefa:** predizer o próximo perímetro cefálico (mm) a partir de uma sequência curta

## Algoritmos

| ID | Nome | Arquivo |
|----|------|---------|
| F-RNN | Vanilla RNN | `models/temporal_hc.py` (`kind="rnn"`) |
| F-LSTM | LSTM (2 camadas) | `models/temporal_hc.py` (`kind="lstm"`) |
| F-GRU | GRU (2 camadas) | `models/temporal_hc.py` (`kind="gru"`) |

## Honestidade

O release Zenodo do HC18 traz **1 imagem por `subject_id`**.  
As séries longitudinais do experimento são um **proxy**: pacientes distintos ordenados por HC (progresso gestacional aproximado).  
Não é coorte clínica de anomalia fetal nem paper diagnóstico.

## Como treinar (demo ~75 s)

Os scripts de preparação/treino completos ficam em:

`Vescanis/fetal_us_experiment/`

```bash
cd path/to/Vescanis/fetal_us_experiment
python prepare_subset.py
python run_experiment.py --demo
```

Neste repositório, o módulo importável é:

```python
from fetal_ultrasound.models import TemporalHC, build_model

model = build_model("lstm", input_dim=34, hidden=64)
```
