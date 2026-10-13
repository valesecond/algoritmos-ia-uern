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

## Dataset neste repositório

O subset já está em `data/hc18_subset/` (80 imagens + anotações + `manifest.csv`, ~10 MB).

Licença **CC BY 4.0**. Citação e link do conjunto completo: [`data/README.md`](data/README.md).

```bash
# na raiz do repositório
python fetal_ultrasound/examples/run_hc18_experiment.py
```

Figuras e `metrics.json` saem em `fetal_ultrasound/results/` (não versionado).

O zip oficial do treino (~127 MB) não entra aqui: o GitHub recusa arquivo acima de 100 MB. O release completo continua em https://doi.org/10.5281/zenodo.1322001

Neste repositório, o módulo importável é:

```python
from fetal_ultrasound.models import TemporalHC, build_model

model = build_model("lstm", input_dim=34, hidden=64)
```
