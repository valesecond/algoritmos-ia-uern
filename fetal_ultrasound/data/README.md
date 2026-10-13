# Subset HC18 — ultrassom fetal (cabeça)

Recorte didático do **HC18** para a disciplina. Cabe no GitHub (~10 MB). O zip oficial do treino (~127 MB) passa do limite de 100 MB por arquivo e continua no Zenodo.

## O que está aqui

| Pasta / arquivo | Conteúdo |
|-----------------|----------|
| `hc18_subset/images/` | 80 planos de cabeça fetal (PNG) |
| `hc18_subset/annotations/` | máscara da elipse correspondente |
| `hc18_subset/manifest.csv` | índice deste recorte: arquivo, `subject_id`, HC (mm), pixel size, proxy de idade gestacional |
| `hc18_subset/training_set_pixel_size_and_HC.csv` | tabela oficial do treino completo (999 linhas). Só as linhas cujo arquivo está em `images/` têm imagem neste repositório |

O release Zenodo traz **1 imagem por paciente**. As séries do experimento são um proxy: pacientes distintos ordenados por perímetro cefálico (HC).

## Licença e citação

Licença do dataset: **Creative Commons Attribution 4.0** (redistribuição permitida com crédito).

Ao usar estas imagens, cite:

> Thomas L. A. van den Heuvel, Dagmar de Bruijn, Chris L. de Korte and Bram van Ginneken. Automated measurement of fetal head circumference using 2D ultrasound images. *PLOS ONE* 13(8): e0200412, 2018. https://doi.org/10.1371/journal.pone.0200412
>
> Thomas L. A. van den Heuvel, Dagmar de Bruijn, Chris L. de Korte and Bram van Ginneken. Automated measurement of fetal head circumference using 2D ultrasound images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1322001

Página do desafio: https://hc18.grand-challenge.org/

Conjunto completo (999 treino + 335 teste): https://zenodo.org/records/1322001

## Como usar

```bash
python fetal_ultrasound/examples/run_hc18_experiment.py
```

O script lê `manifest.csv` e as imagens desta pasta, treina RNN, LSTM e GRU e grava figuras em `fetal_ultrasound/results/` (essa pasta não entra no git).
