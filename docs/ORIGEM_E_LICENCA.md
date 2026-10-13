# Origem do código e avisos

## Procedência

Os arquivos deste repositório foram **curados/copiados** a partir dos workspaces locais de pesquisa:

- `academic/Vescanis` (OncoChronos / pipelines MRI / multimodal_rnn / Patient-067)
- `ERA5-RN-Wind-Power/project/src/models` (previsão eólica)

## Dados

Datasets clínicos restritos (LUMIERE, INbreast) e séries ONS/ERA5 **não** entram neste repositório.

O único dataset versionado é um **recorte de 80 imagens** do HC18 (ultrassom fetal, plano da cabeça), em `fetal_ultrasound/data/hc18_subset/`.

- Licença: Creative Commons Attribution 4.0 International.
- Fonte: van den Heuvel et al., Zenodo https://doi.org/10.5281/zenodo.1322001
- O zip completo do treino (~127 MB) fica no Zenodo: o GitHub recusa arquivos acima de 100 MB.

## Uso acadêmico

Material destinado a orientação, disciplina e transparência metodológica.  
Cite os trabalhos de origem (Iberamia/OncoChronos, ERA5-RN/SWN, ReCiC) conforme as normas do curso.

## Limitações

- Dependências cruzadas originais podem exigir ajuste de imports/`PYTHONPATH`.
- Alguns arquivos de exemplo são pipelines longos (não “toy demos”).
- Checkpoints e artefatos de treino foram omitidos de propósito.
