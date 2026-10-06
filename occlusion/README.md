# Oclusão e predição de movimento (capítulo conceitual)

## O que é

Em vídeo, um alvo pode ficar **temporariamente oculto**. A ideia é usar o **histórico de trajetória observada** para estimar onde o objeto provavelmente estará durante a oclusão, até reaparecer.

```text
trajetória observada → [OCULTO] → trajetória prevista → reaparecimento
```

## Status neste repositório

| Item | Status |
|------|--------|
| Explicação conceitual | Sim |
| Figuras / GIF de apoio | `../assets/occlusion_*.gif|png` |
| Stub pedagógico PyTorch/NumPy | `concept/trajectory_predictor_stub.py` |
| Dataset, treino, métricas, paper | **Não** — nenhum experimento local foi atribuído |

## Aplicações típicas (contexto didático)

- Continuidade de tracking sob obstáculos  
- Robótica / veículos (antecipação)  
- Análise de movimento em esporte  
- Efeitos / edição de vídeo  

## Como o professor deve ler

Este módulo **não** afirma desempenho numérico.  
Serve para conectar o fio condutor do portfólio: *histórico temporal → predição sob informação incompleta*, análogo conceitual (não arquitetural) ao uso de séries em MRI e eólica.
