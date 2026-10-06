# wind_power — algoritmos para energia eólica

## Deep temporal

Arquivo central: `models/deep_temporal.py`

- `GRUForecaster`
- `LSTMForecaster`
- `TCNForecaster`

## Baselines

- `models/baselines.py` — persistência, média, regressão linear
- `models/trainers.py` — RF, GBM/MLP
- `models/spatial_forecast.py` — mapas (climatologia / persistência)

## Princípio

**ERA5-Land ≠ ONS (MWmed).**  
ERA5 = recurso meteo; ONS = geração elétrica observada.
