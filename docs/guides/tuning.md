# Tuning forecaster


Skforecast library allows to combine grid search strategy with cross-validation and backtesting in order to identify the combination of lags and hyperparameters that achieve the best prediccion performance.


## Libraries

``` python
# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensembre import Ridge
from sklearn.metrics import mean_squared_error
```
## Data

``` python
# Download data
# ==============================================================================
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv')
data = pd.read_csv(url, sep=',')

# Data preprocessing
# ==============================================================================
data['fecha'] = pd.to_datetime(data['fecha'], format='%Y/%m/%d')
data = data.set_index('fecha')
data = data.rename(columns={'x': 'y'})
data = data.asfreq('MS')
data = data['y']
data = data.sort_index()

# Split train-test
# ==============================================================================
steps = 36
data_train = data[:-steps]
data_test  = data[-steps:]

# Plot
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
data_train.plot(ax=ax, label='train')
data_test.plot(ax=ax, label='test')
ax.legend();
```
<img src="../img/data.png">


## Tuning forecaster


``` python
# Grid search hiperparameters and lags
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # Placeholder, the value will be overwritten
             )

# Regressor hiperparameters
param_grid = {'n_estimators': [50, 100],
              'max_depth': [5, 10]}

# Lags used as predictors
lags_grid = [3, 10, [1,2,3,20]]

results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = data_train,
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = 10,
                        method      = 'cv',
                        metric      = 'mean_squared_error',
                        initial_train_size    = int(len(data_train)*0.5),
                        allow_incomplete_fold = False,
                        return_best = True,
                        verbose     = False
                    )
```

```
2021-08-13 14:28:00,536 root       INFO  Number of models to fit: 12
loop lags_grid:   0%|          | 0/3 [00:00<?, ?it/s]
loop param_grid:   0%|          | 0/4 [00:00<?, ?it/s]
loop param_grid:  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]
loop param_grid:  50%|█████     | 2/4 [00:02<00:02,  1.11s/it]
loop param_grid:  75%|███████▌  | 3/4 [00:02<00:00,  1.08it/s]
loop param_grid: 100%|██████████| 4/4 [00:04<00:00,  1.09s/it]
loop lags_grid:  33%|███▎      | 1/3 [00:04<00:08,  4.16s/it] 
loop param_grid:   0%|          | 0/4 [00:00<?, ?it/s]
loop param_grid:  25%|██▌       | 1/4 [00:00<00:02,  1.22it/s]
loop param_grid:  50%|█████     | 2/4 [00:02<00:02,  1.40s/it]
loop param_grid:  75%|███████▌  | 3/4 [00:03<00:01,  1.32s/it]
loop param_grid: 100%|██████████| 4/4 [00:05<00:00,  1.42s/it]
loop lags_grid:  67%|██████▋   | 2/3 [00:09<00:04,  4.91s/it] 
loop param_grid:   0%|          | 0/4 [00:00<?, ?it/s]
loop param_grid:  25%|██▌       | 1/4 [00:00<00:02,  1.18it/s]
loop param_grid:  50%|█████     | 2/4 [00:02<00:02,  1.23s/it]
loop param_grid:  75%|███████▌  | 3/4 [00:03<00:01,  1.01s/it]
loop param_grid: 100%|██████████| 4/4 [00:04<00:00,  1.27s/it]
loop lags_grid: 100%|██████████| 3/3 [00:14<00:00,  4.78s/it] 
2021-08-13 14:28:14,897 root       INFO  Refitting `forecaster` using the best found parameters and the whole data set: 
lags: [ 1  2  3  4  5  6  7  8  9 10] 
params: {'max_depth': 10, 'n_estimators': 50}
```

``` python
results_grid
```

|    | lags                            | params                                 |    metric |
|---:|:--------------------------------|:---------------------------------------|----------:|
|  6 | [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 10, 'n_estimators': 50}  | 0.0265202 |
|  4 | [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 5, 'n_estimators': 50}   | 0.0269665 |
|  7 | [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 10, 'n_estimators': 100} | 0.0280916 |
|  5 | [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 5, 'n_estimators': 100}  | 0.0286928 |
| 11 | [ 1  2  3 20]                   | {'max_depth': 10, 'n_estimators': 100} | 0.0295003 |
|  0 | [1 2 3]                         | {'max_depth': 5, 'n_estimators': 50}   | 0.0332516 |
|  9 | [ 1  2  3 20]                   | {'max_depth': 5, 'n_estimators': 100}  | 0.0338282 |
|  1 | [1 2 3]                         | {'max_depth': 5, 'n_estimators': 100}  | 0.0341536 |
|  8 | [ 1  2  3 20]                   | {'max_depth': 5, 'n_estimators': 50}   | 0.0365391 |
|  3 | [1 2 3]                         | {'max_depth': 10, 'n_estimators': 100} | 0.036623  |
| 10 | [ 1  2  3 20]                   | {'max_depth': 10, 'n_estimators': 50}  | 0.0393264 |
|  2 | [1 2 3]                         | {'max_depth': 10, 'n_estimators': 50}  | 0.046249  |