# Tuning forecaster


Skforecast library allows to combine grid search strategy with backtesting in order to identify the combination of lags and hyperparameters that achieve the best prediction performance.


## Libraries

``` python
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensembre import Ridge
```
## Data

``` python
# Download data
# ==============================================================================
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv')
data = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])

# Data preprocessing
# ==============================================================================
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d')
data = data.set_index('datetime')
data = data.asfreq('MS')
data = data['y']
data = data.sort_index()

# Split train-test
# ==============================================================================
steps = 24
data_train = data.loc[: '2001-01-01']
data_val = data.loc['2001-01-01' : '2006-01-01']
data_test  = data.loc['2006-01-01':]
```


## Grid search

``` python
# Grid search hyperparameter and lags
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # Placeholder, the value will be overwritten
             )

# Regressor hyperparameters
param_grid = {'n_estimators': [50, 100],
              'max_depth': [5, 10, 15]}

# Lags used as predictors
lags_grid = [3, 10, [1, 2, 3, 20]]

results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = data.loc[:'2006-01-01'],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = 12,
                        refit       = True,
                        metric      = 'mean_squared_error',
                        initial_train_size = len(data_train),
                        return_best = True,
                        verbose     = False
                    )
```

```
Number of models compared: 18
loop lags_grid:   0%|                                               | 0/3 [00:00<?, ?it/s]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:00<00:02,  2.01it/s]
loop param_grid:  33%|████████████▋                         | 2/6 [00:01<00:03,  1.13it/s]
loop param_grid:  50%|███████████████████                   | 3/6 [00:02<00:02,  1.40it/s]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:03<00:01,  1.20it/s]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:03<00:00,  1.42it/s]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:04<00:00,  1.12it/s]
loop lags_grid:  33%|█████████████                          | 1/3 [00:04<00:09,  4.93s/it]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:00<00:03,  1.43it/s]
loop param_grid:  33%|████████████▋                         | 2/6 [00:01<00:04,  1.02s/it]
loop param_grid:  50%|███████████████████                   | 3/6 [00:02<00:02,  1.24it/s]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:03<00:01,  1.14it/s]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:04<00:00,  1.33it/s]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:05<00:00,  1.20it/s]
loop lags_grid:  67%|██████████████████████████             | 2/3 [00:09<00:04,  4.98s/it]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:00<00:02,  2.08it/s]
loop param_grid:  33%|████████████▋                         | 2/6 [00:01<00:03,  1.33it/s]
loop param_grid:  50%|███████████████████                   | 3/6 [00:02<00:02,  1.25it/s]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:03<00:01,  1.01it/s]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:04<00:00,  1.22it/s]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:04<00:00,  1.17it/s]
loop lags_grid: 100%|███████████████████████████████████████| 3/3 [00:14<00:00,  4.98s/it]
Refitting `forecaster` using the best found lags and parameters and the whole data set: 
  Lags: [ 1  2  3  4  5  6  7  8  9 10] 
  Parameters: {'max_depth': 5, 'n_estimators': 50}
  Backtesting metric: 0.03344857370906804
```

``` python
results_grid
```

| lags                            | params                                 |    metric |   max_depth |   n_estimators |
|---------------------------------|----------------------------------------|-----------|-------------|----------------|
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 5, 'n_estimators': 50}   | 0.0334486 |           5 |             50 |
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 10, 'n_estimators': 50}  | 0.0392212 |          10 |             50 |
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 15, 'n_estimators': 100} | 0.0392658 |          15 |            100 |
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 5, 'n_estimators': 100}  | 0.0395258 |           5 |            100 |
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 10, 'n_estimators': 100} | 0.0402408 |          10 |            100 |
| [ 1  2  3  4  5  6  7  8  9 10] | {'max_depth': 15, 'n_estimators': 50}  | 0.0407645 |          15 |             50 |
| [ 1  2  3 20]                   | {'max_depth': 15, 'n_estimators': 100} | 0.0439092 |          15 |            100 |
| [ 1  2  3 20]                   | {'max_depth': 5, 'n_estimators': 100}  | 0.0449923 |           5 |            100 |
| [ 1  2  3 20]                   | {'max_depth': 5, 'n_estimators': 50}   | 0.0462237 |           5 |             50 |
| [1 2 3]                         | {'max_depth': 5, 'n_estimators': 50}   | 0.0486662 |           5 |             50 |
| [ 1  2  3 20]                   | {'max_depth': 10, 'n_estimators': 100} | 0.0489914 |          10 |            100 |
| [ 1  2  3 20]                   | {'max_depth': 10, 'n_estimators': 50}  | 0.0501932 |          10 |             50 |
| [1 2 3]                         | {'max_depth': 15, 'n_estimators': 100} | 0.0505563 |          15 |            100 |
| [ 1  2  3 20]                   | {'max_depth': 15, 'n_estimators': 50}  | 0.0512172 |          15 |             50 |
| [1 2 3]                         | {'max_depth': 5, 'n_estimators': 100}  | 0.0531229 |           5 |            100 |
| [1 2 3]                         | {'max_depth': 15, 'n_estimators': 50}  | 0.0602604 |          15 |             50 |
| [1 2 3]                         | {'max_depth': 10, 'n_estimators': 50}  | 0.0609513 |          10 |             50 |
| [1 2 3]                         | {'max_depth': 10, 'n_estimators': 100} | 0.0673343 |          10 |            100 |

If argument `return_best = True`, the forecaster is retrained using all data available and the best combination of lags and hyperparameters.

``` python
skforecast
```

```
================= 
ForecasterAutoreg 
================= 
Regressor: RandomForestRegressor(max_depth=5, n_estimators=50, random_state=123) 
Lags: [ 1  2  3  4  5  6  7  8  9 10] 
Window size: 10 
Included exogenous: False 
Type of exogenous variable: None 
Exogenous variables names: None 
Training range: [Timestamp('1991-07-01 00:00:00'), Timestamp('2006-01-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 5, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'random_state': 123, 'verbose': 0, 'warm_start': False} 
Creation date: 2022-01-02 16:40:40 
Last fit date: 2022-01-02 16:40:55 
Skforecast version: 0.4.2
```