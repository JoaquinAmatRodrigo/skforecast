# Predictions on training data

In the **Skforecast** library, training predictions can be obtained either by using the `backtesting_forecaster()` function or by accessing the `predict()` method of the regressor stored inside the forecaster object."



## Libraries

``` python
# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

## Data

``` python
# Download data
# ==============================================================================
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv')
data_raw = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])

# Data preprocessing
# ==============================================================================
data = data_raw.copy()
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d')
data = data.set_index('datetime')
data = data.asfreq('MS')
data = data['y']
data = data.sort_index()
print(data.head(4))
```

| datetime            |        y |
|:--------------------|---------:|
| 1991-07-01 00:00:00 | 0.429795 |
| 1991-08-01 00:00:00 | 0.400906 |
| 1991-09-01 00:00:00 | 0.432159 |
| 1991-10-01 00:00:00 | 0.492543 |

``` python
# Plot
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
data.plot(ax=ax, label='data')
ax.legend();
```

<img src="../img/data_full_serie.png" style="width: 500px;">


## Training predictions with backtesting_forecaster()

``` python
# Split train-test
# ==============================================================================
n_backtest = 36*3  # Last 9 years are used for backtest
data_train = data[:-n_backtest]
data_test  = data[-n_backtest:]
```

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data_train)
```

**Backtesting_forecaster() parameters:**
+ Forecaster already trained
+ `initial_train_size = None` and `refit = False`
+ If `steps = 1`, all predictions are t(+1). If `steps` > 1 the data will be predicted in folds

``` python
# Backtest train data
# ==============================================================================
metric, predictions_train = backtesting_forecaster(
                                forecaster = forecaster,
                                y          = data_train,
                                initial_train_size = None,
                                steps      = 1,
                                metric     = 'mean_squared_error',
                                refit      = False,
                                verbose    = False
                           )

print(f"Backtest error: {metric}")
```

Backtest error: [0.00045087]

``` python
predictions_train.head(4)
```

|                     |     pred |
|:--------------------|---------:|
| 1992-10-01 00:00:00 | 0.553134 |
| 1992-11-01 00:00:00 | 0.567766 |
| 1992-12-01 00:00:00 | 0.721389 |
| 1993-01-01 00:00:00 | 0.750997 |

``` python
# Plot training predictions
# ==============================================================================
n_lags = max(forecaster.lags)

print(f"The first {n_lags} observations are not predicted because "\
      "it is not possible to create the lags matrix")

fig, ax = plt.subplots(figsize=(9, 4))
data_train.plot(ax=ax, label='train')
predictions_train.plot(ax=ax, label='predictions')
ax.legend();
```

The first 15 observations are not predicted because it is not possible to create the lags matrix

<img src="../img/training_predictions_backtesting_forecaster.png" style="width: 500px;">


## Training predictions through the regressor

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data_train)
```

``` python
# Create lags matrix
# ==============================================================================
X, y = forecaster.create_train_X_y(
            y = data_train, 
            exog = None
       )
```

``` python
# Predict through the regressor
# ==============================================================================
forecaster.regressor.predict(X)[:4]
```

array([0.55313393, 0.56776596, 0.72138941, 0.75099737])