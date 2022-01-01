# Predictions on training data

Predictions on training data can be obtained either by using the `backtesting_forecaster()` function or by accessing the `predict()` method of the regressor stored inside the forecaster object.



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
data = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])

# Data preprocessing
# ==============================================================================
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d')
data = data.set_index('datetime')
data = data.asfreq('MS')
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

<br>

## Backtesting forecaster on training data

First, the forecaster is trained.

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data['y'])
```

It is possible to perform backtesting using an already trained forecaster without modifying it if arguments `initial_train_size = None` and `refit = False`. 

``` python
# Backtest train data
# ==============================================================================
metric, predictions_train = backtesting_forecaster(
                                forecaster = forecaster,
                                y          = data['y'],
                                initial_train_size = None,
                                steps      = 1,
                                metric     = 'mean_squared_error',
                                refit      = False,
                                verbose    = False
                           )

print(f"Backtest training error: {metric}")
```

Backtest training error: [0.00045087]

``` python
predictions_train.head(4)
```

|                     |     pred |
|:--------------------|---------:|
| 1992-10-01 00:00:00 | 0.553134 |
| 1992-11-01 00:00:00 | 0.567766 |
| 1992-12-01 00:00:00 | 0.721389 |
| 1993-01-01 00:00:00 | 0.750997 |

<br>

The first 15 observations are not predicted since they are needed to create the lags used as predictors.

``` python
# Plot training predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data.plot(ax=ax)
predictions_train.plot(ax=ax)
ax.legend();
```

<img src="../img/training_predictions_backtesting_forecaster.png" style="width: 500px;">

<br>

## Predict using the internal regressor

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data['y')
```

``` python
# Create training matrix
# ==============================================================================
X, y = forecaster.create_train_X_y(
            y = data['y'], 
            exog = None
       )
```

Using the internal regressor only allows predicting one step.

``` python
# Predict using the internal regressor
# ==============================================================================
forecaster.regressor.predict(X)[:4]
```
array([0.55313393, 0.56776596, 0.72138941, 0.75099737])