# Backtesting SARIMAX and ARIMA models

## Libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.model_selection_statsmodels import backtesting_sarimax
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

# Split data in train snd backtest
# ==============================================================================
n_backtest = 36*3  # Last 9 years are used for backtest
data_train = data[:-n_backtest]
data_backtest = data[-n_backtest:]
```

## Backtest

``` python
metric, predictions_backtest = backtesting_sarimax(
                                    y = data,
                                    order = (12, 1, 1),
                                    seasonal_order = (0, 0, 0, 0),
                                    initial_train_size = len(data_train),
                                    steps = 7,
                                    metric = 'mean_absolute_error',
                                    refit = False,
                                    verbose = True,
                                    fit_kwargs = {'maxiter': 250, 'disp': 0},
                                )
```

<pre>
Number of observations used for training: 96
Number of observations used for backtesting: 108
    Number of folds: 16
    Number of steps per fold: 7
    Last fold only includes 3 observations.
</pre>

``` python
print(f"Error backtest: {metric}")
```

<pre>
Error backtest: [0.05544072]
</pre>

``` python
predictions_backtest.head(4)
```

|                     |   predicted_mean |   lower y |   upper y |
|:--------------------|-----------------:|----------:|----------:|
| 1999-07-01 00:00:00 |         0.734616 |  0.650009 |  0.819222 |
| 1999-08-01 00:00:00 |         0.751851 |  0.660717 |  0.842986 |
| 1999-09-01 00:00:00 |         0.86531  |  0.768134 |  0.962486 |
| 1999-10-01 00:00:00 |         0.832383 |  0.730362 |  0.934404 |