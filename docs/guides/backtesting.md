# Backtesting

**Backtesting with refit**

The model is trained each time before making the predictions, in this way, the model use all the information available so far. It is a variation of the standard cross-validation but, instead of making a random distribution of the observations, the training set is increased sequentially, maintaining the temporal order of the data.

<p align="center"><img src="../img/diagram-backtesting-refit.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_refit.gif" style="width: 600px;"></p>

**Backtesting with refit and fixed train size**

A technique similar to Backtesting with refit but, in this case, the training set doesn't increase sequentially.

<p align="center"><img src="../img/diagram-backtesting-refit-fixed-train-size.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_refit_fixed_train_size.gif" style="width: 600px;"></p>

**Backtesting without refit**

After an initial train, the model is used sequentially without updating it and following the temporal order of the data. This strategy has the advantage of being much faster since the model is only trained once. However, the model does not incorporate the latest information available so it may lose predictive capacity over time.

<p align="center"><img src="../img/diagram-backtesting-no-refit.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_no_refit.gif" style="width: 600px;"></p>

## Libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from sklearn.linear_model import Ridge
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
data = data[['y']]
data = data.sort_index()
display(data.head(4))

# Split data in train and backtest
# ==============================================================================
n_backtest = 36*3  # Last 9 years are used for backtest
data_train = data[:-n_backtest]
data_backtest = data[-n_backtest:]
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
data.plot(ax=ax)
ax.legend();
```

<img src="../img/data_full_serie.png" style="width: 500px;">

## Backtest

``` python
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

metric, predictions_backtest = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['y'],
                                    initial_train_size = len(data_train),
                                    fixed_train_size   = False,
                                    steps      = 10,
                                    metric     = 'mean_squared_error',
                                    refit      = True,
                                    verbose    = True
                               )
```

<pre>
Information of backtesting process
----------------------------------
Number of observations used for initial training: 96
Number of observations used for backtesting: 108
    Number of folds: 11
    Number of steps per fold: 10
    Last fold only includes 8 observations.

Data partition in fold: 0
    Training:   1991-07-01 00:00:00 -- 1999-06-01 00:00:00
    Validation: 1999-07-01 00:00:00 -- 2000-04-01 00:00:00
Data partition in fold: 1
    Training:   1991-07-01 00:00:00 -- 2000-04-01 00:00:00
    Validation: 2000-05-01 00:00:00 -- 2001-02-01 00:00:00
Data partition in fold: 2
    Training:   1991-07-01 00:00:00 -- 2001-02-01 00:00:00
    Validation: 2001-03-01 00:00:00 -- 2001-12-01 00:00:00
Data partition in fold: 3
    Training:   1991-07-01 00:00:00 -- 2001-12-01 00:00:00
    Validation: 2002-01-01 00:00:00 -- 2002-10-01 00:00:00
Data partition in fold: 4
    Training:   1991-07-01 00:00:00 -- 2002-10-01 00:00:00
    Validation: 2002-11-01 00:00:00 -- 2003-08-01 00:00:00
Data partition in fold: 5
    Training:   1991-07-01 00:00:00 -- 2003-08-01 00:00:00
    Validation: 2003-09-01 00:00:00 -- 2004-06-01 00:00:00
Data partition in fold: 6
    Training:   1991-07-01 00:00:00 -- 2004-06-01 00:00:00
    Validation: 2004-07-01 00:00:00 -- 2005-04-01 00:00:00
Data partition in fold: 7
    Training:   1991-07-01 00:00:00 -- 2005-04-01 00:00:00
    Validation: 2005-05-01 00:00:00 -- 2006-02-01 00:00:00
Data partition in fold: 8
    Training:   1991-07-01 00:00:00 -- 2006-02-01 00:00:00
    Validation: 2006-03-01 00:00:00 -- 2006-12-01 00:00:00
Data partition in fold: 9
    Training:   1991-07-01 00:00:00 -- 2006-12-01 00:00:00
    Validation: 2007-01-01 00:00:00 -- 2007-10-01 00:00:00
Data partition in fold: 10
    Training:   1991-07-01 00:00:00 -- 2007-10-01 00:00:00
    Validation: 2007-11-01 00:00:00 -- 2008-06-01 00:00:00
</pre>

``` python
print(f"Backtest error: {metric}")
```

<pre>
Backtest error: 0.007266210302212634
</pre>

``` python
predictions_backtest.head(4)
```

|                     |     pred |
|:--------------------|---------:|
| 1999-07-01 00:00:00 | 0.712336 |
| 1999-08-01 00:00:00 | 0.750542 |
| 1999-09-01 00:00:00 | 0.802371 |
| 1999-10-01 00:00:00 | 0.806941 | 

``` python
# Plot backtest predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_backtest.plot(ax=ax)
predictions_backtest.plot(ax=ax)
ax.legend();
```

<img src="../img/predictions_backtesting_forecaster.png" style="width: 500px;">



## Backtest with prediction intervals

``` python
forecaster = ForecasterAutoreg(
                regressor = Ridge(),
                lags      = 15 
             )

metric, predictions_backtest = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['y'],
                                    initial_train_size = len(data_train),
                                    fixed_train_size   = False,
                                    steps      = 10,
                                    metric     = 'mean_squared_error',
                                    refit      = True,
                                    interval   = [5, 95],
                                    n_boot     = 500,
                                    verbose    = True
                               )

```

<pre>
Information of backtesting process
----------------------------------
Number of observations used for initial training: 96
Number of observations used for backtesting: 108
    Number of folds: 11
    Number of steps per fold: 10
    Last fold only includes 8 observations.

Data partition in fold: 0
    Training:   1991-07-01 00:00:00 -- 1999-06-01 00:00:00
    Validation: 1999-07-01 00:00:00 -- 2000-04-01 00:00:00
Data partition in fold: 1
    Training:   1991-07-01 00:00:00 -- 2000-04-01 00:00:00
    Validation: 2000-05-01 00:00:00 -- 2001-02-01 00:00:00
Data partition in fold: 2
    Training:   1991-07-01 00:00:00 -- 2001-02-01 00:00:00
    Validation: 2001-03-01 00:00:00 -- 2001-12-01 00:00:00
Data partition in fold: 3
    Training:   1991-07-01 00:00:00 -- 2001-12-01 00:00:00
    Validation: 2002-01-01 00:00:00 -- 2002-10-01 00:00:00
Data partition in fold: 4
    Training:   1991-07-01 00:00:00 -- 2002-10-01 00:00:00
    Validation: 2002-11-01 00:00:00 -- 2003-08-01 00:00:00
Data partition in fold: 5
    Training:   1991-07-01 00:00:00 -- 2003-08-01 00:00:00
    Validation: 2003-09-01 00:00:00 -- 2004-06-01 00:00:00
Data partition in fold: 6
    Training:   1991-07-01 00:00:00 -- 2004-06-01 00:00:00
    Validation: 2004-07-01 00:00:00 -- 2005-04-01 00:00:00
Data partition in fold: 7
    Training:   1991-07-01 00:00:00 -- 2005-04-01 00:00:00
    Validation: 2005-05-01 00:00:00 -- 2006-02-01 00:00:00
Data partition in fold: 8
    Training:   1991-07-01 00:00:00 -- 2006-02-01 00:00:00
    Validation: 2006-03-01 00:00:00 -- 2006-12-01 00:00:00
Data partition in fold: 9
    Training:   1991-07-01 00:00:00 -- 2006-12-01 00:00:00
    Validation: 2007-01-01 00:00:00 -- 2007-10-01 00:00:00
Data partition in fold: 10
    Training:   1991-07-01 00:00:00 -- 2007-10-01 00:00:00
    Validation: 2007-11-01 00:00:00 -- 2008-06-01 00:00:00
</pre>

``` python
# Plot backtest predictions
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
data_backtest.plot(ax=ax, label='test')
predictions_backtest.iloc[:, 0].plot(ax=ax, label='predictions')
ax.fill_between(
    predictions_backtest.index,
    predictions_backtest.iloc[:, 1],
    predictions_backtest.iloc[:, 2],
    color = 'red',
    alpha = 0.2,
    label = 'prediction interval'
)
ax.legend();
```

<img src="../img/prediction_interval_backtesting_forecaster.png" style="width: 500px;">



## Predictions on training data

Predictions on training data can be obtained either by using the `backtesting_forecaster()` function or by accessing the `predict()` method of the regressor stored inside the forecaster object.

### Predict using backtesting_forecaster()

A trained forecaster is needed.

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data['y'])
```

Set arguments `initial_train_size = None` and `refit = False` to perform backtesting using the already trained forecaster. 

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

Backtest training error: 0.0005392479040738611

``` python
predictions_train.head(4)
```

|                     |     pred |
|:--------------------|---------:|
| 1992-10-01 00:00:00 | 0.553611 |
| 1992-11-01 00:00:00 | 0.568324 |
| 1992-12-01 00:00:00 | 0.735167 |
| 1993-01-01 00:00:00 | 0.723217 |

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

### Predict using the internal regressor

``` python
# Fit forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

forecaster.fit(y=data['y'])
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

<pre>
array([0.55361079, 0.56832448, 0.73516725, 0.72321715])
</pre>



## Backtest with a custom metric

`backtesting_forecaster()` function allows a calleable metric.

``` python
def custom_metric(y_true, y_pred):
    '''
    Custom metric function
    '''
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric

metric, predictions_backtest = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['y'],
                                    initial_train_size = len(data_train),
                                    fixed_train_size   = False,
                                    steps      = 10,
                                    metric     = custom_metric,
                                    refit      = True,
                                    verbose    = False
                               )

print(f"Backtest error custom metric: {metric}")
```

<pre>
Backtest error custom metric: 3.136057139917678e-05
</pre>



## Backtest with exogenous variables

``` python
# Download data
# ==============================================================================
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv')
data = pd.read_csv(url, sep=',', header=0, names=['datetime', 'y', 'exog_1', 'exog_2'])

# Data preprocessing
# ==============================================================================
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d')
data = data.set_index('datetime')
data = data.asfreq('MS')
data = data.sort_index()

# Split data in train and backtest
# ==============================================================================
n_backtest = 36*3  # Last 9 years are used for backtest
data_train = data[:-n_backtest]
data_backtest = data[-n_backtest:]

# Plot
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
data.plot(ax=ax);
```

<img src="../img/data_exog.png" style="width: 500px;">

``` python
# Backtest forecaster with exogenous variables
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15 
             )

metric, predictions_backtest = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['y'],
                                    exog       = data[['exog_1', 'exog_2']],
                                    initial_train_size = len(data_train),
                                    fixed_train_size   = False,
                                    steps      = 10,
                                    metric     = 'mean_squared_error',
                                    refit      = True,
                                    verbose    = True
                               )
```

<pre>
Information of backtesting process
----------------------------------
Number of observations used for initial training: 87
Number of observations used for backtesting: 108
    Number of folds: 11
    Number of steps per fold: 10
    Last fold only includes 8 observations.

Data partition in fold: 0
    Training:   1992-04-01 00:00:00 -- 1999-06-01 00:00:00
    Validation: 1999-07-01 00:00:00 -- 2000-04-01 00:00:00
Data partition in fold: 1
    Training:   1992-04-01 00:00:00 -- 2000-04-01 00:00:00
    Validation: 2000-05-01 00:00:00 -- 2001-02-01 00:00:00
Data partition in fold: 2
    Training:   1992-04-01 00:00:00 -- 2001-02-01 00:00:00
    Validation: 2001-03-01 00:00:00 -- 2001-12-01 00:00:00
Data partition in fold: 3
    Training:   1992-04-01 00:00:00 -- 2001-12-01 00:00:00
    Validation: 2002-01-01 00:00:00 -- 2002-10-01 00:00:00
Data partition in fold: 4
    Training:   1992-04-01 00:00:00 -- 2002-10-01 00:00:00
    Validation: 2002-11-01 00:00:00 -- 2003-08-01 00:00:00
Data partition in fold: 5
    Training:   1992-04-01 00:00:00 -- 2003-08-01 00:00:00
    Validation: 2003-09-01 00:00:00 -- 2004-06-01 00:00:00
Data partition in fold: 6
    Training:   1992-04-01 00:00:00 -- 2004-06-01 00:00:00
    Validation: 2004-07-01 00:00:00 -- 2005-04-01 00:00:00
Data partition in fold: 7
    Training:   1992-04-01 00:00:00 -- 2005-04-01 00:00:00
    Validation: 2005-05-01 00:00:00 -- 2006-02-01 00:00:00
Data partition in fold: 8
    Training:   1992-04-01 00:00:00 -- 2006-02-01 00:00:00
    Validation: 2006-03-01 00:00:00 -- 2006-12-01 00:00:00
Data partition in fold: 9
    Training:   1992-04-01 00:00:00 -- 2006-12-01 00:00:00
    Validation: 2007-01-01 00:00:00 -- 2007-10-01 00:00:00
Data partition in fold: 10
    Training:   1992-04-01 00:00:00 -- 2007-10-01 00:00:00
    Validation: 2007-11-01 00:00:00 -- 2008-06-01 00:00:00
</pre>

``` python
print(f"Backtest error with exogenous variables: {metric}")
```

<pre>
Backtest error with exogenous variables: 0.0068340846122841425
</pre>

``` python
fig, ax = plt.subplots(figsize=(9, 4))
data_backtest.plot(ax=ax)
predictions_backtest.plot(ax=ax)
ax.legend();
```

<img src="../img/predictions_backtesting_forecaster_with_exog.png" style="width: 500px;">