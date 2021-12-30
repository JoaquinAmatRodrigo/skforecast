# Recursive multi-step forecasting

Since the value of *t(n)* is required to predict the point *t(n-1)*, and *t(n-1)* is unknown, it is necessary to make recursive predictions in which, each new prediction, is based on the previous one. This process is known as recursive forecasting or recursive multi-step forecasting.

The main challenge when using machine learning models for recursive multi-step forecasting is transforming the time series in an matrix where, each value of the series, is related to the time window (lags) that precedes it. This forecasting strategy can be easily generated with the classes `ForecasterAutoreg` and `ForecasterAutoregCustom`.

<img src="../img/matrix_transformation_time_serie.png" style="width: 500px;">



## Libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```
## Data

``` python

url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv')
data = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])

data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d')
data = data.set_index('datetime')
data = data.asfreq('MS')
data = data['y']
data = data.sort_index()

steps = 36
data_train = data[:-steps]
data_test  = data[-steps:]

fig, ax=plt.subplots(figsize=(9, 4))
data_train.plot(ax=ax, label='train')
data_test.plot(ax=ax, label='test')
ax.legend()
```

<img src="../img/data.png" style="width: 500px;">


## Create and train forecaster


``` python
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 15
             )

forecaster.fit(y=data_train)
forecaster
```

```
================= 
ForecasterAutoreg 
================= 
Regressor: RandomForestRegressor(random_state=123)
Lags: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15] 
Window size: 15 
Included exogenous: False 
Type of exogenous variable: None 
Exogenous variables names: None 
Training range: [Timestamp('1991-07-01 00:00:00'), Timestamp('2005-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False} 
Creation date: 2021-12-06 23:22:17 
Last fit date: 2021-12-06 23:22:17 
Skforecast version: 0.4.1
```

## Prediction 

``` python
predictions = forecaster.predict(steps=36)
predictions.head(3)
```

```
2005-07-01    0.921840
2005-08-01    0.954921
2005-09-01    1.101716
Freq: MS, Name: pred, dtype: float64
```

``` python
fig, ax=plt.subplots(figsize=(9, 4))
data_train.plot(ax=ax, label='train')
data_test.plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend()
```

<img src="../img/prediction.png" style="width: 500px;">

``` python
error_mse = mean_squared_error(
                y_true = data_test,
                y_pred = predictions
            )
print(f"Test error (mse): {error_mse}")
```

```
Test error (mse): 0.00429855684785846
```

## Feature importance

``` python
forecaster.get_feature_importance()
```

| feature   |   importance |
|:----------|-------------:|
| lag_1     |   0.0123397  |
| lag_2     |   0.0851603  |
| lag_3     |   0.0134071  |
| lag_4     |   0.00437446 |
| lag_5     |   0.00318805 |
| lag_6     |   0.00343593 |
| lag_7     |   0.00313612 |
| lag_8     |   0.00714094 |
| lag_9     |   0.00783053 |
| lag_10    |   0.0127507  |
| lag_11    |   0.00901919 |
| lag_12    |   0.807098   |
| lag_13    |   0.00481128 |
| lag_14    |   0.0163282  |
| lag_15    |   0.0099792  |

## Extract training matrix

``` python
X, y = forecaster.create_train_X_y(data_train)
print(X)
print(y)
```

| datetime            |    lag_1 |    lag_2 |    lag_3 |    lag_4 |    lag_5 |    lag_6 |    lag_7 |    lag_8 |    lag_9 |   lag_10 |   lag_11 |   lag_12 |   lag_13 |   lag_14 |   lag_15 |
|:--------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1992-10-01 00:00:00 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 | 0.400906 | 0.429795 |
| 1992-11-01 00:00:00 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 | 0.400906 |
| 1992-12-01 00:00:00 | 0.595223 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 |
| 1993-01-01 00:00:00 | 0.771258 | 0.595223 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 |
| 1993-02-01 00:00:00 | 0.751503 | 0.771258 | 0.595223 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 |

| datetime            |        y |
|:--------------------|---------:|
| 1992-10-01 00:00:00 | 0.568606 |
| 1992-11-01 00:00:00 | 0.595223 |
| 1992-12-01 00:00:00 | 0.771258 |
| 1993-01-01 00:00:00 | 0.751503 |
| 1993-02-01 00:00:00 | 0.387554 |
