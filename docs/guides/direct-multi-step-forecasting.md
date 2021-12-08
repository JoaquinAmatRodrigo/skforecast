# Direct multi-step forecaster

`ForecasterAutoreg` and `ForecasterAutoregCustom` models follow a recursive prediction strategy in which, each new prediction, builds on the previous prediction. An alternative is to train a model for each step that has to be predicted. This strategy, commonly known as direct multistep forecasting, is computationally more expensive than the recursive since it requires training several models. However, in some scenarios, it achieves better results. This type of model can be obtained with the `ForecasterAutoregMultiOutput` class and can also include one or multiple exogenous variables.

In order to train a `ForecasterAutoregMultiOutput` a different training matrix is created for each model.

<img src="../img/diagram_skforecast_multioutput.jpg">

## Libraries

``` python
# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import Ridge
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
<img src="../img/data.png" style="width: 500px;">


## Create and train forecaster


``` python
# Create and fit forecaster
# ==============================================================================
forecaster = ForecasterAutoregMultiOutput(
                    regressor = make_pipeline(StandardScaler(), Ridge()),
                    steps     = 36,
                    lags      = 15
             )

forecaster.fit(y=data_train)
forecaster
```

```
============================ 
ForecasterAutoregMultiOutput 
============================ 
Regressor: Pipeline(steps=[('standardscaler', StandardScaler()), ('ridge', Ridge())]) 
Lags: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15] 
Window size: 15 
Maximum steps predicted: 36 
Included exogenous: False 
Type of exogenous variable: None 
Exogenous variables names: None 
Training range: [Timestamp('1991-07-01 00:00:00'), Timestamp('2005-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'ridge__alpha': 1.0, 'ridge__copy_X': True, 'ridge__fit_intercept': True, 'ridge__max_iter': None, 'ridge__normalize': 'deprecated', 'ridge__positive': False, 'ridge__random_state': None, 'ridge__solver': 'auto', 'ridge__tol': 0.001} 
Creation date: 2021-12-07 21:30:45 
Last fit date: 2021-12-07 21:30:45 
Skforecast version: 0.4.0 

```

## Prediction

If the `Forecaster` has been trained with exogenous variables, they shlud be provided when predictiong.


``` python
# Predict
# ==============================================================================
predictions = forecaster.predict(steps=36)
predictions.head(3)
```

```
2005-07-01    0.965991
2005-08-01    0.973200
2005-09-01    1.144204
Freq: MS, Name: pred, dtype: float64
```

``` python
# Plot predictions
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
data_train.plot(ax=ax, label='train')
data_test.plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend();
```

<img src="../img/prediction_with_direct_multi_output.png" style="width: 500px;">

``` python
# Prediction error
# ==============================================================================
error_mse = mean_squared_error(
                y_true = data_test,
                y_pred = predictions
            )
print(f"Test error (mse): {error_mse}")
```

```
Test error (mse): 0.009067941608532212
```

## Feature importance

Since `ForecasterAutoregMultiOutput` fits one model per step,it is necessary to specify from which model retrieve its feature importance.

``` python
forecaster.get_coef(step=1)
```

```
| feature   |         coef |
|:----------|-------------:|
| lag_1     |  0.0306858   |
| lag_2     |  0.0407212   |
| lag_3     |  0.0345991   |
| lag_4     | -0.0018438   |
| lag_5     | -0.00110815  |
| lag_6     |  0.000759624 |
| lag_7     | -0.00282664  |
| lag_8     |  0.00109809  |
| lag_9     | -0.0046271   |
| lag_10    |  0.00610878  |
| lag_11    |  0.0054867   |
| lag_12    |  0.159351    |
| lag_13    | -0.0260086   |
| lag_14    | -0.0460312   |
| lag_15    | -0.0358484   |
```

## Extract training matrix

Two steps are needed. One to create the whole training matrix and a second one to subset the data needed for each model (step).

``` python
X, y = forecaster.create_train_X_y(data_train)
# X and y to train model for step 1
X_1, y_1 = forecaster.filter_train_X_y_for_step(
                step    = 1,
                X_train = X,
                y_train = y,
            )
print(X_1.head(4))
print(y_1.head(4))
```

```
|    lag_1 |    lag_2 |    lag_3 |    lag_4 |    lag_5 |    lag_6 |    lag_7 |    lag_8 |    lag_9 |   lag_10 |   lag_11 |   lag_12 |   lag_13 |   lag_14 |   lag_15 |
|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 | 0.400906 | 0.429795 |
| 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 | 0.400906 |
| 0.595223 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 | 0.432159 |
| 0.771258 | 0.595223 | 0.568606 | 0.534761 | 0.475463 | 0.483389 | 0.410534 | 0.361801 | 0.379808 | 0.351348 | 0.33622  | 0.660119 | 0.602652 | 0.502369 | 0.492543 |

|   y_step_1 |
|-----------:|
|   0.595223 |
|   0.771258 |
|   0.751503 |
|   0.387554 |
```