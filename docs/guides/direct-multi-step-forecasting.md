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
from sklearn.ensemble import GradientBoostingRegressor
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
<img src="../img/data.png" style="width: 500px;">


## Create and train forecaster


``` python
# Create and fit forecaster
# ==============================================================================
forecaster = ForecasterAutoregMultiOutput(
                    regressor = GradientBoostingRegressor(),
                    steps     = 36,
                    lags      = 15
                )

forecaster.fit(y=data_train)
forecaster
```

```
============================ForecasterAutoregMultiOutput============================
Regressor: GradientBoostingRegressor()
Steps: 36
Lags: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
Exogenous variable: False
Parameters: {'alpha': 0.9, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'ls', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': None, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

```

## Prediction

If the `Forecaster` has been trained with exogenous variables, they shlud be provided when predictiong.


``` python
# Predict
# ==============================================================================
steps = 36
predictions = forecaster.predict(steps=steps)
# Add datetime index to predictions
predictions = pd.Series(data=predictions, index=data_test.index)
predictions.head(3)
```

```
fecha
2005-07-01    0.877073
2005-08-01    0.974353
2005-09-01    1.021718
Freq: MS, dtype: float64
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
Test error (mse): 0.006208145903529839
```

## Feature importance

Since `ForecasterAutoregMultiOutput` fits one model per step,it is necessary to specify from which model retrieve its feature importance.

``` python
# When using as regressor LinearRegression, Ridge or Lasso
# forecaster.get_coef()

# When using as regressor RandomForestRegressor or GradientBoostingRegressor
forecaster.get_feature_importances(step=1)
```

```
array([0.0070333 , 0.07330653, 0.03192484, 0.00284055, 0.00525716,
       0.0039042 , 0.00717631, 0.00676659, 0.00587334, 0.00856639,
       0.01275252, 0.81823596, 0.005138  , 0.00413031, 0.007094])
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
```