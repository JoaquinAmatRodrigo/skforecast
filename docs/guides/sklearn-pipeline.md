# Forecasting with scikit-learn pipelines

Since version 0.4.0, skforecast allows using scikit-learn pipelines as regressors. This is useful since, many machine learning models, need specific data preprocessing transformations. For example, linear models with Ridge or Lasso regularization benefits from features been scaled.

> **⚠ WARNING:**  
> Version 0.4 does not allow including ColumnTransformer in the pipeline used as regressor, so if the preprocessing transformations only apply to some specific columns, they have to be applied on the data set before training the model. A more detailed example can be found [here](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html#Exogenous-variables).

## Libraries

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
```
## Data

``` python
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv')
data = pd.read_csv(url, sep=',', header=0, names=['date', 'y', 'exog_1', 'exog_2'])

data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data = data.set_index('date')
data = data.asfreq('MS')
```

## Create pipeline

``` python
pipe = make_pipeline(StandardScaler(), Ridge())
pipe
```

```
Pipeline(steps=[('standardscaler', StandardScaler()), ('ridge', Ridge())])
```

## Create and train forecaster


``` python
pipe = make_pipeline(StandardScaler(), Ridge())
forecaster = ForecasterAutoreg(
                    regressor = pipe,
                    lags = 10
                )

forecaster.fit(y=data['y'], exog=data[['exog_1', 'exog_2']])
forecaster
```

```
================= 
ForecasterAutoreg 
================= 
Regressor: Pipeline(steps=[('standardscaler', StandardScaler()), ('ridge', Ridge())]) 
Lags: [ 1  2  3  4  5  6  7  8  9 10] 
Window size: 10 
Included exogenous: True 
Type of exogenous variable: <class 'pandas.core.frame.DataFrame'> 
Exogenous variables names: ['exog_1', 'exog_2'] 
Training range: [Timestamp('1992-04-01 00:00:00'), Timestamp('2008-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'ridge__alpha': 1.0, 'ridge__copy_X': True, 'ridge__fit_intercept': True, 'ridge__max_iter': None, 'ridge__normalize': 'deprecated', 'ridge__positive': False, 'ridge__random_state': None, 'ridge__solver': 'auto', 'ridge__tol': 0.001} 
Creation date: 2021-12-30 17:30:04 
Last fit date: 2021-12-30 17:30:04 
Skforecast version: 0.4.1
```

## Grid Search

When performing grid search over a sklearn pipeline, the name of the parameters is preceded by the name of the model.

``` python
pipe = make_pipeline(StandardScaler(), Ridge())
forecaster = ForecasterAutoreg(
                    regressor = pipe,
                    lags = 10  # This value will be replaced in the grid search
                )

# Regressor's hyperparameters
param_grid = {'ridge__alpha': np.logspace(-3, 5, 10)}

# Lags used as predictors
lags_grid = [5, 24, [1, 2, 3, 23, 24]]

results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = data['y'],
                        exog        = data[['exog_1', 'exog_2']],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = 5,
                        metric      = 'mean_absolute_error',
                        refit       = False,
                        initial_train_size = len(data.loc[:'2000-04-01']),
                        return_best = True,
                        verbose     = False
                  )
```

|    | lags                                                                      | params                                 |      metric |    ridge__alpha |
|----|---------------------------------------------------------------------------|----------------------------------------|-------------|-----------------|
|  0 | [1 2 3 4 5]                                                               | {'ridge__alpha': 0.001}                | 6.84531e-05 |      0.001      |
| 10 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 0.001}                | 0.000187797 |      0.001      |
|  1 | [1 2 3 4 5]                                                               | {'ridge__alpha': 0.007742636826811269} | 0.000526168 |      0.00774264 |
| 11 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 0.007742636826811269} | 0.00141293  |      0.00774264 |
|  2 | [1 2 3 4 5]                                                               | {'ridge__alpha': 0.05994842503189409}  | 0.00385988  |      0.0599484  |
| 12 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 0.05994842503189409}  | 0.00896885  |      0.0599484  |
|  3 | [1 2 3 4 5]                                                               | {'ridge__alpha': 0.46415888336127775}  | 0.0217507   |      0.464159   |
| 13 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 0.46415888336127775}  | 0.0295054   |      0.464159   |
| 14 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 3.593813663804626}    | 0.046323    |      3.59381    |
| 23 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 0.46415888336127775}  | 0.0606231   |      0.464159   |
| 22 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 0.05994842503189409}  | 0.0615665   |      0.0599484  |
| 21 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 0.007742636826811269} | 0.0617473   |      0.00774264 |
| 20 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 0.001}                | 0.0617715   |      0.001      |
| 24 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 3.593813663804626}    | 0.0635121   |      3.59381    |
| 15 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 27.825594022071257}   | 0.0645505   |     27.8256     |
|  4 | [1 2 3 4 5]                                                               | {'ridge__alpha': 3.593813663804626}    | 0.0692201   |      3.59381    |
| 25 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 27.825594022071257}   | 0.077934    |     27.8256     |
| 16 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 215.44346900318823}   | 0.130016    |    215.443      |
|  5 | [1 2 3 4 5]                                                               | {'ridge__alpha': 27.825594022071257}   | 0.143189    |     27.8256     |
| 26 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 215.44346900318823}   | 0.146446    |    215.443      |
| 17 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 1668.1005372000557}   | 0.204469    |   1668.1        |
|  6 | [1 2 3 4 5]                                                               | {'ridge__alpha': 215.44346900318823}   | 0.205496    |    215.443      |
| 27 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 1668.1005372000557}   | 0.212896    |   1668.1        |
| 18 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 12915.496650148827}   | 0.227536    |  12915.5        |
| 28 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 12915.496650148827}   | 0.228974    |  12915.5        |
| 19 | [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] | {'ridge__alpha': 100000.0}             | 0.231157    | 100000          |
| 29 | [ 1  2  3 23 24]                                                          | {'ridge__alpha': 100000.0}             | 0.231356    | 100000          |
|  7 | [1 2 3 4 5]                                                               | {'ridge__alpha': 1668.1005372000557}   | 0.236227    |   1668.1        |
|  8 | [1 2 3 4 5]                                                               | {'ridge__alpha': 12915.496650148827}   | 0.244788    |  12915.5        |
|  9 | [1 2 3 4 5]                                                               | {'ridge__alpha': 100000.0}             | 0.246091    | 100000          |