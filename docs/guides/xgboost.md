# Forecasting with XGBoost

XGBoost, acronym for Extreme Gradient Boosting, is a very efficient implementation of the stochastic gradient boosting algorithm that has become a benchmark in the field of machine learning. In addition to its own API, XGBoost library includes the XGBRegressor class which follows the scikit learn API and therefore it is compatible with skforecast.

> **NOTE:**  
> Since the success of XGBoost as a machine learning algorithm, new implementations have been developed that also achieve excellent results, two of them are: [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and [CatBoost](https://catboost.ai/). A more detailed example can be found [here](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html).



## Libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
```
## Data

``` python
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv')
data = pd.read_csv(url, sep=',', header=0, names=['date', 'y', 'exog_1', 'exog_2'])

data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data = data.set_index('date')
data = data.asfreq('MS')

steps = 36
data_train = data.iloc[:-steps, :]
data_test  = data.iloc[-steps:, :]
```

## Create and train forecaster


``` python
forecaster = ForecasterAutoreg(
                    regressor = XGBRegressor(),
                    lags = 8
                )

forecaster.fit(y=data['y'], exog=data[['exog_1', 'exog_2']])
forecaster
```

```
================= 
ForecasterAutoreg 
================= 
Regressor: XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None) 
Lags: [1 2 3 4 5 6 7 8] 
Window size: 8 
Included exogenous: True 
Type of exogenous variable: <class 'pandas.core.frame.DataFrame'> 
Exogenous variables names: ['exog_1', 'exog_2'] 
Training range: [Timestamp('1992-04-01 00:00:00'), Timestamp('2008-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'objective': 'reg:squarederror', 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.300000012, 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 100, 'n_jobs': 8, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 1, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None} 
Creation date: 2021-12-30 17:24:05 
Last fit date: 2021-12-30 17:24:05 
Skforecast version: 0.4.1
```

## Prediction 

``` python
forecaster.predict(steps=10, exog=data_test[['exog_1', 'exog_2']])
```

<pre>
2008-07-01    0.700701
2008-08-01    0.829139
2008-09-01    0.983677
2008-10-01    1.098782
2008-11-01    1.078021
2008-12-01    1.206761
2009-01-01    1.149827
2009-02-01    1.049927
2009-03-01    0.947129
2009-04-01    0.700440
Freq: MS, Name: pred, dtype: float64
</pre>

## Feature importance

``` python
forecaster.get_feature_importance()
```

|    | feature   |   importance |
|----|-----------|--------------|
|  0 | lag_1     |    0.358967  |
|  1 | lag_2     |    0.0935667 |
|  2 | lag_3     |    0.0167286 |
|  3 | lag_4     |    0.0446115 |
|  4 | lag_5     |    0.054733  |
|  5 | lag_6     |    0.0095096 |
|  6 | lag_7     |    0.0971787 |
|  7 | lag_8     |    0.0279641 |
|  8 | exog_1    |    0.186294  |
|  9 | exog_2    |    0.110447  |