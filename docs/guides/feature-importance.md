# Feature importance

The importance of predictors included in a forecaster can be obtained using the method `get_feature_importance`. This method access the attributes `coef_` and `feature_importances_` of the internal regressor.

> **⚠ WARNING:**  
> This methods only return values if the regressor used inside the forecaster has the attribute `coef_` or `feature_importances_`.



## Libraries

``` python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
```
## Data

``` python
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv')
data = pd.read_csv(url, sep=',', header=0, names=['date', 'y', 'exog_1', 'exog_2'])

data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data = data.set_index('date')
data = data.asfreq('MS')
```

## Extract feature importance from trained forecaster


``` python
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(),
                lags = 5
             )

forecaster.fit(y=data['y'], exog=data[['exog_1', 'exog_2']])

forecaster.get_feature_importance()
```

|    | feature   |   importance |
|----|-----------|--------------|
|  0 | lag_1     |    0.522544  |
|  1 | lag_2     |    0.0998894 |
|  2 | lag_3     |    0.0203564 |
|  3 | lag_4     |    0.078274  |
|  4 | lag_5     |    0.067372  |
|  5 | exog_1    |    0.0483118 |
|  6 | exog_2    |    0.163252  |

``` python
orecaster = ForecasterAutoreg(
                    regressor = Ridge(),
                    lags = 5
                )

forecaster.fit(y=data['y'], exog=data[['exog_1', 'exog_2']])

forecaster.get_feature_importance()
```

|    | feature   |       coef |
|----|-----------|------------|
|  0 | lag_1     |  0.327688  |
|  1 | lag_2     | -0.0735932 |
|  2 | lag_3     | -0.152202  |
|  3 | lag_4     | -0.217106  |
|  4 | lag_5     | -0.1458    |
|  5 | exog_1    |  0.379798  |
|  6 | exog_2    |  0.668162  |


When using a `ForecasterAutoregMultiOutput`, since a different model is fit for each step, it is necessary to indicate which model to extract the information from.

``` python
forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(),
                steps = 10,
                lags = 5
             )

forecaster.fit(y=data['y'], exog=data[['exog_1', 'exog_2']])

forecaster.get_feature_importance(step=1)
```

|    | feature   |   importance |
|----|-----------|--------------|
|  0 | lag_1     |    0.529107  |
|  1 | lag_2     |    0.109431  |
|  2 | lag_3     |    0.0197598 |
|  3 | lag_4     |    0.0813373 |
|  4 | lag_5     |    0.0480799 |
|  5 | exog_1    |    0.0371122 |
|  6 | exog_2    |    0.175173  |