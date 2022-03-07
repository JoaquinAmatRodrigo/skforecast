# Use forecaster in production

A trained model may be deployed in production in order to generate predictions regularly. Suppose predictions have to be generated on a weekly basis, for example, every Monday. By default, when using the `predict` method on a trained forecaster object, predictions start right after the last training observation. Therefore, the model could be retrained weekly, just before the first prediction is needed, and call its predict method. This strategy, although simple, may not be possible to use for several reasons:


+ Model training is very expensive and cannot be run as often.

+ The history with which the model was trained is no longer available.

+ The prediction frequency is so high that there is no time to train the model between predictions.

In these scenarios, the model must be able to predict at any time, even if it has not been recently trained.

Every model generated using skforecast has the `last_window` argument in its `predict` method. Using this argument, it is possible to provide only the past values needs to create the autoregressive predictors (lags) and thus, generate the predictions without the need to retrain the model.


## Libraries

``` python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
```

## Data

``` python
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv')
data = pd.read_csv(url, sep=',', header=0, names=['y', 'date'])

data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data = data.set_index('date')
data = data.asfreq('MS')
data_train = data.loc[:'2005-01-01']
data_train.tail()
```

| date                |       y |
|:--------------------|--------:|
| 2004-09-01 00:00:00 | 1.13443 |
| 2004-10-01 00:00:00 | 1.18101 |
| 2004-11-01 00:00:00 | 1.21604 |
| 2004-12-01 00:00:00 | 1.25724 |
| 2005-01-01 00:00:00 | 1.17069 |


## Predict

``` python
forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(),
                    lags = 5
                )

forecaster.fit(y=data_train['y'])
```

<pre>
2005-02-01    0.927480
2005-03-01    0.756215
2005-04-01    0.692595
Freq: MS, Name: pred, dtype: float64
</pre>


As expected, predictions follow directly from the end of training data.

When `last window` is provided, the forecaster uses this data to generate the lags needed as predictors and starts the prediction afterwards.

``` python
forecaster.predict(steps=3, last_window=data['y'].tail(5))
```

<pre>
2008-07-01    0.803853
2008-08-01    0.870858
2008-09-01    0.905003
Freq: MS, Name: pred, dtype: float64
</pre>

Since the provided `last_window` contains values from 2008-02-01 to 2008-06-01, the forecaster is able to create the needed lags and predict the next 5 steps.


!!! warning

     It is important to note that the length of last windows must be enough to include the maximum lag used by the forecaster. Fore example, if the forecaster uses lags 1, 24, 48 and 72 `last_window` must include the last 72 values of the series.
