# Use forecaster after training

By default, when using `predict` method on a trained forecaster object, predictions starts right after the last training observation.



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
## Predict

By default, when using `predict` method on a trained forecaster object, predictions starts right after the last training observation.

``` python
forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(),
                    lags = 5
                )

forecaster.fit(y=data_train['y'])
```

<pre>
2005-02-01    0.872294
2005-03-01    0.679100
2005-04-01    0.759507
Freq: MS, Name: pred, dtype: float64
</pre>

As expected, predictions follow directly from the end of training data.

If the training sample is relatively small or if it is desired compute the best possible forecasts, the forecaster should be retrained using all the available data before making predictions. However, if that strategy is infeasible (for example, because the training set is very large), it is useful to generate predictions without retraining the model each time.

With skforecast, it is possible to generate predictions starting time ahead of training date. When `last_window` is provided, the forecaster use this data to generate the lads needed as predictors.

``` python
forecaster.predict(steps=3, last_window=data['y'].tail(5))
```

<pre>
2008-07-01    0.783482
2008-08-01    0.865240
2008-09-01    0.897055
Freq: MS, Name: pred, dtype: float64
</pre>

Since the provided `last_window` contains values from 2008-02-01 to 2008-06-01, the forecaster is able to create the needed lags and predict the next 5 steps.


> **⚠ WARNING:**  
> It is important to note that the length of last windows must be enough to include the maximum lag used by the forecaster. Fore example, if the forecaster uses lags 1, 24, 48 and 72 `last_window` must include the last 72 values of the series.
