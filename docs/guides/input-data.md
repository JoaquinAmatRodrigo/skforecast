# Input data

Since **Version 0.4.0** only pandas series and dataframes are allowed (although internally numpy arrays are used for performance). Base on the type of pandas index, the following rules are applied:

+ If index is not of type DatetimeIndex, a RangeIndex is created.

+ If index is of type DatetimeIndex but has no frequency, a RangeIndex is created.

+ If index is of type DatetimeIndex and has frequency, nothing is changed.

!!! Note

    There is nothing wrong with using data that does not have an associated DatetimeIndex with frequency. However, results will have less informative index.



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
```

## Train and predict using input with datetime and frequency index


``` python
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 5
             )

forecaster.fit(y=data['y'])

forecaster.predict(steps=5)
```

<pre>
2008-07-01    0.714526
2008-08-01    0.789144
2008-09-01    0.818433
2008-10-01    0.845027
2008-11-01    0.914621
Freq: MS, Name: pred, dtype: float64
</pre>

## Train and predict using input without datetime index


``` python
data = data.reset_index(drop=True)
data
```

|    |        y |
|---:|---------:|
|  0 | 0.429795 |
|  1 | 0.400906 |
|  2 | 0.432159 |
|  3 | 0.492543 |
|  4 | 0.502369 |


``` python
forecaster.fit(y=data['y'])
forecaster.predict(steps=5)
```

<pre>
204    0.714526
205    0.789144
206    0.818433
207    0.845027
208    0.914621
Name: pred, dtype: float64
</pre>

