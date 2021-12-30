# Save and load forecaster

Skforecast models can be loaded and stored using pickle or joblib library. A simple example using joblib is shown below.


## Libraries

``` python
import numpy as np
import pandas as pd
from joblib import dump, load
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

## Save and load model


``` python
# Create and train forecaster
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 5
             )

forecaster.fit(y=data['y'])
forecaster.predict(steps=3)
```

```
2008-07-01    0.714526
2008-08-01    0.789144
2008-09-01    0.818433
Freq: MS, Name: pred, dtype: float64
```

``` python
# Save model
dump(forecaster, filename='forecaster.py')

# Load model
forecaster_loaded = load('forecaster.py')

# Predict
forecaster_loaded.predict(steps=3)
```

```
2008-07-01    0.714526
2008-08-01    0.789144
2008-09-01    0.818433
Freq: MS, Name: pred, dtype: float64
```

