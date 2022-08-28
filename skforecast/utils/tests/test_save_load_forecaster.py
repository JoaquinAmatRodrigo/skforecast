# Unit test save_forecaster and load_forecaster
# ==============================================================================
import os
import joblib
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from sklearn.linear_model import LinearRegression


def test_save_and_load_forecaster_persistence():
    """ 
    Test if a loaded forecaster is exactly the same as the original one.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    save_forecaster(forecaster=forecaster, file_name='forecaster.py', verbose=False)
    forecaster_loaded = load_forecaster(file_name='forecaster.py', verbose=False)
    os.remove('forecaster.py')

    for key in vars(forecaster).keys():
        attribute_forecaster = forecaster.__getattribute__(key)
        attribute_forecaster_loaded = forecaster_loaded.__getattribute__(key)

        if key == 'regressor':
            assert joblib.hash(attribute_forecaster) == joblib.hash(attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, (np.ndarray, pd.Series, pd.DataFrame, pd.Index)):
            assert (attribute_forecaster == attribute_forecaster_loaded).all()
        else:
            assert attribute_forecaster == attribute_forecaster_loaded