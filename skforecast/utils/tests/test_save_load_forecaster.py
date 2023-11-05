# Unit test save_forecaster and load_forecaster
# ==============================================================================
import os
import re
import joblib
import pytest
import numpy as np
import pandas as pd
import warnings
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from skforecast.exceptions import SkforecastVersionWarning
from sklearn.linear_model import LinearRegression


def test_save_and_load_forecaster_persistence():
    """ 
    Test if a loaded forecaster is exactly the same as the original one.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    save_forecaster(forecaster=forecaster, file_name='forecaster.py', verbose=True)
    forecaster_loaded = load_forecaster(file_name='forecaster.py', verbose=True)
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


def test_save_and_load_forecaster_SkforecastVersionWarning():
    """ 
    Test warning used to notify that the skforecast version installed in the 
    environment differs from the version used to create the forecaster.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    forecaster.skforecast_version = '0.0.0'
    save_forecaster(forecaster=forecaster, file_name='forecaster.py', verbose=False)

    warn_msg = re.escape(
        (f"The skforecast version installed in the environment differs "
         f"from the version used to create the forecaster.\n"
         f"    Installed Version  : {skforecast.__version__}\n"
         f"    Forecaster Version : 0.0.0\n"
         f"This may create incompatibilities when using the library.")
    )
    with pytest.warns(SkforecastVersionWarning, match = warn_msg):
        load_forecaster(file_name='forecaster.py', verbose=False)
        os.remove('forecaster.py')