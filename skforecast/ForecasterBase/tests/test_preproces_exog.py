import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression

def test_preproces_exog_when_exog_is_pandas_series():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    assert (forecaster._preproces_exog(exog=pd.Series([0, 1, 2])) == np.arange(3).reshape(-1, 1)).all()
    

def test_preproces_exog_when_exog_is_1d_numpy_array():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    assert (forecaster._preproces_exog(exog=np.arange(3)) == np.arange(3).reshape(-1, 1)).all()
    