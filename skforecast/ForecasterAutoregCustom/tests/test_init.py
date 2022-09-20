# Unit test __init__ ForecasterAutoregCustom
# ==============================================================================
import pytest
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags  


def test_init_exception_when_window_size_argument_is_string():
   """
   """
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = '5'
                    )

def test_init_exception_when_fun_predictors_argument_is_string():
   """
   """
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = 'create_predictors',
                        window_size    = 5
                    )