# Unit test __init__ ForecasterAutoregCustom
# ==============================================================================
import re
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
   window_size = '5'
   err_msg = re.escape(
                f'`window_size` must be int, got {type(window_size)}'
            )
   with pytest.raises(TypeError, match = err_msg):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = window_size
                    )

def test_init_exception_when_fun_predictors_argument_is_string():
   """
   """
   fun_predictors = 'create_predictors'
   err_msg = re.escape(
                f'`fun_predictors` must be callable, got {type(fun_predictors)}.'
            )
   with pytest.raises(TypeError, match = err_msg):
        forecaster = ForecasterAutoregCustom(
                         regressor      = LinearRegression(),
                         fun_predictors = fun_predictors,
                         window_size    = 5
                     )