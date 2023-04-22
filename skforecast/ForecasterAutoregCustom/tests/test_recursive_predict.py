# Unit test _recursive_predict ForecasterAutoregCustom
# ==============================================================================
from pytest import approx
import pytest
import re
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags  


def test_recursive_predict_ValueError_when_fun_predictors_return_nan():
    """
    Test ValueError is raised when y and exog have different index.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(
        y = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'))
    )

    err_msg = re.escape("`fun_predictors()` is returning `NaN` values.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster._recursive_predict(last_window=np.array([np.nan]), steps=3) 


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster._recursive_predict(
                    steps = 5,
                    last_window = forecaster.last_window.values,
                    exog = None
                  )
    expected = np.array([50., 51., 52., 53., 54.])
    
    assert (predictions == approx(expected))