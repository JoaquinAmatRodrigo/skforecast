# Unit test _recursive_predict ForecasterAutoregCustom
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    '''
    Create first 5 lags of a time series.
    '''
    
    lags = y[-1:-6:-1]
    
    return lags  


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    '''
    Test _recursive_predict output when using LinearRegression as regressor.
    '''
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