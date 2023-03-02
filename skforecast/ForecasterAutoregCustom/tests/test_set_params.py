# Unit test set_params ForecasterAutoregCustom
# ==============================================================================
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags  


def test_set_paramns():
    """
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(fit_intercept=True),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    new_params = {'fit_intercept': False}
    forecaster.set_params(new_params)
    expected = {'copy_X': True,
                 'fit_intercept': False,
                 'n_jobs': None,
                 'normalize': 'deprecated', #For sklearn < 1.2
                 'positive': False
                }
    results = forecaster.regressor.get_params()
    results.update({'normalize': 'deprecated'})
    assert results == expected