# Unit test __init__ ForecasterSarimax
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import y_datetime


def test_ForecasterSarimax_get_info_criteria_ValueError_criteria_invalid_value():
    """
    Test ForecasterSarimax get_info_criteria ValueError when `criteria` is an invalid value.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    criteria = 'not_valid'

    err_msg = re.escape(
                (f"Invalid value for `criteria`. Valid options are 'aic', 'bic', "
                 f"and 'hqic'.")
              )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(criteria=criteria)


def test_ForecasterSarimax_get_info_criteria_ValueError_method_invalid_value():
    """
    Test ForecasterSarimax get_info_criteria ValueError when `method` is an invalid value.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    method = 'not_valid'

    err_msg = re.escape(
                (f"Invalid value for `method`. Valid options are 'standard' and "
                 f"'lutkepohl'.")
              )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(method=method)


def test_Sarimax_get_info_criteria():
    """
    Test ForecasterSarimax get_info_criteria after fit `y`.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    results = forecaster.get_info_criteria(criteria='aic', method='standard')

    expected = -56.80222086732853

    assert results == expected