# Unit test __init__ ForecasterSarimax
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima.arima import ARIMA
from sklearn.linear_model import LinearRegression


def test_exception_when_regressor_is_pmdarima_ARIMA_when_initialization():
    """
    Raise exception if regressor is not of type pmdarima.arima.ARIMA when initializing the forecaster.
    """
    regressor = LinearRegression()

    err_msg = re.escape(
                (f"`regressor` must be an instance of type pmdarima.arima.ARIMA. "
                 f"Got {type(regressor)}.")
            ) 
    with pytest.raises(ValueError, match = err_msg):
        ForecasterSarimax(regressor = regressor)


def test_check_regressor_is_pmdarima_ARIMA_when_initialization():
    """
    Check `params` are stored in the forecaster.
    """
    forecaster = ForecasterSarimax(regressor = ARIMA(order=(1,1,1)))
    expected_params = ARIMA(order=(1,1,1)).get_params(deep=True)

    assert forecaster.params == expected_params


@pytest.mark.parametrize("order   , window_size", 
                         [((1,4,1), 5), 
                          ((1,4,3), 7), 
                          ((1,1,1), 4)], 
                         ids = lambda values : f'order: {values}'
                        )
def test_check_input_predict_exception_when_len_exog_is_less_than_steps(order, window_size):
    """
    Check `window_size` is correctly selected.
    """
    forecaster = ForecasterSarimax(regressor = ARIMA(order=order, seasonal_order=(2,2,1,0)))

    assert forecaster.window_size == window_size

