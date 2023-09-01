# Unit test fit ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import y_datetime


def test_fit_ValueError_when_len_exog_is_not_the_same_as_len_y():
    """
    Raise ValueError if the length of `exog` is different from the length of `y`.
    """
    y = pd.Series(data=np.arange(10))
    exog = pd.Series(data=np.arange(11))
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
                  (f"`exog` must have same number of samples as `y`. "
                   f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=y, exog=exog)


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq

    assert results == expected


@pytest.mark.parametrize("suppress_warnings", 
                         [True, False], 
                         ids = lambda v : f'suppress_warnings: {v}')
def test_fit_last_window_stored_with_suppress_warnings(suppress_warnings):
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=pd.Series(np.arange(50)), 
                   suppress_warnings=suppress_warnings)
    expected = pd.Series(np.arange(50))

    pd.testing.assert_series_equal(forecaster.last_window, expected)


@pytest.mark.parametrize("regressor", 
                         [ARIMA(order=(1, 0, 0)), 
                          Sarimax(order=(1, 0, 0))], 
                         ids = lambda reg : f'regressor: {type(reg)}')
@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {type(values)}')
def test_fit_extended_index_stored(regressor, y, idx):
    """
    Test that values of self.regressor.arima_res_.fittedvalues.index are 
    stored after fitting in forecaster.extended_index.
    """
    forecaster = ForecasterSarimax(regressor=regressor)
    forecaster.fit(y=y)

    pd.testing.assert_index_equal(forecaster.extended_index, idx)