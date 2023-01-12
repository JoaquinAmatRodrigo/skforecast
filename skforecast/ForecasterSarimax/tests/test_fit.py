# Unit test fit ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima.arima import ARIMA


def test_fit_exception_when_len_exog_is_not_the_same_as_len_y():
    """
    Raise exception if the length of `exog` is different from the length of `y`.
    """
    y = pd.Series(data=np.arange(10))
    exog = pd.Series(data=np.arange(11))
    forecaster = ForecasterSarimax(regressor = ARIMA(order=(1,1,1)))

    err_msg = re.escape(
                    (f'`exog` must have same number of samples as `y`. '
                     f'length `exog`: ({len(exog)}), length `y`: ({len(y)})')
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
    forecaster = ForecasterSarimax(regressor = ARIMA(order=(1,1,1)))
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterSarimax(regressor = ARIMA(order=(1,1,1)))
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq

    assert results == expected


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterSarimax(regressor = ARIMA(order=(1,1,1)))
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.arange(50))

    pd.testing.assert_series_equal(forecaster.last_window, expected)