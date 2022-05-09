# Unit test fit ForecasterAutoregDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def test_forecaster_index_freq_stored():
    '''
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    '''
    serie_with_DatetimeIndex = pd.Series(
                data  = np.arange(10),
                index = pd.date_range(start='2022-01-01', periods=10)
                )
    forecaster = ForecasterAutoregDirect(XGBRegressor(random_state=123), lags=3, steps=2)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq
    assert results == expected


def test_fit_last_window_stored():
    '''
    Test that values of last window are stored after fitting.
    '''    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([46, 47, 48, 49]), index=[46, 47, 48, 49])
    results = forecaster.last_window
    pd.testing.assert_series_equal(expected, results)
