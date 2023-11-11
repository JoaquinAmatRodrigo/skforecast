# Unit test fit ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from skforecast.ForecasterBaseline import ForecasterEquivalentDate


def test_fit_TypeError_offset_DateOffset_y_index_not_DatetimeIndex():
    """
    Test TypeError is raised when offset is a DateOffset and y index is 
    not a DatetimeIndex or has no freq.
    """
    forecaster = ForecasterEquivalentDate(
        offset=DateOffset(days=1), n_offsets=2, agg_func=np.mean
    )

    err_msg = re.escape(
        ("If `offset` is a pandas DateOffset, the index of `y` must be a "
         "pandas DatetimeIndex.")
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.fit(y = pd.Series(np.arange(10)))


def test_fit_y_index_DatetimeIndex():
    """
    Test index_freq is set correctly when y index is a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
        offset = 5,
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10), index=pd.date_range(start='1/1/2021', periods=10))
    forecaster.fit(y)

    assert forecaster.index_freq == y.index.freqstr


def test_fit_y_index_not_DatetimeIndex():
    """
    Test index_freq is set correctly when y index is not a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
        offset = 5,
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10))
    forecaster.fit(y)

    assert forecaster.index_freq == y.index.step


def test_fit_offset_int():
    """
    Test window_size and last_window are set correctly when offset is an int.
    """
    forecaster = ForecasterEquivalentDate(
        offset = 2,
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=0, stop=10))
    forecaster.fit(y)

    assert forecaster.window_size == 4.0
    assert forecaster.last_window.equals(y.iloc[-4:])


def test_fit_offset_DateOffset():
    """
    Test window_size and last_window are set correctly when offset is a DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
        offset = DateOffset(days=2),
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10), index=pd.date_range(start='1/1/2021', periods=10))
    forecaster.fit(y)

    assert forecaster.window_size == 4.0
    assert forecaster.last_window.equals(y.iloc[-4:])