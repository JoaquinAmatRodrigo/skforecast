# Unit test fit ForecasterEquivalentDate
# ==============================================================================
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import pytest
from skforecast.ForecasterBaseline import ForecasterEquivalentDate


def test_fit_offset_DateOffset_y_index_not_DatetimeIndex():
    """
    Test Exception is raised when offset is a DateOffset and y index is not a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
        offset=DateOffset(days=1), n_offsets=2, agg_func=np.mean, forecaster_id=None
    )
    y = pd.Series(np.random.rand(10))
    with pytest.raises(
        Exception,
        match="If `offset` is a pandas DateOffset, the index of `y` must be a pandas DatetimeIndex with a frequency.",
    ):
        forecaster.fit(y)


def test_fit_reset_values():
    """
    Test values are reset correctly when the forecaster is fitted again.
    """
    forecaster = ForecasterEquivalentDate(
        offset = 5,
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10))
    forecaster.fit(y)
    assert forecaster.index_type is not None
    assert forecaster.index_freq is not None
    assert forecaster.last_window is not None
    assert forecaster.fitted is True
    assert forecaster.training_range is not None
    forecaster.fit(y)
    assert forecaster.index_type is not None
    assert forecaster.index_freq is not None
    assert forecaster.last_window is not None
    assert forecaster.fitted is True
    assert forecaster.training_range is not None


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


def test_fit_offset_DateOffset():
    """
    Test window_size and last_window are set correctly when offset is a DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
        offset = DateOffset(days=1),
        n_offsets = 2,
        agg_func = np.mean,
        forecaster_id = None
    )
    y = pd.Series(np.random.rand(10), index=pd.date_range(start='1/1/2021', periods=10))
    forecaster.fit(y)
    last_window_start = (y.index[-1] + y.index.freq) - (forecaster.offset * forecaster.n_offsets)
    assert forecaster.window_size == len(y.loc[last_window_start:])
    assert forecaster.last_window.equals(y.iloc[-forecaster.window_size:])