# Unit test set_out_sample_residuals ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_set_out_sample_residuals_exception_when_residuals_is_not_pd_Series():
    """
    Test exception is raised when residuals argument is not pd.Series.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    residuals=[1, 2, 3]

    err_msg = re.escape(
                f"`residuals` argument must be `pd.Series`. Got {type(residuals)}."
              )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, level='1')


def test_set_out_sample_residuals_exception_when_level_not_in_series_levels():
    """
    Test exception is raised when level is not in forecaster.series_levels.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']

    err_msg = re.escape(
                f'`level` must be one of the `series_levels` : {forecaster.series_levels}.'
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(10)), 
                                            level     = '4')


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(2000)),
                                        level     = '1')

    assert len(forecaster.out_sample_residuals) == 1000


def test_same_out_sample_residuals_stored_when_residuals_length_is_greater_than_1000():
    """
    Test same `out_sample_residuals` are stored when its length is greater than 1000
    executing function two times.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(2000)),
                                        level     = '1')
    out_sample_residuals_1 = forecaster.out_sample_residuals

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(2000)),
                                        level     = '1')
    out_sample_residuals_2 = forecaster.out_sample_residuals

    assert (out_sample_residuals_1 == out_sample_residuals_2).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(20)),
                                        level     = '1')
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(10)),
                                        level     = '1',
                                        append    = False)
    expected = pd.Series(np.arange(10))
    results = forecaster.out_sample_residuals

    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(10)),
                                        level     = '1')
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(10)),
                                        level     = '1',
                                        append    = True)
    expected = pd.Series(np.hstack([np.arange(10), np.arange(10)]))
    results = forecaster.out_sample_residuals

    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append():
    """
    Test residuals stored when new residuals length is more than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2', '3']
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(10)),
                                        level     = '1')
    forecaster.set_out_sample_residuals(residuals = pd.Series(np.arange(1000)),
                                        level     = '1',
                                        append    = True)
    expected = pd.Series(np.hstack([np.arange(10), np.arange(1200)])[:1000])
    results = forecaster.out_sample_residuals

    assert (results == expected).all()