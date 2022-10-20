# Unit test set_out_sample_residuals ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError


def test_set_out_sample_residuals_exception_when_residuals_is_not_pd_DataFrame():
    """
    Test exception is raised when residuals argument is not pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    residuals = [1, 2, 3]

    err_msg = re.escape(
                f"`residuals` argument must be a pandas DataFrame. Got {type(residuals)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_exception_when_forecaster_not_fitted():
    """
    Test exception is raised when forecaster is not fitted.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    residuals = pd.DataFrame({'1': [1, 2, 3, 4, 5], 
                              '2': [1, 2, 3, 4, 5]})

    err_msg = re.escape(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_out_sample_residuals()`."
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_exception_when_append_and_column_not_in_keys():
    """
    Test exception is raised when `append = True` but a column from residuals is in
    self.series_levels but not in out_sample_residuals keys.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    level = '1'
    forecaster.series_levels = ['1']
    forecaster.out_sample_residuals = {'2': np.arange(5)}
    forecaster.fitted = True
    residuals = pd.DataFrame({'1': [1, 2, 3, 4, 5], 
                              '2': [1, 2, 3, 4, 5]})

    err_msg = re.escape(f'{level} does not exists in `forecaster.out_sample_residuals` keys: {list(forecaster.out_sample_residuals.keys())}')
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1']
    forecaster.fitted = True
    forecaster.set_out_sample_residuals(residuals = pd.DataFrame({'1': np.arange(2000), 
                                                                  '2': np.arange(2000)}))

    assert len(forecaster.out_sample_residuals['1']) == 1000


def test_same_out_sample_residuals_stored_when_residuals_length_is_greater_than_1000():
    """
    Test same `out_sample_residuals` are stored when its length is greater than 1000
    executing function two times.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1']
    forecaster.fitted = True
    forecaster.set_out_sample_residuals(residuals = pd.DataFrame({'1': np.arange(2000), 
                                                                  '2': np.arange(2000)}))
    out_sample_residuals_1 = forecaster.out_sample_residuals['1']

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1']
    forecaster.fitted = True
    forecaster.set_out_sample_residuals(residuals = pd.DataFrame({'1': np.arange(2000)}))
    out_sample_residuals_2 = forecaster.out_sample_residuals['1']

    assert (out_sample_residuals_1 == out_sample_residuals_2).all()


@pytest.mark.parametrize("level, expected", 
                         [('1' , np.arange(10)),
                          ('2' , np.arange(10))])
def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append(level, expected):
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2']
    forecaster.fitted = True
    residuals = pd.DataFrame({'1': np.arange(10), 
                              '2': np.arange(10)})

    forecaster.set_out_sample_residuals(residuals = pd.DataFrame({'1': np.arange(20), 
                                                                  '2': np.arange(20)}))
    forecaster.set_out_sample_residuals(residuals = residuals,
                                        append    = False)

    assert (forecaster.out_sample_residuals[level] == expected).all()


@pytest.mark.parametrize("level, expected", 
                         [('1' , np.hstack([np.arange(10), np.arange(10)])),
                          ('2' , np.hstack([np.arange(10), np.arange(10)]))])
def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append(level, expected):
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2']
    forecaster.fitted = True
    residuals = pd.DataFrame({'1': np.arange(10), 
                              '2': np.arange(10)})

    forecaster.set_out_sample_residuals(residuals = residuals)
    forecaster.set_out_sample_residuals(residuals = residuals,
                                        append    = True)

    assert (forecaster.out_sample_residuals[level] == expected).all()


@pytest.mark.parametrize("level, expected", 
                         [('1' , np.hstack([np.arange(10), np.arange(1200)])[:1000]),
                          ('2' , np.hstack([np.arange(10), np.arange(1200)])[:1000])])
def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append(level, expected):
    """
    Test residuals stored when new residuals length is more than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.series_levels = ['1', '2']
    forecaster.fitted = True
    residuals = pd.DataFrame({'1': np.arange(10), 
                              '2': np.arange(10)})
    residuals_2 = pd.DataFrame({'1': np.arange(1000), 
                                '2': np.arange(1000)})

    forecaster.set_out_sample_residuals(residuals = residuals)
    forecaster.set_out_sample_residuals(residuals = residuals_2,
                                        append    = True)

    assert (forecaster.out_sample_residuals[level] == expected).all()