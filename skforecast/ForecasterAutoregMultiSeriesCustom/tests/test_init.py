# Unit test __init__ ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags


def test_init_exception_when_fun_predictors_is_not_a_callable():
    """
    Test exception is raised when fun_predictors is not a Callable.
    """
    fun_predictors = 'not_valid_type'
    err_msg = re.escape(f"Argument `fun_predictors` must be a Callable. Got {type(fun_predictors)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeriesCustom(
            regressor       = LinearRegression(),
            fun_predictors  = 'not_valid_type',
            window_size     = 5
        )


def test_init_exception_when_window_size_is_not_int():
    """
    Test exception is raised when window_size is not an int.
    """
    window_size = 'not_valid_type'
    err_msg = re.escape(
                f'Argument `window_size` must be an int. Got {type(window_size)}.'
            )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeriesCustom(
            regressor       = LinearRegression(),
            fun_predictors  = create_predictors,
            window_size     = window_size
        )