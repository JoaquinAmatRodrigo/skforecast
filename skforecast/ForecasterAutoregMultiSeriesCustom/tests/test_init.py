# Unit test __init__ ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags


def test_init_TypeError_when_window_size_is_not_int():
    """
    Test TypeError is raised when window_size is not an int.
    """
    window_size = 'not_valid_type'
    err_msg = re.escape(
        f"Argument `window_size` must be an int. Got {type(window_size)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeriesCustom(
            regressor       = LinearRegression(),
            fun_predictors  = create_predictors,
            window_size     = window_size
        )


def test_init_TypeError_when_fun_predictors_is_not_a_Callable():
    """
    Test TypeError is raised when fun_predictors is not a Callable.
    """
    fun_predictors = 'not_valid_type'
    err_msg = re.escape(
        f"Argument `fun_predictors` must be a Callable. Got {type(fun_predictors)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeriesCustom(
            regressor       = LinearRegression(),
            fun_predictors  = 'not_valid_type',
            window_size     = 5
        )


def test_init_TypeError_when_weight_func_argument_is_not_Callable():
    """
    Test TypeError is raised when weight_func is not a Callable or a dict of Callables.
    """
    weight_func = '---'
    err_msg = re.escape(
        (f"Argument `weight_func` must be a Callable or a dict of "
         f"Callables. Got {type(weight_func)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
         ForecasterAutoregMultiSeriesCustom(
             regressor      = LinearRegression(),
             fun_predictors = create_predictors,
             window_size    = 5,
             weight_func    = weight_func
         )


def test_init_TypeError_when_series_weights_argument_is_not_dict():
    """
    Test TypeError is raised when series_weights is not a dict.
    """
    series_weights = '---'
    err_msg = re.escape(
        (f"Argument `series_weights` must be a dict of floats or ints."
         f"Got {type(series_weights)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
         ForecasterAutoregMultiSeriesCustom(
             regressor      = LinearRegression(),
             fun_predictors = create_predictors,
             window_size    = 5,
             series_weights = series_weights
         )


@pytest.mark.parametrize("dif", 
                         [0, 0.5, 1.5, 'not_int'], 
                         ids = lambda dif : f'differentiation: {dif}')
def test_init_ValueError_when_differentiation_argument_is_not_int_or_greater_than_0(dif):
    """
    Test ValueError is raised when differentiation is not an int or greater than 0.
    """
    err_msg = re.escape(
        (f"Argument `differentiation` must be an integer equal to or "
         f"greater than 1. Got {dif}.")
    )
    with pytest.raises(ValueError, match = err_msg):
         ForecasterAutoregMultiSeriesCustom(
             regressor       = LinearRegression(),
             fun_predictors  = create_predictors,
             window_size     = 5,
             differentiation = dif
         )


@pytest.mark.parametrize("dif", 
                         [1, 2], 
                         ids = lambda dif : f'differentiation: {dif}')
def test_init_window_size_is_increased_when_differentiation(dif):
    """
    Test window_size is increased when including differentiation.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 5,
                     differentiation = dif
                 )
    
    assert forecaster.window_size == len(forecaster.lags) + dif


def test_ForecasterAutoregMultiSeries_init_invalid_encoding():
    """
    Test ValueError is raised when encoding is not valid.
    """

    err_msg = re.escape(
        ("Argument `encoding` must be one of the following values: 'ordinal', "
         "'ordinal_category', 'onehot'. Got 'invalid_encoding'.")
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterAutoregMultiSeriesCustom(
            regressor      = LinearRegression(),
            fun_predictors = create_predictors,
            window_size    = 5,
            encoding       = 'invalid_encoding'
        )