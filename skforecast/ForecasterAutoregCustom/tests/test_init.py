# Unit test __init__ ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import pandas as pd
import numpy as np
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags  


@pytest.mark.parametrize("window_size", 
                         [0, 0.5, 1.5, 'not_int'], 
                         ids = lambda ws : f'window_size: {ws}')
def test_init_ValueError_when_window_size_argument_is_not_int_or_greater_than_0(window_size):
    """
    Test ValueError is raised when window_size is not an int or greater than 0.
    """
    err_msg = re.escape(
                (f"Argument `window_size` must be an integer equal to or "
                 f"greater than 1. Got {window_size}.")
             )
    with pytest.raises(ValueError, match = err_msg):
         ForecasterAutoregCustom(
             regressor      = LinearRegression(),
             fun_predictors = create_predictors,
             window_size    = window_size
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
         ForecasterAutoregCustom(
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
    window_size = 5

    forecaster = ForecasterAutoregCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = window_size,
                     differentiation = dif
                 )
    
    assert forecaster.window_size == window_size + dif


def test_init_TypeError_when_fun_predictors_argument_is_not_Callable():
    """
    Test TypeError is raised when fun_predictors is not a Callable.
    """
    fun_predictors = 'create_predictors'
    err_msg = re.escape(
                 f"Argument `fun_predictors` must be a Callable. Got {type(fun_predictors)}."
             )
    with pytest.raises(TypeError, match = err_msg):
         ForecasterAutoregCustom(
             regressor      = LinearRegression(),
             fun_predictors = fun_predictors,
             window_size    = 5
         )

def test_init_TypeError_when_weight_func_argument_is_not_Callable():
    """
    Test TypeError is raised when weight_func is not a callable.
    """
    weight_func = '---'
    err_msg = re.escape(
                 f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
             )
    with pytest.raises(TypeError, match = err_msg):
         ForecasterAutoregCustom(
             regressor      = LinearRegression(),
             fun_predictors = create_predictors,
             window_size    = 5,
             weight_func    = weight_func
         )


def test_init_IgnoredArgumentWarning_when_series_weights_is_provided_and_regressor_has_not_sample_weights():
    """
    Test IgnoredArgumentWarning is raised when weight_func is provided but the 
    regressor has no argument sample_weights in his fit method.
    """
    def weight_func(index): # pragma: no cover
         return np.ones_like(index)
 
    warn_msg = re.escape(
                 ("Argument `weight_func` is ignored since regressor KNeighborsRegressor() "
                  "does not accept `sample_weight` in its `fit` method.")
             )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
         ForecasterAutoregCustom(
             regressor      = KNeighborsRegressor(),
             fun_predictors = create_predictors,
             window_size    = 5,
             weight_func    = weight_func
         )