# Unit test __init__ ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def test_init_exception_when_steps_is_not_int():
    """
    Test exception is raised when steps is not an int.
    """
    steps = 'not_valid_type'
    err_msg = re.escape(
                f"`steps` argument must be an int greater than or equal to 1. "
                f"Got {type(steps)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregDirect(LinearRegression(), lags=2, steps=steps)


def test_init_exception_when_steps_is_less_than_1():
    """
    Test exception is raised when steps is less than 1.
    """
    steps = 0
    err_msg = re.escape(f"`steps` argument must be greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        ForecasterAutoregDirect(LinearRegression(), lags=2, steps=steps)


def test_init_exception_when_weight_func_is_not_a_callable():
    """
    Test exception is raised when weight_func is not a callable.
    """
    weight_func = 'not_callable'
    err_msg = re.escape(f"Argument `weight_func` must be a callable. Got {type(weight_func)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2, weight_func=weight_func)


def test_init_when_weight_func_is_provided_and_regressor_has_not_sample_weights():
    """
    Test warning is created when weight_func is provided but the regressor has no argument
    sample_weights in his fit method.
    """

    def weight_func():
        pass

    warn_msg = re.escape(
                    f"""
                    Argument `weight_func` is ignored since regressor KNeighborsRegressor()
                    does not accept `sample_weight` in its `fit` method.
                    """
                )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster = ForecasterAutoregDirect(
                        regressor      = KNeighborsRegressor(),
                        lags           = 3,
                        steps          = 2,
                        weight_func    = weight_func
                     )
    
    assert forecaster.weight_func is None
    assert forecaster.source_code_weight_func is None