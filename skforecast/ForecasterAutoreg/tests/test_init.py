# Unit test __init__ ForecasterAutoreg
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def test_init_exception_when_weight_func_is_not_a_callable():
    """
    Test exception is raised when weight_func is not a callable.
    """
    weight_func = 'not_callable'
    err_msg = re.escape(f"Argument `weight_func` must be a callable. Got {type(weight_func)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoreg(LinearRegression(), lags=3, weight_func=weight_func)


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
        forecaster = ForecasterAutoreg(KNeighborsRegressor(), lags=3, weight_func=weight_func)
    
    assert forecaster.weight_func is None
    assert forecaster.source_code_weight_func is None