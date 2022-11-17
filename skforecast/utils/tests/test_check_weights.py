# Unit test initialize_weights
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
import inspect
from skforecast.utils.utils import initialize_weights
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor



@pytest.mark.parametrize("forecaster_type", 
                         ['ForecasterAutoreg', 'ForecasterAutoregCustom',
                          'ForecasterAutoregDirect', 'ForecasterAutoregMultiVariate'], 
                         ids=lambda ft: f'forecaster_type: {ft}')
def test_init_exception_when_weight_func_is_not_a_callable(forecaster_type):
    """
    Test exception is raised when weight_func is not a callable.
    """
    weight_func = 'not_callable'
    err_msg = re.escape(f"Argument `weight_func` must be a callable. Got {type(weight_func)}.")
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_type = forecaster_type, 
            regressor       = LinearRegression(), 
            weight_func     = weight_func, 
            series_weights  = None
        )


def test_init_exception_when_weight_func_is_not_a_callable_or_dict():
    """
    Test exception is raised when weight_func is not a callable or a dict.
    """
    weight_func = 'not_callable_or_dict'
    err_msg = re.escape(f"Argument `weight_func` must be a callable or a dict of callables. Got {type(weight_func)}.")
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_type = 'ForecasterAutoregMultiSeries', 
            regressor       = LinearRegression(), 
            weight_func     = weight_func, 
            series_weights  = None
        )


def test_init_exception_when_series_weights_is_not_a_dict():
    """
    Test exception is raised when series_weights is not a dict.
    """
    series_weights = 'not_callable_or_dict'
    err_msg = re.escape(
        f"Argument `series_weights` must be a dict of floats or ints."
        f"Got {type(series_weights)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_type = 'ForecasterAutoregMultiSeries', 
            regressor       = LinearRegression(), 
            weight_func     = None, 
            series_weights  = series_weights
        )


def test_init_when_weight_func_is_provided_and_regressor_has_not_sample_weights():
    """
    Test warning is created when weight_func is provided but the regressor has no argument
    sample_weights in his fit method.
    """

    def weight_func():
        pass

    warn_msg = re.escape(
                (f'Argument `weight_func` is ignored since regressor KNeighborsRegressor() '
                 f'does not accept `sample_weight` in its `fit` method.')
            )
    with pytest.warns(UserWarning, match = warn_msg):
        weight_func, source_code_weight_func, _ = initialize_weights(
            forecaster_type = 'ForecasterAutoreg', 
            regressor       = KNeighborsRegressor(), 
            weight_func     = weight_func, 
            series_weights  = None
        )
    
    assert weight_func is None
    assert source_code_weight_func is None


def test_init_when_series_weights_is_provided_and_regressor_has_not_sample_weights():
    """
    Test warning is created when series_weights is provided but the regressor has no argument
    sample_weights in his fit method.
    """
    series_weights = {'series_1': 1., 'series_2': 2.}

    warn_msg = re.escape(
                (f'Argument `series_weights` is ignored since regressor KNeighborsRegressor() '
                 f'does not accept `sample_weight` in its `fit` method.')
            )
    with pytest.warns(UserWarning, match = warn_msg):
        weight_func, source_code_weight_func, series_weights = initialize_weights(
            forecaster_type = 'ForecasterAutoregMultiSeries', 
            regressor       = KNeighborsRegressor(), 
            weight_func     = None, 
            series_weights  = series_weights
        )
    
    assert series_weights is None


def test_output_initialize_weights_source_code_weight_func_when_weight_func_not_dict():
    """
    Test source_code_weight_func output of initialize_weights when 
    weight_func is a callable.
    """
    def test_weight_func():
        """
        test
        """
        pass

    weight_func, source_code_weight_func, series_weights = initialize_weights(
        forecaster_type = 'ForecasterAutoregMultiSeries', 
        regressor       = LinearRegression(), 
        weight_func     = test_weight_func, 
        series_weights  = None
    )
    
    assert source_code_weight_func == inspect.getsource(test_weight_func)


def test_output_initialize_weights_source_code_weight_func_when_weight_func_dict():
    """
    Test source_code_weight_func output of initialize_weights when 
    weight_func is a dict.
    """
    def test_weight_func():
        """
        test
        """
        pass

    def test_weight_func_2():
        """
        test2
        """
        pass

    weight_func = {'series_1': test_weight_func, 'series_2': test_weight_func_2}    

    weight_func, source_code_weight_func, series_weights = initialize_weights(
        forecaster_type = 'ForecasterAutoregMultiSeries', 
        regressor       = LinearRegression(), 
        weight_func     = weight_func, 
        series_weights  = None
    )
    
    assert source_code_weight_func['series_1'] == inspect.getsource(test_weight_func)
    assert source_code_weight_func['series_2'] == inspect.getsource(test_weight_func_2)