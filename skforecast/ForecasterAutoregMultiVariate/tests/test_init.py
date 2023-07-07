# Unit test __init__ ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression


def test_init_TypeError_when_level_is_not_a_str():
    """
    Test TypeError is raised when level is not a str.
    """
    level = 5
    err_msg = re.escape(f"`level` argument must be a str. Got {type(level)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiVariate(LinearRegression(), level=level, lags=2, steps=3)


def test_init_TypeError_when_steps_is_not_int():
    """
    Test TypeError is raised when steps is not an int.
    """
    steps = 'not_valid_type'
    err_msg = re.escape(
                f"`steps` argument must be an int greater than or equal to 1. "
                f"Got {type(steps)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=2, steps=steps)


def test_init_ValueError_when_steps_is_less_than_1():
    """
    Test ValueError is raised when steps is less than 1.
    """
    steps = 0
    err_msg = re.escape(f"`steps` argument must be greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=2, steps=steps)


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value : f'n_jobs: {value}')
def test_init_TypeError_when_n_jobs_not_int_or_auto(n_jobs):
    """
    Test TypeError is raised in when n_jobs is not an integer or 'auto'.
    """
    err_msg = re.escape(f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=2, 
                                      steps=2, n_jobs=n_jobs)