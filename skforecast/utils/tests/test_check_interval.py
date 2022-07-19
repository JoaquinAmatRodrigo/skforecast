# Unit test _check_interval
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils.utils import _check_interval


def test_check_interval_typeerror_when_interval_is_not_a_list():
    """
    Check `TypeError` is raised when `interval` is not a `list`.
    """
    with pytest.raises(TypeError):
        _check_interval(interval = 'not_a_list')


def test_check_interval_valueerror_when_interval_len_is_not_2():
    """
    Check `ValueError` is raised when `interval` len is not 2.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, 50.0, 97.5])


def test_check_interval_valueerror_when_lower_bound_less_than_0():
    """
    Check `ValueError` is raised when lower bound is less than 0.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [-1.0, 97.5])


def test_check_interval_valueerror_when_lower_bound_greater_than_or_equal_to_100():
    """
    Check `ValueError` is raised when lower bound is greater than or equal to 100.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [100.0, 97.5])
    with pytest.raises(ValueError):
        _check_interval(interval = [101.0, 97.5])


def test_check_interval_valueerror_when_upper_bound_less_than_or_equal_to_0():
    """
    Check `ValueError` is raised when upper bound is less than or equal to 0.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, 0.0])
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, -1.0])


def test_check_interval_valueerror_when_upper_bound_greater_than_100():
    """
    Check `ValueError` is raised when upper bound is greater than 100.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, 101.0])


def test_check_interval_valueerror_when_lower_bound_greater_than_or_equal_to_upper_bound():
    """
    Check `ValueError` is raised when lower bound is greater than or equal to
    upper bound.
    """
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, 2.5])
    with pytest.raises(ValueError):
        _check_interval(interval = [2.5, 2.0])