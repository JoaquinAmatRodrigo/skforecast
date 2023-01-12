# Unit test _check_interval
# ==============================================================================
import pytest
import re
import numpy as np
import pandas as pd
from skforecast.utils.utils import check_interval


def test_check_interval_typeerror_when_interval_is_not_a_list():
    """
    Check `TypeError` is raised when `interval` is not a `list`.
    """
    err_msg = re.escape(
                ('`interval` must be a `list`. For example, interval of 95% '
                 'should be as `interval = [2.5, 97.5]`.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(interval = 'not_a_list')


def test_check_interval_valueerror_when_interval_len_is_not_2():
    """
    Check `ValueError` is raised when `interval` len is not 2.
    """
    err_msg = re.escape(
                ('`interval` must contain exactly 2 values, respectively the '
                 'lower and upper interval bounds. For example, interval of 95% '
                 'should be as `interval = [2.5, 97.5]`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 50.0, 97.5])


def test_check_interval_valueerror_when_lower_bound_less_than_0():
    """
    Check `ValueError` is raised when lower bound is less than 0.
    """
    err_msg = re.escape(
                ('Lower interval bound (-1.0) must be >= 0 and < 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [-1.0, 97.5])


def test_check_interval_valueerror_when_lower_bound_greater_than_or_equal_to_100():
    """
    Check `ValueError` is raised when lower bound is greater than or equal to 100.
    """
    err_msg = re.escape(
                ('Lower interval bound (100.0) must be >= 0 and < 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [100.0, 97.5])
    err_msg = re.escape(
                ('Lower interval bound (101.0) must be >= 0 and < 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [101.0, 97.5])


def test_check_interval_valueerror_when_upper_bound_less_than_or_equal_to_0():
    """
    Check `ValueError` is raised when upper bound is less than or equal to 0.
    """
    err_msg = re.escape(
                ('Upper interval bound (0.0) must be > 0 and <= 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 0.0])
    err_msg = re.escape(
                ('Upper interval bound (-1.0) must be > 0 and <= 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, -1.0])


def test_check_interval_valueerror_when_upper_bound_greater_than_100():
    """
    Check `ValueError` is raised when upper bound is greater than 100.
    """
    err_msg = re.escape(
                ('Upper interval bound (101.0) must be > 0 and <= 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 101.0])


def test_check_interval_valueerror_when_lower_bound_greater_than_or_equal_to_upper_bound():
    """
    Check `ValueError` is raised when lower bound is greater than or equal to
    upper bound.
    """
    err_msg = re.escape(
                ('Lower interval bound (2.5) must be less than the '
                 'upper interval bound (2.5).')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 2.5])
    err_msg = re.escape(
                ('Lower interval bound (2.5) must be less than the '
                 'upper interval bound (2.0).')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 2.0])