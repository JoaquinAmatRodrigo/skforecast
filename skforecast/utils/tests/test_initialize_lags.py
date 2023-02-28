# Unit test initialize_lags
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import initialize_lags


@pytest.mark.parametrize("lags             , expected", 
                         [(10              , np.arange(10) + 1), 
                          ([1, 2, 3]       , np.array([1, 2, 3])), 
                          (range(1, 4)     , np.array(range(1, 4))), 
                          (np.arange(1, 10), np.arange(1, 10))], 
                         ids = lambda values : f'values: {values}'
                        )
def test_initialize_lags_input_lags_parameter(lags, expected):
    """
    Test creation of attribute lags with different arguments.
    """
    lags = initialize_lags(
               forecaster_name = 'ForecasterAutoreg',
               lags            = lags
           )

    assert (lags == expected).all()


def test_initialize_lags_exception_when_lags_is_int_lower_than_1():
    """
    Test exception is raised when lags is initialized with int lower than 1.
    """
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = -10
        )


def test_initialize_lags_exception_when_lags_list_or_numpy_array_with_values_not_int():
    """
    Test exception is raised when lags is list or numpy array and element(s) are not int.
    """
    lags_list = [1, 1.5, [1, 2], range(5)]
    lags_np_array = np.array([1.2, 1.5])
    err_msg = re.escape('All values in `lags` must be int.')

    for lags in [lags_list, lags_np_array]:
        with pytest.raises(TypeError, match = err_msg):
            initialize_lags(
                forecaster_name = 'ForecasterAutoreg',
                lags            = lags
            )


def test_initialize_lags_exception_when_lags_has_values_lower_than_1():
    """
    Test exception is raised when lags is initialized with any value lower than 1.
    """
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    for lags in [[0, 1], range(0, 2), np.arange(0, 2)]:
        with pytest.raises(ValueError, match = err_msg):
            initialize_lags(
                forecaster_name = 'ForecasterAutoreg',
                lags            = lags
            )


def test_initialize_lags_exception_when_lags_is_not_valid_type():
    """
    Test exception is raised when lags is not a valid type.
    """
    lags = 'not_valid_type'
    err_msg = re.escape(
                f"`lags` argument must be an int, 1d numpy ndarray, range or list. "
                f"Got {type(lags)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = lags
        )


def test_initialize_lags_exception_when_lags_is_not_valid_type_ForecasterAutoregMultiVariate():
    """
    Test exception is raised when lags is not a valid type in ForecasterAutoregMultiVariate.
    """
    lags = 'not_valid_type'
    err_msg = re.escape(
                f"`lags` argument must be a dict, int, 1d numpy ndarray, range or list. "
                f"Got {type(lags)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoregMultiVariate',
            lags            = lags
        )