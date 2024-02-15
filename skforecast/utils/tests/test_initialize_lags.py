# Unit test initialize_lags
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import initialize_lags


def test_ValueError_initialize_lags_when_lags_is_int_lower_than_1():
    """
    Test ValueError is raised when lags is initialized with int lower than 1.
    """
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = -10
        )


@pytest.mark.parametrize("lags", 
                         [[1, 1.5, [1, 2], range(5)], 
                          (1, 1.5, [1, 2], range(5)), 
                          np.array([1.2, 1.5])], 
                         ids = lambda lags : f'lags: {lags}')
def test_TypeError_initialize_lags_when_lags_list_tuple_or_numpy_array_with_values_not_int(lags):
    """
    Test TypeError is raised when lags is list or numpy array and element(s) are not int.
    """
    err_msg = re.escape('All values in `lags` must be integers.')
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [[0, 1], 
                          (0, 1), 
                          range(0, 2), 
                          np.arange(0, 2)], 
                         ids = lambda lags : f'lags: {lags}')
def test_ValueError_initialize_lags_when_lags_has_values_lower_than_1(lags):
    """
    Test ValueError is raised when lags is initialized with any value lower than 1.
    """
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = lags
        )


def test_TypeError_initialize_lags_when_lags_is_not_valid_type():
    """
    Test TypeError is raised when lags is not a valid type.
    """
    lags = 'not_valid_type'
    err_msg = re.escape(
                (f"`lags` argument must be an int, 1d numpy ndarray, range, tuple or list. "
                 f"Got {type(lags)}.")
            )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoreg',
            lags            = lags
        )


def test_TypeError_initialize_lags_when_lags_is_not_valid_type_ForecasterAutoregMultiVariate():
    """
    Test TypeError is raised when lags is not a valid type in ForecasterAutoregMultiVariate.
    """
    lags = 'not_valid_type'
    err_msg = re.escape(
        f"`lags` argument must be a dict, int, 1d numpy ndarray, range, tuple or list. "
        f"Got {type(lags)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterAutoregMultiVariate',
            lags            = lags
        )


@pytest.mark.parametrize("lags             , expected", 
                         [(10              , np.arange(10) + 1), 
                          ([1, 2, 3]       , np.array([1, 2, 3])),
                          ((1, 2, 3)       , np.array((1, 2, 3))),  
                          (range(1, 4)     , np.array(range(1, 4))), 
                          (np.arange(1, 10), np.arange(1, 10))], 
                         ids = lambda values : f'values: {values}' )
def test_initialize_lags_input_lags_parameter(lags, expected):
    """
    Test creation of attribute lags with different arguments.
    """
    lags = initialize_lags(
               forecaster_name = 'ForecasterAutoreg',
               lags            = lags
           )

    assert (lags == expected).all()