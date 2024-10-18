# Unit test exog_to_direct_numpy
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import exog_to_direct_numpy


def test_exog_to_direct_TypeError_when_exog_not_numpy_ndarray():
    """
    Test TypeError is raised when exog is not a numpy ndarray.
    """
    exog = pd.Series(np.arange(7))

    err_msg = re.escape(f"`exog` must be a numpy ndarray. Got {type(exog)}.")
    with pytest.raises(TypeError, match = err_msg):
        exog_to_direct_numpy(exog=exog, steps=2)


def test_exog_to_direct_numpy_when_steps_1_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps 1 and exog is a  
    1d numpy array.
    """
    exog = np.arange(10)
    results = exog_to_direct_numpy(exog=exog, steps=1)
    expected = np.array([[0],
                         [1],
                         [2],
                         [3],
                         [4],
                         [5],
                         [6],
                         [7],
                         [8],
                         [9]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_2_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a  
    1d numpy array.
    """
    exog = np.arange(10)
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = np.array([[0, 1],
                         [1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 5],
                         [5, 6],
                         [6, 7],
                         [7, 8],
                         [8, 9]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_len_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps len_exog (5) and exog 
    is a 1d numpy array.
    """
    exog = np.arange(5)
    results = exog_to_direct_numpy(exog=exog, steps=len(exog))
    expected = np.array([[0, 1, 2, 3, 4]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_1_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 1 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=1)
    expected = np.array([[100, 1000],
                         [101, 1001],
                         [102, 1002],
                         [103, 1003],
                         [104, 1004],
                         [105, 1005],
                         [106, 1006],
                         [107, 1007],
                         [108, 1008],
                         [109, 1009]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_2_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = np.array([[100, 1000, 101, 1001],
                         [101, 1001, 102, 1002],
                         [102, 1002, 103, 1003],
                         [103, 1003, 104, 1004],
                         [104, 1004, 105, 1005],
                         [105, 1005, 106, 1006],
                         [106, 1006, 107, 1007],
                         [107, 1007, 108, 1008],
                         [108, 1008, 109, 1009]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_3_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 3 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=3)
    expected = np.array([[100, 1000, 101, 1001, 102, 1002],
                         [101, 1001, 102, 1002, 103, 1003],
                         [102, 1002, 103, 1003, 104, 1004],
                         [103, 1003, 104, 1004, 105, 1005],
                         [104, 1004, 105, 1005, 106, 1006],
                         [105, 1005, 106, 1006, 107, 1007],
                         [106, 1006, 107, 1007, 108, 1008],
                         [107, 1007, 108, 1008, 109, 1009]])

    np.testing.assert_array_equal(results, expected)


def test_exog_to_direct_numpy_when_steps_len_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps len exog (5) and 
    exog is a 2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 105), np.arange(1000, 1005)])
    results = exog_to_direct_numpy(exog=exog, steps=len(exog))
    expected = np.array([[100, 1000, 101, 1001, 102, 1002, 
                          103, 1003, 104, 1004]])

    np.testing.assert_array_equal(results, expected)