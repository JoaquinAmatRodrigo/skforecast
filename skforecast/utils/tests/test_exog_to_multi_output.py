# Unit test exog_to_multi_output
# ==============================================================================
import numpy as np
from pytest import approx
from skforecast.utils import exog_to_multi_output

def test_exog_to_multi_output_when_lags_3_steps_2_exog_numpy_1d():
    '''
    Test exog_to_multi_output results when using lags 3, steps 2 and exog is a  
    1d numpy array.
    '''
    exog = np.arange(10)
    results = exog_to_multi_output(exog=exog, steps=2)
    expected = np.array([[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [8, 9]])

    assert results == approx(expected)


def test_exog_to_multi_output_when_lags_3_steps_2_exog_numpy_array_2d():
    '''
    Test exog_to_multi_output results when using lags 3, steps 2 and exog is a  
    2d numpy array.
    '''
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_multi_output(exog=exog, steps=2)
    expected = np.array([[ 100,  101, 1000, 1001],
                        [ 101,  102, 1001, 1002],
                        [ 102,  103, 1002, 1003],
                        [ 103,  104, 1003, 1004],
                        [ 104,  105, 1004, 1005],
                        [ 105,  106, 1005, 1006],
                        [ 106,  107, 1006, 1007],
                        [ 107,  108, 1007, 1008],
                        [ 108,  109, 1008, 1009]])

    assert results == approx(expected)


def test_exog_to_multi_output_when_lags_2_steps_3_exog_numpy_array_2d():
    '''
    Test exog_to_multi_output results when using lags 2, steps 3 and exog is a  
    2d numpy array.
    '''
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_multi_output(exog=exog, steps=3)
    expected = np.array([[ 100,  101,  102, 1000, 1001, 1002],
                        [ 101,  102,  103, 1001, 1002, 1003],
                        [ 102,  103,  104, 1002, 1003, 1004],
                        [ 103,  104,  105, 1003, 1004, 1005],
                        [ 104,  105,  106, 1004, 1005, 1006],
                        [ 105,  106,  107, 1005, 1006, 1007],
                        [ 106,  107,  108, 1006, 1007, 1008],
                        [ 107,  108,  109, 1007, 1008, 1009]])

    assert results == approx(expected)