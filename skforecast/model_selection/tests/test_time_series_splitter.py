# Unit test time_series_splitter
# ==============================================================================
import pytest
import numpy as np
from skforecast.model_selection import time_series_splitter


def test_time_series_splitter_exception_when_y_is_numpy_array_with_more_than_1_dimesion():
    """
    """
    results = time_series_splitter(np.arange(10).reshape(-1, 2), initial_train_size=3, steps=1)

    with pytest.raises(Exception):
       list(results)
       

def test_time_series_splitter_exception_when_y_is_list():
    """
    """
    results = time_series_splitter([0,1,2,3,4], initial_train_size=3, steps=1)

    with pytest.raises(Exception):
       list(results)


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_1_allow_incomplete_fold_True():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=1,
                allow_incomplete_fold=True,
                verbose=True
            )
    results = list(results)
    expected = [(range(0, 5), range(5, 6)),
                (range(0, 6), range(6, 7)),
                (range(0, 7), range(7, 8)),
                (range(0, 8), range(8, 9)),
                (range(0, 9), range(9, 10))]

    assert results == expected


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_5_allow_incomplete_fold_True():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=5,
                allow_incomplete_fold=True,
                verbose=True
            )
    results = list(results)
    expected = [(range(0, 5), range(5, 10))]

    assert results == expected


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_3_allow_incomplete_fold_False():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=3,
                allow_incomplete_fold=False,
                verbose=True
            )
    results = list(results)
    expected = [(range(0, 5), range(5, 8))]

    assert results == expected


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_3_allow_incomplete_fold_True():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=3,
                allow_incomplete_fold=True,
                verbose=True
            )
    results = list(results)
    expected = [(range(0, 5), range(5, 8)), (range(0, 8), range(8, 10))]

    assert results == expected


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_20_allow_incomplete_fold_False():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=20,
                allow_incomplete_fold=False,
                verbose=True
            )

    results = list(results)
    expected = []

    assert results == expected


def test_time_series_splitter_when_y_is_numpy_arange_10_initial_train_size_5_steps_20_allow_incomplete_fold_True():
    """
    """
    results = time_series_splitter(
                y=np.arange(10),
                initial_train_size=5,
                steps=20,
                allow_incomplete_fold=True,
                verbose=True
            )
          
    results = list(results)
    expected = []
    results == expected
    
    assert results == expected