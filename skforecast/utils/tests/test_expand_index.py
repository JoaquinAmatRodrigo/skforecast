# Unit test expand_index
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import expand_index


def test_output_expand_index_when_index_is_DatetimeIndex():
    """
    Test values returned by expand_index when input is DatetimeIndex.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06'],
                             dtype='datetime64[ns]', freq='D')
    results = expand_index(index, steps=3)
    
    assert (results == expected).all()


def test_output_expand_index_when_index_is_RangeIndex():
    """
    Test values returned by expand_index when input is RangeIndex.
    """
    index = pd.RangeIndex(start=0, stop=3, step=1)
    expected = pd.RangeIndex(start=3, stop=6, step=1)
    results = expand_index(index, steps=3)
    
    assert (results == expected).all()


def test_output_expand_index_when_index_is_not_pandas_index():
    """
    Test values returned by expand_index when input is not a pandas index.
    """
    index = ['0', '1', '2']
    expected = pd.RangeIndex(start=0, stop=3, step=1)
    results = expand_index(index, steps=3)
    
    assert (results == expected).all()