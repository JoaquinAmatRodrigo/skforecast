# Unit test expand_index
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.utils import expand_index


def test_ValueError_expand_index_when_date_is_before_last_index():
    """
    Test ValueError is raised when the provided date is earlier than or equal 
    to the last date in index.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'], 
                             dtype='datetime64[ns]', freq='D')
    
    err_msg = re.escape(
        "The provided date cannot be earlier than or equal to the last "
        "observation date."
    )
    with pytest.raises(ValueError, match=err_msg):
        expand_index(index, steps='1990-01-02')


def test_ValueError_expand_index_when_index_is_not_DatetimeIndex():
    """
    Test ValueError is raised when `steps` is a date but the index is not a DatetimeIndex.
    """
    index = pd.RangeIndex(start=0, stop=3, step=1)
    
    err_msg = re.escape(
        "`index` must be a pandas DatetimeIndex when `steps` is not an integer."
    )
    with pytest.raises(ValueError, match=err_msg):
        expand_index(index, steps='1990-01-10')


def test_TypeError_expand_index_when_steps_is_not_int_str_or_Timestamp():
    """
    Test TypeError is raised when `steps` is not a int, str or pd.Timestamp.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    
    err_msg = re.escape(
        "`steps` must be an integer, string or pandas Timestamp."
    )
    with pytest.raises(TypeError, match=err_msg):
        expand_index(index, steps=2.5)


def test_output_expand_index_when_steps_is_string_date():
    """
    Test values returned by expand_index when `steps` is a string date.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07'],
                             dtype='datetime64[ns]', freq='D')
    results, _ = expand_index(index, steps='1990-01-07')
    
    assert (results == expected).all()


def test_output_expand_index_when_steps_is_Timestamp():
    """
    Test values returned by expand_index when `steps` is a pd.Timestamp.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07'],
                             dtype='datetime64[ns]', freq='D')
    results, _ = expand_index(index, steps=pd.Timestamp('1990-01-07'))
    
    assert (results == expected).all()


def test_output_expand_index_when_steps_is_string_date_with_kwargs():
    """
    Test values returned by expand_index when `steps` is a string date and `kwargs` are passed.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07'],
                             dtype='datetime64[ns]', freq='D')
    results, _ = expand_index(index, steps='1990-07-01', kwargs_pd_to_datetime={'format': '%Y-%d-%m'})
    
    assert (results == expected).all()


def test_TypeError_expand_index_when_index_is_no_pandas_DatetimeIndex_or_RangeIndex():
    """
    Test TypeError is raised when input is not a pandas DatetimeIndex or RangeIndex.
    """
    index = pd.Index([0, 1, 2])

    err_msg = "Argument `index` must be a pandas DatetimeIndex or RangeIndex."
    with pytest.raises(TypeError, match = err_msg):
        expand_index(index, steps=3)


def test_output_expand_index_when_index_is_DatetimeIndex():
    """
    Test values returned by expand_index when input is DatetimeIndex.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06'],
                             dtype='datetime64[ns]', freq='D')
    results, _ = expand_index(index, steps=3)
    
    assert (results == expected).all()


def test_output_expand_index_when_index_is_RangeIndex():
    """
    Test values returned by expand_index when input is RangeIndex.
    """
    index = pd.RangeIndex(start=0, stop=3, step=1)
    expected = pd.RangeIndex(start=3, stop=6, step=1)
    results, _  = expand_index(index, steps=3)
    
    assert (results == expected).all()


def test_output_expand_index_when_index_is_not_pandas_index():
    """
    Test values returned by expand_index when input is not a pandas index.
    """
    index = ['0', '1', '2']
    expected = pd.RangeIndex(start=0, stop=3, step=1)
    results, _ = expand_index(index, steps=3)
    
    assert (results == expected).all()
