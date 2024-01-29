# Unit test check_preprocess_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_preprocess_series
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import series


def test_TypeError_check_preprocess_series_when_series_is_not_pandas_DataFrame_or_dict():
    """
    Test TypeError is raised when series is not a pandas DataFrame or dict.
    """
    series = np.array([1, 2, 3])

    err_msg = re.escape(
        (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
         f"Got {type(series)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series)


def test_check_preprocess_series_when_series_is_pandas_DataFrame():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame.
    """
    series_dict, series_index = check_preprocess_series(series=series)

    expected_index = pd.RangeIndex(start=0, stop=len(series), step=1)

    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k, v in series_dict.items():
        assert isinstance(v, pd.Series)
        assert k == v.name

    assert isinstance(series_index, pd.RangeIndex)
    pd.testing.assert_index_equal(series_index, expected_index)


def test_check_preprocess_series_when_series_is_pandas_DataFrame_with_DatetimeIndex():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(
        start='2000-01-01', periods=len(series_datetime), freq='D'
    )

    series_dict, series_index = check_preprocess_series(series=series_datetime)

    expected_index = pd.date_range(
        start='2000-01-01', periods=len(series_datetime), freq='D'
    )

    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k, v in series_dict.items():
        assert isinstance(v, pd.Series)
        assert k == v.name

    assert isinstance(series_index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(series_index, expected_index)


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when series is dict not containing 
    a pandas DataFrame or dict.
    """
    series_dict = {'l1': np.array([1, 2, 3]),
                   'l2': series}

    err_msg = re.escape(
        ("All series must be a named pandas Series or a pandas Dataframe. "
         "with a single column. Review series: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_series_is_dict_with_a_DataFrame_with_2_columns():
    """
    Test ValueError is raised when series is dict containing a pandas DataFrame 
    with 2 columns.
    """
    series_dict = {'l1': series['1'],
                   'l2': series}

    err_msg = re.escape(
        ("All series must be a named pandas Series or a pandas Dataframe "
         "with a single column. Review series: l2")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_DatetimeIndex():
    """
    Test TypeError is raised when series is dict containing series with no
    DatetimeIndex.
    """
    series_dict = {'l1': series['1'],
                   'l2': series['2']}

    err_msg = re.escape(
        ("All series must have a DatetimeIndex index with the same frequency. "
         "Review series: ['l1', 'l2']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_series_is_dict_with_different_freqs():
    """
    Test ValueError is raised when series is dict containing series with no
    DatetimeIndex.
    """
    series_1 = series['1'].copy()
    series_1.index = pd.date_range(start='2000-01-01', periods=len(series_1), freq='D')

    series_2 = series['2'].copy()
    series_2.index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='MS')

    series_dict = {'l1': series_1,
                   'l2': series_2}

    err_msg = re.escape(
        ("All series must have a DatetimeIndex index with the same frequency. "
         "Found frequencies: ['<Day>', '<MonthBegin>']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_check_preprocess_series_when_series_is_dict():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_1 = series['1'].copy()
    series_1.index = pd.date_range(start='2000-01-01', periods=len(series_1), freq='D')

    series_2 = series['2'].copy()
    series_2 = series_2.to_frame()
    series_2.index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D')

    series_dict = {'l1': series_1,
                   'l2': series_2}

    series_dict, series_index = check_preprocess_series(series=series_dict)

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series['1'].values,
                  index = pd.date_range(start='2000-01-01', periods=len(series['1']), freq='D'),
                  name  ='l1'
              ),
        'l2': pd.Series(
                  data  = series['2'].values,
                  index = pd.date_range(start='2000-01-01', periods=len(series['2']), freq='D'),
                  name  ='l2'
              ),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(series_dict[k].index, pd.DatetimeIndex)

    indexes_freq = [f'{v.index.freq}' 
                    for v in series_dict.values()]
    assert len(set(indexes_freq)) == 1

    assert series_index is None