# Unit test check_preprocess_exog_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesExogWarning
from skforecast.utils import check_preprocess_series
from skforecast.utils import check_preprocess_exog_multiseries
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import series
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import series_2
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import series_as_dict
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import exog
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import exog_as_dict
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import exog_predict

def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_valid_type():
    """
    Test TypeError is raised when exog is not a pandas Series, DataFrame or dict.
    """
    _, series_indexes = check_preprocess_series(series=series)

    not_valid_exog = 'not_valid_exog'

    err_msg = re.escape(
        ("`exog` must be a pandas Series, DataFrame or dict. "
         "Got <class 'str'>.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_col_names     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


@pytest.mark.parametrize("not_valid_exog", 
                         [pd.Series([1, 2, 3], name='exog1'),
                          pd.DataFrame([1, 2, 3], columns=['exog1'])], 
                         ids=lambda exog: f'exog: {type(exog)}')
def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_dict_and_series_is_dict(not_valid_exog):
    """
    Test TypeError is raised when exog is a pandas Series or DataFrame and
    input series is a dict (input_series_is_dict = True).
    """
    _, series_indexes = check_preprocess_series(series=series_as_dict)

    err_msg = re.escape(
        (f"`exog` must be a dict of DataFrames or Series if "
         f"`series` is a dict. Got {type(not_valid_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = True,
            series_indexes       = series_indexes,
            series_col_names     = ['l1', 'l2'],
            exog                 = not_valid_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


@pytest.mark.parametrize("not_valid_exog", 
                         [exog['exog_1'].iloc[:30].copy(),
                          exog.iloc[:30].copy()], 
                         ids=lambda exog: f'exog: {type(exog)}')
def test_ValueError_check_preprocess_exog_multiseries_when_exog_pandas_with_different_length_from_series(not_valid_exog):
    """
    Test ValueError is raised when exog is a pandas Series or DataFrame with a
    different length from input series (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series)

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(not_valid_exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_col_names     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_pandas_with_different_index_from_series():
    """
    Test ValueError is raised when exog is a pandas Series or DataFrame with a
    different index from input series (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series)

    err_msg = re.escape(
        ("Different index for `series` and `exog`. They must be equal "
         "to ensure the correct alignment of values.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_col_names     = ['1', '2'],
            exog                 = exog_predict,
            exog_dict            = {'1': None, '2': None}
        )
    

def test_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame.
    """
    _, series_indexes = check_preprocess_series(series=series)

    exog_dict, exog_col_names = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_col_names     = ['1', '2'],
                                    exog                 = exog,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog,
        '2': exog
    }
    expected_exog_col_names = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_col_names) - set(expected_exog_col_names)) == 0
    

def test_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame_with_RangeIndex_not_starting_0():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame with 
    a RangeIndex that doesn't start at 0.
    """
    series_range = series.copy()
    series_range.index = pd.RangeIndex(start=3, stop=3+len(series), step=1)
    _, series_indexes = check_preprocess_series(series=series_range)
    
    exog_range = exog.copy()
    exog_range.index = pd.RangeIndex(start=3, stop=3+len(exog_range), step=1)

    exog_dict, exog_col_names = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_col_names     = ['1', '2'],
                                    exog                 = exog_range,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog_range,
        '2': exog_range
    }
    expected_exog_col_names = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_col_names) - set(expected_exog_col_names)) == 0
    

def test_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame_with_DatetimeIndex():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    _, series_indexes = check_preprocess_series(series=series_datetime)

    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(
        start='2000-01-01', periods=len(exog_datetime), freq='D'
    )

    exog_dict, exog_col_names = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_col_names     = ['1', '2'],
                                    exog                 = exog_datetime,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog_datetime,
        '2': exog_datetime
    }
    expected_exog_col_names = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_col_names) - set(expected_exog_col_names)) == 0


def test_MissingValuesExogWarning_check_preprocess_exog_multiseries_when_exog_is_dict_without_all_series():
    """
    Test MissingValuesExogWarning is issues when exog is a dict without all the 
    series as keys.
    """
    _, series_indexes = check_preprocess_series(series=series_2)

    incomplete_exog = exog_as_dict.copy()
    incomplete_exog.pop('l1')

    warn_msg = re.escape(
        ("{'l1'} not present in `exog`. All values "
         "of the exogenous variables for these series will be NaN.")
    )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_col_names     = ['l1', 'l2'],
            exog                 = incomplete_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_dict_with_different_length_from_series():
    """
    Test ValueError is raised when exog is a dict with different length from 
    input series as pandas DataFrame (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series_2)

    not_valid_exog = exog_as_dict.copy()
    not_valid_exog['l2'] = not_valid_exog['l2'].iloc[:30]

    err_msg = re.escape(
        (f"`exog` for series 'l2' must have same number of samples as `series`. "
         f"length `exog`: (30), length `series`: (50)")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_col_names     = ['l1', 'l2'],
            exog                 = not_valid_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )

# TODO: Next test is:
# raise ValueError(
#     (f"Different index for series '{k}' and its exog. "
#     f"When `series` is a pandas DataFrame, they must "
#     f"be equal to ensure the correct alignment of values.")
# )




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