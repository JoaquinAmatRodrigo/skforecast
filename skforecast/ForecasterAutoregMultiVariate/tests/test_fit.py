# Unit test fit ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@pytest.mark.parametrize('exog', ['l1', ['l1'], ['l1', 'l2']])
def test_fit_exception_when_exog_columns_same_as_series_col_names(exog):
    """
    Test exception is raised when an exog column is named the same as
    the series columns.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)
    series_col_names = list(series.columns)
    exog_col_names = exog if isinstance(exog, list) else [exog]

    err_msg = re.escape(
                    (f'`exog` cannot contain a column named the same as one of the series'
                     f' (column names of series).\n'
                     f'    `series` columns : {series_col_names}.\n'
                     f'    `exog`   columns : {exog_col_names}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series, exog=series[exog])


def test_fit_correct_dict_create_weight_func_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts transformer_series_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10)), 
                           'l3': pd.Series(np.arange(10))})

    transformer_series = {'l1': StandardScaler(), 'l3': StandardScaler(), 'l4': StandardScaler()}

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(), 
                     level              = 'l1',
                     lags               = 3,
                     steps              = 2,
                     transformer_series = transformer_series
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None, 'l3': forecaster.transformer_series_['l3']}

    assert forecaster.transformer_series_ == expected_transformer_series_

    forecaster.fit(series=series[['l1', 'l2']], store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None}

    assert forecaster.transformer_series_ == expected_transformer_series_


def test_forecaster_index_freq_stored():
    """
    Test series.index.freqstr is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))
                          })

    series.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')

    forecaster = ForecasterAutoregMultiVariate(XGBRegressor(random_state=123), 
                                               level='l1', lags=3, steps=1)
    forecaster.fit(series=series)
    expected = series.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))
                          })
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), 
                                               level='l1', lags=3, steps=2)
    forecaster.fit(series=series)
    expected = series.index.step
    results = forecaster.index_freq

    assert results == expected


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))
                          })

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), 
                                               level='l1', lags=3, steps=2)
    forecaster.fit(series=series)
    expected = pd.DataFrame({'l1': pd.Series(np.array([7, 8, 9])), 
                             'l2': pd.Series(np.array([57, 58, 59]))
                            })
    expected.index = pd.RangeIndex(start=7, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)


def test_fit_last_window_stored_when_different_lags():
    """
    Test that values of last window are stored after fitting when different lags
    configurations.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))
                          })

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags = {'l1': 3, 'l2': [1, 5]}, 
                                               steps = 2)
    forecaster.fit(series=series)
    expected = pd.DataFrame({'l1': pd.Series(np.array([5, 6, 7, 8, 9])), 
                             'l2': pd.Series(np.array([105, 106, 107, 108, 109]))
                            })
    expected.index = pd.RangeIndex(start=5, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)