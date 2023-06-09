# Unit test fit ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags


@pytest.mark.parametrize('exog', ['l1', ['l1'], ['l1', 'l2']])
def test_fit_ValueError_when_exog_columns_same_as_series_col_names(exog):
    """
    Test ValueError is raised when an exog column is named the same as
    the series levels.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    series_col_names = ['l1', 'l2']
    exog_col_names = exog if isinstance(exog, list) else [exog]

    err_msg = re.escape(
                    (f'`exog` cannot contain a column named the same as one of the series'
                     f' (column names of series).\n'
                     f'    `series` columns : {series_col_names}.\n'
                     f'    `exog`   columns : {exog_col_names}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series, exog=series[exog], store_in_sample_residuals=False)


def test_fit_correct_dict_create_series_weights_weight_func_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts, series_weights_,
    weight_func_, transformer_series_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10)), 
                           'l3': pd.Series(np.arange(10))})
                    
    series.index = pd.DatetimeIndex(
                       ['2022-01-04', '2022-01-05', '2022-01-06', 
                        '2022-01-07', '2022-01-08', '2022-01-09', 
                        '2022-01-10', '2022-01-11', '2022-01-12', 
                        '2022-01-13'], dtype='datetime64[ns]', freq='D' 
                   )

    def custom_weights(index):
        """
        Return 0 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
        """
        weights = np.where(
                    (index >= '2022-01-08') & (index <= '2022-01-10'),
                    0,
                    1
                )
        
        return weights

    transformer_series = {'l1': StandardScaler()}
    weight_func = {'l2': custom_weights}    
    series_weights = {'l1': 3., 'l3': 0.5, 'l4': 2.}

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = transformer_series,
                     weight_func        = weight_func,
                     series_weights     = series_weights
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)

    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None, 'l3': None}
    expected_weight_func_ = {'l1': lambda index: np.ones_like(index, dtype=float), 'l2': custom_weights, 'l3': lambda index: np.ones_like(index, dtype=float)}
    expected_series_weights_ = {'l1': 3., 'l2': 1., 'l3': 0.5}

    assert forecaster.transformer_series_ == expected_transformer_series_
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_

    forecaster.fit(series=series[['l1', 'l2']], store_in_sample_residuals=False)

    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None}
    expected_weight_func_ = {'l1': lambda index: np.ones_like(index, dtype=float), 'l2': custom_weights}
    expected_series_weights_ = {'l1': 3., 'l2': 1.}

    assert forecaster.transformer_series_ == expected_transformer_series_
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    series.index = pd.date_range(start='2022-01-01', periods=5, freq='1D')

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    expected = series.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    expected = series.index.step
    results = forecaster.index_freq

    assert results == expected


def test_fit_in_sample_residuals_stored():
    """
    Test that values of in_sample_residuals are stored after fitting
    when `store_in_sample_residuals=True`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    expected = {'1': np.array([-4.4408921e-16, 0.0000000e+00]),
                '2': np.array([0., 0.])}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert all(all(np.isclose(results[k], expected[k])) for k in expected.keys())


def test_fit_in_sample_residuals_stored_XGBRegressor():
    """
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = XGBRegressor(random_state=123),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=True)
    expected = {'1': np.array([-0.00049472,  0.00049543]),
                '2': np.array([-0.00049472,  0.00049543])}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert all(all(np.isclose(results[k], expected[k])) for k in expected.keys())


def test_fit_same_residuals_when_residuals_greater_than_1000():
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster. Residuals shouldn't be more than 
    1000 values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(1010)), 
                           '2': pd.Series(np.arange(1010))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals

    assert isinstance(results_1, dict)
    assert all(isinstance(x, np.ndarray) for x in results_1.values())
    assert isinstance(results_2, dict)
    assert all(isinstance(x, np.ndarray) for x in results_2.values())
    assert results_1.keys() == results_2.keys()
    assert all(len(results_1[k] == 1000) for k in results_1.keys())
    assert all(len(results_2[k] == 1000) for k in results_2.keys())
    assert all(all(results_1[k] == results_2[k]) for k in results_2.keys())


def test_fit_in_sample_residuals_not_stored():
    """
    Test that values of in_sample_residuals are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected = {'1': None, '2': None}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    assert all(results[k] == expected[k] for k in expected.keys())


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                          })

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    expected = pd.DataFrame({'1': pd.Series(np.array([2, 3, 4])), 
                             '2': pd.Series(np.array([2, 3, 4]))
                            })
    expected.index = pd.RangeIndex(start=2, stop=5, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)