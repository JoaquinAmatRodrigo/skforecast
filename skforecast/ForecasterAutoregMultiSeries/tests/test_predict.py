# Unit test predict ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_ForecasterAutoregMultiSeries import series
from .fixtures_ForecasterAutoregMultiSeries import exog
from .fixtures_ForecasterAutoregMultiSeries import exog_predict

series_2 = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                         '2': pd.Series(np.arange(start=50, stop=100))})


@pytest.fixture(params=[('1'  , [50., 51., 52., 53., 54.]), 
                        (['2'], [100., 101., 102., 103., 104.]),
                        (['1', '2'], [[50., 100.],
                                      [51., 101.],
                                      [52., 102.],
                                      [53., 103.],
                                      [54., 104.]])
                        ],
                        ids=lambda d: f'levels: {d[0]}, preds: {d[1]}')
def expected_pandas_dataframe(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    levels = request.param[0]
    levels_names = [levels] if isinstance(levels, str) else levels
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = levels_names,
                      index   = pd.RangeIndex(start=50, stop=55, step=1)
                  )

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_with_fixture(expected_pandas_dataframe):
    """
    Test predict output when using LinearRegression as regressor with pytest fixture.
    This test is equivalent to the next one.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions = forecaster.predict(steps=5, levels=expected_pandas_dataframe[0])
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression():
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions_1 = forecaster.predict(steps=5, levels='1')
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'])
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=None)
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_last_window():
    """
    Test predict output when using LinearRegression as regressor and last_window.
    """
    last_window = pd.DataFrame(
                      {'1': [45, 46, 47, 48, 49], 
                       '2': [95, 96, 97, 98, 99], 
                       '3': [1, 2, 3, 4, 5]}, 
                      index = pd.RangeIndex(start=45, stop=50, step=1)
                  )

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions_1 = forecaster.predict(steps=5, levels='1', last_window=last_window)
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'], last_window=last_window)
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=['1', '2'], last_window=last_window)
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, levels='1')
    expected = pd.DataFrame(
                   data    = np.array([0.52791431, 0.44509712, 0.42176045, 0.48087237, 0.48268008]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler()}
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, levels=['1'])
    expected = pd.DataFrame(
                   data    = np.array([0.59619193, 0.46282914, 0.41738496, 0.48522676, 0.47525733]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=5, levels='1', exog=exog_predict)
    expected = pd.DataFrame(
                   data    = np.array([0.53267333, 0.44478046, 0.52579563, 0.57391142, 0.54633594]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)