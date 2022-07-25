# Unit test predict ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


@pytest.fixture(params=[('1', [50., 51., 52., 53., 54.]), 
                        ('2', [100., 101., 102., 103., 104.])],
                        ids=lambda d: f"level: {d[0]}, preds: {d[-1]}")
def expected_pandas_series(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    level = request.param[0]
    expected = pd.Series(
                    data = np.array(request.param[1]),
                    index = pd.RangeIndex(start=50, stop=55, step=1),
                    name = 'pred'
               )

    return level, expected


def test_predict_output_when_regressor_is_LinearRegression_with_fixture(expected_pandas_series):
    '''
    Test predict output when using LinearRegression as regressor with pytest fixture.
    This test is equivalent to the next one.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, level=expected_pandas_series[0])
    expected = expected_pandas_series[1]

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression():
    '''
    Test predict output when using LinearRegression as regressor.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    predictions_1 = forecaster.predict(steps=5, level='1')
    expected_1 = pd.Series(
                    data = np.array([50., 51., 52., 53., 54.]),
                    index = pd.RangeIndex(start=50, stop=55, step=1),
                    name = 'pred'
                 )

    predictions_2 = forecaster.predict(steps=5, level='2')
    expected_2 = pd.Series(
                    data = np.array([100., 101., 102., 103., 104.]),
                    index = pd.RangeIndex(start=50, stop=55, step=1),
                    name = 'pred'
                 )

    pd.testing.assert_series_equal(predictions_1, expected_1)
    pd.testing.assert_series_equal(predictions_2, expected_2)