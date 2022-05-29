# Unit test predict ForecasterAutoregMultiSeries
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


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