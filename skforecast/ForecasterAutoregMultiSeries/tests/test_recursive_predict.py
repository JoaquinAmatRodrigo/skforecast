# Unit test _recursive_predict ForecasterAutoregMultiSeries
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    level = '1'
    predictions_1 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_1 = np.array([50., 51., 52., 53., 54.])

    level = '2'
    predictions_2 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_2 = np.array([100., 101., 102., 103., 104.])

    assert predictions_1 == approx(expected_1)
    assert predictions_2 == approx(expected_2)