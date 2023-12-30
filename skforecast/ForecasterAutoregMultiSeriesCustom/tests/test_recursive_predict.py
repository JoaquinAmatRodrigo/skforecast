# Unit test _recursive_predict ForecasterAutoregMultiSeriesCustom
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression

def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 5,
                     transformer_series = None
                 )
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


def test_recursive_predict_output_when_regressor_is_LinearRegression_StandardScaler():
    """
    Test _recursive_predict output when using LinearRegression as regressor and
    StandardScaler.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 5
                 )
    forecaster.fit(series=series)
    level = '1'
    predictions_1 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_1 = np.array([47.42080743, 47.6709354 , 47.99768315, 48.20560339, 48.26631367])

    level = '2'
    predictions_2 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_2 = np.array([97.42080743, 97.6709354 , 97.99768315, 98.20560339, 98.26631367])

    assert predictions_1 == approx(expected_1)
    assert predictions_2 == approx(expected_2)