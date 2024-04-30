# Unit test _recursive_predict ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom

def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot"],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_regressor_is_LinearRegression(encoding):
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 5,
                     encoding           = encoding,
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


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot"],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_regressor_is_LinearRegression_StandardScaler(encoding):
    """
    Test _recursive_predict output when using LinearRegression as regressor and
    StandardScaler.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 5,
                     encoding       = encoding 
                 )
    forecaster.fit(series=series)
    level = '1'
    predictions_1 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_1 = np.array([47.07918986, 47.49389001, 47.79185048, 47.94978724, 47.93977217])

    level = '2'
    predictions_2 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_2 = np.array([96.94237815, 97.32979082, 97.59502126, 97.71369989, 97.65659655])

    assert predictions_1 == approx(expected_1)
    assert predictions_2 == approx(expected_2)