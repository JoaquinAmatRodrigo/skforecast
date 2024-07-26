# Unit test _recursive_predict ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot", None],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_regressor_is_LinearRegression(encoding):
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))})

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
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

    np.testing.assert_array_almost_equal(predictions_1, expected_1)
    np.testing.assert_array_almost_equal(predictions_2, expected_2)


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot"],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler(encoding):
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))})

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = encoding,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    level = '1'
    predictions_1 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_1 = np.array([46.8874583, 46.33497711, 46.55721213, 47.16416632, 47.96216963])

    level = '2'
    predictions_2 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_2 = np.array([96.54682709, 95.82519059, 95.96342415, 96.52866375, 97.30595186])

    np.testing.assert_array_almost_equal(predictions_1, expected_1)
    np.testing.assert_array_almost_equal(predictions_2, expected_2)


def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler_encoding_None():
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler with encoding=None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))})

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = None,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    level = '1'
    predictions_1 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_1 = np.array([46.83128368, 46.25031411, 46.45861438, 47.05926117, 47.85494363])

    level = '2'
    predictions_2 = forecaster._recursive_predict(
                        steps       = 5,
                        level       = level,
                        last_window = forecaster.last_window[level].to_numpy(),
                        exog        = None
                    )
    expected_2 = np.array([96.54134313, 95.81624394, 95.95289729, 96.51792835, 97.29590622])

    np.testing.assert_array_almost_equal(predictions_1, expected_1)
    np.testing.assert_array_almost_equal(predictions_2, expected_2)