# Unit test _recursive_predict ForecasterAutoregCustom
# ==============================================================================
import pytest
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom

def create_predictors(y):  # pragma: no cover
    """
    Create first 5 lags of a time series.
    """

    lags = y[-1:-6:-1]

    return lags

def create_predictors_1_5(y):  # pragma: no cover
    """
    Create lags 1 and 5 of a time series.
    """

    lag_1 = y[-1]
    lag_5 = y[-5]
    lags = np.hstack([lag_1, lag_5])

    return lags  


def test_recursive_predict_ValueError_when_fun_predictors_return_nan():
    """
    Test ValueError is raised when y and exog have different index.
    """
    forecaster = ForecasterAutoregCustom(
        regressor=LinearRegression(), fun_predictors=create_predictors, window_size=5
    )
    forecaster.fit(
        y=pd.Series(
            np.arange(10),
            index=pd.date_range(start="2022-01-01", periods=10, freq="1D"),
        )
    )

    err_msg = re.escape(
        f"`{forecaster.fun_predictors.__name__}` is returning `NaN` values."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._recursive_predict(last_window=np.array([np.nan]), steps=3)


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster._recursive_predict(
                      steps       = 5,
                      last_window = forecaster.last_window.to_numpy().ravel(),
                      exog        = None
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler():
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors_1_5,
                     window_size    = 5,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))
    predictions = forecaster._recursive_predict(
                      steps       = 5,
                      last_window = forecaster.last_window.to_numpy().ravel(),
                      exog        = None
                  )
    
    expected = np.array([46.571365, 45.866715, 46.012391, 46.577477, 47.34943])

    np.testing.assert_array_almost_equal(predictions, expected)
