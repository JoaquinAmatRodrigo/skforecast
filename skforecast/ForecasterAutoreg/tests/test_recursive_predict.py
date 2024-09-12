# Unit test _recursive_predict ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoreg import ForecasterAutoreg


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster._recursive_predict(
                      steps       = 5,
                      last_window = forecaster.last_window_.to_numpy().ravel(),
                      exog        = None
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler():
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler.
    """
    forecaster = ForecasterAutoreg(
                     regressor     = Ridge(random_state=123),
                     lags          = [1, 5],
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))
    predictions = forecaster._recursive_predict(
                      steps       = 5,
                      last_window = forecaster.last_window_.to_numpy().ravel(),
                      exog        = None
                  )
    
    expected = np.array([46.571365, 45.866715, 46.012391, 46.577477, 47.34943])

    np.testing.assert_array_almost_equal(predictions, expected)
