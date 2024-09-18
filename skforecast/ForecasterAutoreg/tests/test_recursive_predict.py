# Unit test _recursive_predict ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Fixtures
from .fixtures_ForecasterAutoreg import y
from .fixtures_ForecasterAutoreg import exog
from .fixtures_ForecasterAutoreg import exog_predict


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
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

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1.745476, 1.803196, 1.865844, 1.930923, 1.997202])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals_zero():
    """
    Test _recursive_predict output with residuals when all residuals are zero.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0], [0], [0], [0], [0]])
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values,
                      residuals          = residuals,
                      binned_residuals   = False
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals_last_step():
    """
    Test _recursive_predict output with residuals when all residuals are zero 
    except the last step.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0], [0], [0], [0], [100]])
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values,
                      residuals          = residuals,
                      binned_residuals   = False
                  )
    
    expected = np.array([50., 51., 52., 53., 154.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals():
    """
    Test _recursive_predict output with residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[10], [20], [30], [40], [50]])
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values,
                      residuals          = residuals,
                      binned_residuals   = False
                  )
    
    expected = np.array([60., 74.333333, 93.111111, 117.814815, 147.08642])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_binned_residuals():
    """
    Test _recursive_predict output with binned residuals.
    """
    forecaster = ForecasterAutoreg(LGBMRegressor(verbose=-1), lags=3)
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values,
                      residuals          = forecaster.in_sample_residuals_by_bin_,
                      binned_residuals   = True,
                      rng                = np.random.default_rng(seed=123)
                  )
    
    expected = np.array(
                   [0.526382, 0.623953, 0.322959, 0.181225, 0.618064, 
                    0.719469, 0.634401, 0.724455, 0.611024, 0.640175]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)
