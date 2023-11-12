# Unit test predict_quantiles ForecasterAutoregDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_ForecasterAutoregDirect import y
from .fixtures_ForecasterAutoregDirect import exog
from .fixtures_ForecasterAutoregDirect import exog_predict


def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )

    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_quantiles(
                  steps               = 2,
                  exog                = exog_predict,
                  quantiles           = [0.05, 0.55, 0.95],
                  n_boot              = 4,
                  in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.25414219, 0.63969605, 0.72561416],
                                       [0.27143513, 0.50052004, 0.78431847]]),
                   columns = ['q_0.05', 'q_0.55', 'q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_quantiles(
                  steps               = 2,
                  exog                = exog_predict,
                  quantiles           = [0.05, 0.55, 0.95],
                  n_boot              = 4,
                  in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.25414219, 0.63969605, 0.72561416],
                                       [0.27143513, 0.50052004, 0.78431847]]),
                   columns = ['q_0.05', 'q_0.55', 'q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)