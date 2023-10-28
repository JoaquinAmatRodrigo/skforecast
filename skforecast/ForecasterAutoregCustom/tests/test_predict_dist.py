# Unit test predict_dist ForecasterAutoregCustom
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Fixtures
from .fixtures_ForecasterAutoregCustom import y
from .fixtures_ForecasterAutoregCustom import exog
from .fixtures_ForecasterAutoregCustom import exog_predict

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    
    lags = y[-1:-4:-1]
    
    return lags 


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_dist(
                  steps = 2,
                  exog = exog_predict,
                  distribution = norm,
                  n_boot = 4,
                  in_sample_residuals = True
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.52256427, 0.123189  ],
                                       [0.11121952, 0.17472563]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_dist(
                  steps = 2,
                  exog = exog_predict,
                  distribution = norm,
                  n_boot = 4,
                  in_sample_residuals = False
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.52256427, 0.123189  ],
                                       [0.11121952, 0.17472563]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)