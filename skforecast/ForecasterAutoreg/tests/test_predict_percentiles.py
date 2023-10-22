# Unit test predict_percentiles ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_ForecasterAutoreg import y
from .fixtures_ForecasterAutoreg import exog
from .fixtures_ForecasterAutoreg import exog_predict


def test_predict_percentiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_percentiles when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )

    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_percentiles(
                  steps               = 2,
                  exog                = exog_predict,
                  percentiles         = [5, 55, 95],
                  n_boot              = 4,
                  in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[ 0.40057552,  0.50101108,  0.69024814],
                                       [-0.05165537,  0.05909311,  0.34957492]]),
                   columns = ['p5', 'p55', 'p95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_percentiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_percentiles when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_percentiles(
                  steps               = 2,
                  exog                = exog_predict,
                  percentiles         = [5, 55, 95],
                  n_boot              = 4,
                  in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[ 0.40057552,  0.50101108,  0.69024814],
                                       [-0.05165537,  0.05909311,  0.34957492]]),
                   columns = ['p5', 'p55', 'p95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)