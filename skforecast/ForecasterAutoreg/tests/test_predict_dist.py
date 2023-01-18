# Unit test predict_dist ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


# Fixtures
from .fixtures import y
from .fixtures import exog
from .fixtures import exog_predict


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoreg(
                    LinearRegression(),
                    lags = 3,
                    transformer_y = StandardScaler(),
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
                    data = np.array([[0.52256427, 0.123189  ],
                                     [0.11121952, 0.17472563]]),
                    columns = ['loc', 'scale'],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoreg(
                    LinearRegression(),
                    lags = 3,
                    transformer_y = StandardScaler(),
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
                    data = np.array([[0.52256427, 0.123189  ],
                                     [0.11121952, 0.17472563]]),
                    columns = ['loc', 'scale'],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)

