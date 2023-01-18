# Unit test predict_bootstrapping ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures import y
from .fixtures import exog
from .fixtures import exog_predict
        

def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression and
    1 step ahead is predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame(
                    data    = np.array([[0.39132925, 0.71907802, 0.45297104, 0.52687879]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=51)
                )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression and
    2 steps ahead are predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame(
                    data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                        [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression and
    1 step ahead is predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                    data    = np.array([[0.39132925, 0.71907802, 0.45297104, 0.52687879]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=51)
                )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression and
    2 steps ahead are predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                    data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                        [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterAutoreg(
                    LinearRegression(),
                    lags = 3,
                    transformer_y = StandardScaler(),
                    transformer_exog = StandardScaler(),
                )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=0)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[0.67998879, 0.67998879, 0.67998879, 0.67998879],
                                    [0.46135022, 0.46135022, 0.46135022, 0.46135022]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
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
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array(
                                [[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]
                        ),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)
        