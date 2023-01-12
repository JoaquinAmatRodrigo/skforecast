# Unit test predict_bootstrapping ForecasterAutoreg
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
        

def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                    data    = np.array([[20., 20., 20., 20.]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = [10]
               )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    2 steps are predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                data = np.array([[20., 20., 20., 20.],
                                 [24.3333, 24.3333, 24.3333, 24.3333]]),
                columns = [f"pred_boot_{i}" for i in range(4)],
                index   = [10, 11]
            )          
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    1 step is predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                    data    = np.array([[20., 20., 20., 20.]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = [10]
               )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    2 steps are predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                data = np.array([[20., 20., 20., 20.],
                                 [24.3333, 24.3333, 24.3333, 24.3333]]),
                columns = [f"pred_boot_{i}" for i in range(4)],
                index   = [10, 11]
            )          
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)
    