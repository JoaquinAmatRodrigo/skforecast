# Unit test predict_bootstrapping ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )

    err_msg = re.escape(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1)
 

def test_predict_interval_exception_when_out_sample_residuals_is_None():
    """
    Test exception is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
                ("`forecaster.out_sample_residuals` is `None`. Use "
                 "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                 "before `predict_interval()`, `predict_bootstrapping()`, "
                 "`predict_quantiles()` or `predict_dist()`.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, in_sample_residuals=False)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )
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
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )
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
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )
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
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )
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
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor        = LinearRegression(),
                        fun_predictors   = create_predictors,
                        window_size      = 3,
                        transformer_y    = StandardScaler(),
                        transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=0)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                   data    = np.array([[0.67998879, 0.67998879, 0.67998879, 0.67998879],
                                       [0.46135022, 0.46135022, 0.46135022, 0.46135022]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
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
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                   data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                       [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)