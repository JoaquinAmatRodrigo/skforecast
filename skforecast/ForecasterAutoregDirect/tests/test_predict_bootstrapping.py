# Unit test predict_bootstrapping ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_ForecasterAutoregDirect import y
from .fixtures_ForecasterAutoregDirect import exog
from .fixtures_ForecasterAutoregDirect import exog_predict


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
                ('`forecaster.out_sample_residuals` is `None`. Use '
                 '`in_sample_residuals=True` or method `set_out_sample_residuals()` '
                 'before `predict_interval()`, `predict_bootstrapping()` or '
                 '`predict_dist()`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, in_sample_residuals=False)


@pytest.mark.parametrize("steps", [2, [1, 2], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(steps):
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
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
    results = forecaster.predict_bootstrapping(steps=steps, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[0.73195423, 0.6896871 , 0.20248689, 0.54685553],
                                     [0.13621074, 0.29541489, 0.51606685, 0.3341628 ]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
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
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=False)
    expected = pd.DataFrame(
                    data = np.array([[0.73195423, 0.6896871 , 0.20248689, 0.54685553],
                                     [0.13621074, 0.29541489, 0.51606685, 0.3341628 ]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_fixed():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals that are fixed.
    """

    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(),
                     steps     = 2,
                     lags      = 3
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals = {1: pd.Series([1, 1, 1, 1, 1, 1, 1]),
                                      2: pd.Series([5, 5, 5, 5, 5, 5, 5])}
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[1.67523588, 1.67523588, 1.67523588, 1.67523588],
                                     [5.38024988, 5.38024988, 5.38024988, 5.38024988]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)        