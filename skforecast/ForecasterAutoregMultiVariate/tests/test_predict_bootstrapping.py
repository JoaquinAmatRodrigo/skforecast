# Unit test predict_bootstrapping ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Fixtures
from .fixtures_ForecasterAutoregMultiVariate import series
from .fixtures_ForecasterAutoregMultiVariate import exog
from .fixtures_ForecasterAutoregMultiVariate import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)

    err_msg = re.escape(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=5)


def test_predict_bootstrapping_ValueError_when_not_in_sample_residuals_for_some_step():
    """
    Test ValueError is raised when in_sample_residuals=True but there is no
    residuals for some step.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=2)
    forecaster.fit(series=series)
    forecaster.in_sample_residuals = {2: np.array([1, 2, 3])}

    err_msg = re.escape(
                (f"Not `forecaster.in_sample_residuals` for steps: "
                 f"{set([1, 2]) - set(forecaster.in_sample_residuals.keys())}.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=None, in_sample_residuals=True)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=2)
    forecaster.fit(series=series)

    err_msg = re.escape(
                ("`forecaster.out_sample_residuals` is `None`. Use "
                 "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                 "before `predict_interval()`, `predict_bootstrapping()`, "
                 "`predict_quantiles()` or `predict_dist()`.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_not_out_sample_residuals_for_all_steps_predicted():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is not available for all steps predicted.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    residuals = {2: np.array([1, 2, 3, 4, 5]), 
                 3: np.array([1, 2, 3, 4, 5])}
    forecaster.out_sample_residuals = residuals

    err_msg = re.escape(
                    (f"Not `forecaster.out_sample_residuals` for steps: "
                     f"{set([1, 2]) - set(forecaster.out_sample_residuals.keys())}. "
                     f"Use method `set_out_sample_residuals()`.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=[1, 2], in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_step_out_sample_residuals_value_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals has a step with a None.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    residuals = {1: np.array([1, 2, 3, 4, 5]),
                 2: np.array([1, 2, 3, 4, 5])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
                    ("forecaster residuals for step 3 are `None`. Check forecaster.out_sample_residuals.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_step_out_sample_residuals_value_contains_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals has a step with a None value.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    residuals = {1: np.array([1, 2, 3, 4, 5]),
                 2: np.array([1, 2, 3, 4, 5]), 
                 3: np.array([1, 2, 3, 4, None])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
                    ("forecaster residuals for step 3 contains `None` values. Check forecaster.out_sample_residuals.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


@pytest.mark.parametrize("steps", [2, [1, 2], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(steps):
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(steps=steps, exog=exog_predict, 
                                               n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[0.68370403, 0.61428329, 0.38628787, 0.46804785],
                                     [0.53932503, 0.13843292, 0.39752999, 0.86694953]]),
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
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, 
                                               n_boot=4, in_sample_residuals=False)
    expected = pd.DataFrame(
                    data = np.array([[0.68370403, 0.61428329, 0.38628787, 0.46804785],
                                     [0.53932503, 0.13843292, 0.39752999, 0.86694953]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_fixed():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals that are fixed.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3
                 )
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.in_sample_residuals = {1: pd.Series([1, 1, 1, 1, 1, 1, 1]),
                                      2: pd.Series([5, 5, 5, 5, 5, 5, 5])}
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict['exog_1'], 
                                               n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[1.57457831, 1.57457831, 1.57457831, 1.57457831],
                                     [5.3777698 , 5.3777698 , 5.3777698 , 5.3777698 ]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    
    pd.testing.assert_frame_equal(expected, results)        