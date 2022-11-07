# Unit test get_feature_importance ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                        'l2': pd.Series(np.arange(10))
                        })
exog = pd.Series(np.arange(10), name='exog')


def test_exception_is_raised_when_step_is_not_int():
    """
    Test exception is raised when calling get_feature_importance() and step is 
    not an int.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)
    step = 'not_an_int'

    err_msg = re.escape(f'`step` must be an integer. Got {type(step)}.')
    with pytest.raises(TypeError, match = err_msg):         
        forecaster.get_feature_importance(step=step)


def test_exception_is_raised_when_forecaster_is_not_fitted():
    """
    Test exception is raised when calling get_feature_importance() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importance()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importance(step=1)


@pytest.mark.parametrize("step", [0, 2], ids=lambda step: f'step: {step}')
def test_exception_is_raised_when_step_is_greater_than_forecaster_steps(step):
    """
    Test exception is raised when calling get_feature_importance() and step is 
    less than 1 or greater than the forecaster.steps.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    err_msg = re.escape(
                f"The step must have a value from 1 to the maximum number of steps "
                f"({forecaster.steps}). Got {step}."
            )
    with pytest.raises(ValueError, match = err_msg):         
        forecaster.get_feature_importance(step=step)


def test_output_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_1():
    """
    Test output of get_feature_importance for step 1, when regressor is 
    RandomForestRegressor with lags=3.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.17522374, 0.17408016, 0.17918514, 0.13738555, 0.17994996, 0.15417544])
               })
    
    pd.testing.assert_frame_equal(results, expected)
  
    
def test_output_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_2_exog_included():
    """
    Test output of get_feature_importance for step 2, when regressor is 
    RandomForestRegressor with lags=3 and exog.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l2',
                     lags      = 3,
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.get_feature_importance(step=2)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'exog'],
                   'importance': np.array([0.10651387, 0.19219748, 0.17405775, 0.11926979, 0.13654122, 0.16224002, 0.10917987])
               })
    
    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_get_feature_importance_when_regressor_is_LinearRegression_lags_3_step_1():
    """
    Test output of get_feature_importance for step 1, when regressor is 
    LinearRegression with lags=3.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l2',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importance_when_regressor_is_LinearRegression_lags_3_step_2_exog_included():
    """
    Test output of get_feature_importance for step 1, when regressor is 
    LinearRegression with lags=3 and exog.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.get_feature_importance(step=2)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'exog'],
                   'importance': np.array([0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714])
               })
    
    pd.testing.assert_frame_equal(results, expected)
    

def test_output_get_feature_importance_when_regressor_no_attributes():
    """
    Test output of get_feature_importance when regressor is MLPRegressor with lags=5
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = MLPRegressor(solver='lbfgs', max_iter=50, random_state=123),
                     level     = 'l1',
                     lags      = 5,
                     steps     = 1
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importance(step=1)
    expected = None

    assert results is expected


def test_output_get_feature_importance_when_pipeline():
    """
    Test output of get_feature_importance when regressor is pipeline,
    (StandardScaler() + RandomForestRegressor with lags=3).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.17522374, 0.17408016, 0.17918514, 0.13738555, 0.17994996, 0.15417544])
               })
    
    pd.testing.assert_frame_equal(expected, results)