# Unit test get_feature_importance ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def test_exception_is_raised_when_forecaster_is_not_fitted():
    """
    Test exception is raised when calling get_feature_importance() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = RandomForestRegressor(random_state=123),
                    lags = 3,
                    steps = 1
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importance()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importance(step=1)


def test_output_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_1():
    """
    Test output of get_feature_importance for step 1, when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = RandomForestRegressor(random_state=123),
                    lags = 3,
                    steps = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.3902439024390244, 0.3170731707317073, 0.2926829268292683])
                })
    
    pd.testing.assert_frame_equal(results, expected)
  
    
def test_output_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_1_exog_included():
    """
    Test output of get_feature_importance for step 1, when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and
    exog=pd.Series(np.arange(5), name='exog').
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = RandomForestRegressor(random_state=123),
                    lags = 3,
                    steps = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(5), name='exog'))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                    'importance': np.array([0.1951219512195122, 0.0975609756097561,
                                            0.36585365853658536, 0.34146341463414637])
                })
    
    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_get_feature_importance_when_regressor_is_LinearRegression_lags_3_step_1():
    """
    Test output of get_feature_importance for step 1, when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=1)
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.33333333, 0.33333333, 0.33333333])
                })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importance_when_regressor_is_LinearRegression_lags_3_step_1_exog_included():
    """
    Test output of get_feature_importance for step 1, when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and
    exog=pd.Series(np.arange(5), name='exog').
    """
    forecaster = ForecasterAutoregDirect(regressor=LinearRegression(), lags=3, steps=1)
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(5), name='exog'))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                    'importance': np.array([0.25, 0.25, 0.25, 0.25])
                })
    
    pd.testing.assert_frame_equal(results, expected)
    

def test_output_get_feature_importance_when_regressor_no_attributes():
    """
    Test output of get_feature_importance when regressor is MLPRegressor with lags=5
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = MLPRegressor(solver = 'lbfgs', max_iter= 50, random_state=123),
                    lags      = 5,
                    steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    results = forecaster.get_feature_importance(step=1)
    expected = None

    assert results is expected


def test_output_get_feature_importance_when_pipeline():
    """
    Test output of get_feature_importance when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = make_pipeline(StandardScaler(), LinearRegression()),
                    lags      = 3,
                    steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.166667, 0.166667, 0.166667])
                })
    results = forecaster.get_feature_importance(step=1)
    
    pd.testing.assert_frame_equal(expected, results)