# Unit test get_feature_importance ForecasterAutoreg
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def test_exception_is_raised_when_forecaster_is_not_fitted():
    """
    Test exception is raised when calling get_feature_importance() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoreg(
                    regressor = LinearRegression(),
                    lags = 3,
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importance()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importance()


def test_output_get_feature_importance_when_regressor_is_RandomForest():
    """
    Test output of get_feature_importance when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)).
    """
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.94766355, 0., 0.05233645])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importance when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)) and a exogenous variable
    exog=pd.Series(np.arange(10, 20), name='exog').
    """
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), exog=pd.Series(np.arange(10, 20), name='exog'))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                    'importance': np.array([0.94766355, 0.05233645, 0., 0.])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_LinearRegression():
    """
    Test output of get_feature_importance when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.33333333, 0.33333333, 0.33333333])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_LinearRegression_with_exog():
    """
    Test output of get_feature_importance when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and a exogenous variable
    exog=pd.Series(np.arange(10, 15), name='exog').
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(10, 15), name='exog'))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                    'importance': np.array([0.25, 0.25, 0.25, 0.25])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_no_attributes():
    """
    Test output of get_feature_importance when regressor is MLPRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    forecaster = ForecasterAutoreg(MLPRegressor(solver = 'lbfgs', max_iter= 50, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = None
    results = forecaster.get_feature_importance()

    assert results is expected


def test_output_get_feature_importance_when_pipeline():
    """
    Test output of get_feature_importance when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterAutoreg(
                    regressor = make_pipeline(StandardScaler(), LinearRegression()),
                    lags      = 3
                    )
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.166667, 0.166667, 0.166667])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)