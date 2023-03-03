# Unit test get_feature_importance ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Fixtures
series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                       '2': pd.Series(np.arange(10))})
                    

def test_exception_is_raised_when_forecaster_is_not_fitted():
    """
    Test exception is raised when calling get_feature_importance() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor = LinearRegression(),
                    lags      = 3,
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
    and it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(series=series)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                    'importance': np.array([0.9446366782006932, 0.0, 0.05536332179930687, 0.0, 0.0])
                })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importance when regressor is RandomForestRegressor with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable
    exog=pd.Series(np.arange(10, 20), name='exog').
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 20), name='exog'))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog', '1', '2'],
                    'importance': np.array([0.73269896, 0., 0.21193772, 0.05536332, 0., 0.])
               })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_LinearRegression():
    """
    Test output of get_feature_importance when regressor is LinearRegression with lags=3
    and it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                    'importance': np.array([3.33333333e-01, 3.33333333e-01, 3.33333333e-01, 
                                            -1.48164367e-16, 1.48164367e-16])
               })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_is_LinearRegression_with_exog():
    """
    Test output of get_feature_importance when regressor is LinearRegression with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable
    exog=pd.Series(np.arange(10, 15), name='exog').
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                          })

    forecaster = ForecasterAutoregMultiSeriesCustom(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 15), name='exog'))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog', '1', '2'],
                    'importance': np.array([2.50000000e-01, 2.50000000e-01, 2.50000000e-01, 
                                            2.50000000e-01, -8.32667268e-17, 8.32667268e-17])
               })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importance_when_regressor_no_attributes():
    """
    Test output of get_feature_importance when regressor is MLPRegressor with lags=3
    and it is trained with series pandas DataFrame. Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                          })

    forecaster = ForecasterAutoregMultiSeriesCustom(MLPRegressor(solver = 'lbfgs', max_iter= 50, random_state=123), lags=3)
    forecaster.fit(series=series)
    expected = None
    results = forecaster.get_feature_importance()

    assert results is expected


def test_output_get_feature_importance_when_pipeline():
    """
    Test output of get_feature_importance when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor = make_pipeline(StandardScaler(), LinearRegression()),
                    lags      = 3
                    )
    forecaster.fit(series=series)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                    'importance': np.array([6.66666667e-01, 6.66666667e-01, 6.66666667e-01, 
                                            -7.40821837e-17, 7.40821837e-17])
               })
    results = forecaster.get_feature_importance()

    pd.testing.assert_frame_equal(expected, results)