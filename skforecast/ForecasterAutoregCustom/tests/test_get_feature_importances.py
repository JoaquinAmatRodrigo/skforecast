# Unit test get_feature_importances ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]
    
    return lags


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_when_regressor_is_RandomForest():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor 
    with create_predictors and it is trained with y=pd.Series(np.arange(10)).
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))

    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4'],
                   'importance': np.array([0.82142857, 0., 0.17857143, 0., 0.])
               })

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor 
    with create_predictors and it is trained with y=pd.Series(np.arange(10)) and 
    a exogenous variable exog=pd.Series(np.arange(10, 20), name='exog').
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)), exog=pd.Series(np.arange(10, 20), name='exog'))
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog'],
                   'importance': np.array([0.76190476, 0., 0.05952381, 0.17857143, 0., 0.])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression():
    """
    Test output of get_feature_importances when regressor is LinearRegression with
    create_predictors and it is trained with y=pd.Series(np.arange(7)).
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(7)))

    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4'],
                   'importance': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_with_exog():
    """
    Test output of get_feature_importances when regressor is LinearRegression with
    create_predictors and it is trained with y=pd.Series(np.arange(7)) and 
    a exogenous variable exog=pd.Series(np.arange(10, 17), name='exog').
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(7)), exog=pd.Series(np.arange(10, 17), name='exog'))
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog'],
                   'importance': np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667])
               })

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_and_UserWarning_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when regressor is MLPRegressor with create_predictors
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a UserWarning is issues.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = MLPRegressor(solver = 'lbfgs', max_iter= 100, random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))

    estimator = forecaster.regressor
    expected = None

    warn_msg = re.escape(
                (f"Impossible to access feature importances for regressor of type "
                 f"{type(estimator)}. This method is only valid when the "
                 f"regressor stores internally the feature importances in the "
                 f"attribute `feature_importances_` or `coef_`.")
            )
    with pytest.warns(UserWarning, match = warn_msg):
        results = forecaster.get_feature_importances()
        assert results is expected


def test_output_get_feature_importances_when_pipeline_LinearRegression():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + LinearRegression with create_predictors),
    it is trained with y=pd.Series(np.arange(7)).
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = make_pipeline(StandardScaler(), LinearRegression()),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(7)))

    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4'],
                   'importance': np.array([0.1, 0.1, 0.1, 0.1, 0.1])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_pipeline_RandomForestRegressor():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + RandomForestRegressor with create_predictors),
    it is trained with y=pd.Series(np.arange(7)).
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = make_pipeline(StandardScaler(), 
                                                    RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123)),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4'],
                   'importance': np.array([0.82142857, 0., 0.17857143, 0., 0.])
               })

    pd.testing.assert_frame_equal(results, expected)