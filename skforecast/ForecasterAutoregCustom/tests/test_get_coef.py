# Unit test get_coef ForecasterAutoregCustom
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def create_predictors(y):
    '''
    Create first 5 lags of a time series.
    '''
    
    lags = y[-1:-6:-1]
    
    return lags 
    

def test_output_get_coef_when_regressor_is_LinearRegression():
    '''
    Test output of get_coef when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(7)))
    expected = pd.DataFrame({
                    'feature': ['custom_predictor_0', 'custom_predictor_1',
                                'custom_predictor_2', 'custom_predictor_3',
                                'custom_predictor_4'],
                    'coef': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                })
    results = forecaster.get_coef()
    assert (results['feature'] == expected['feature']).all()
    assert results['coef'].values == approx(expected['coef'].values)
    

def test_get_coef_when_regressor_is_RandomForest():
    '''
    Test output of get_coef when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoregCustom(
                        regressor      = RandomForestRegressor(n_estimators=1, max_depth=2),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(6)))
    expected = None
    results = forecaster.get_coef()
    assert results is expected


def test_output_get_coef_when_pipeline():
    '''
    Test output of get_coef when regressor is pipeline,
    (StandardScaler() + LinearRegression with create_predictors),
    it is trained with y=pd.Series(np.arange(7)).
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = make_pipeline(StandardScaler(), LinearRegression()),
                    fun_predictors = create_predictors,
                    window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(7)))
    expected = pd.DataFrame({
                    'feature': ['custom_predictor_0', 'custom_predictor_1',
                                'custom_predictor_2', 'custom_predictor_3',
                                'custom_predictor_4'],
                    'coef': np.array([0.1, 0.1, 0.1, 0.1, 0.1])
                })
    results = forecaster.get_coef()
    pd.testing.assert_frame_equal(expected, results)