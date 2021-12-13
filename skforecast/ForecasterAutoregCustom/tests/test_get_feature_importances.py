from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_predictors(y):
    '''
    Create first 5 lags of a time series.
    '''
    
    lags = y[-1:-6:-1]
    
    return lags 


def test_output_get_feature_importance_when_regressor_is_RandomForest():
    '''
    '''
    forecaster = ForecasterAutoregCustom(
                        regressor      = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = np.array([0.82142857, 0., 0.17857143, 0., 0.])
    expected = pd.DataFrame({
                    'feature': ['custom_predictor_0', 'custom_predictor_1',
                                'custom_predictor_2', 'custom_predictor_3',
                                'custom_predictor_4'],
                    'importance': np.array([0.82142857, 0., 0.17857143, 0., 0.])
                })
    results = forecaster.get_feature_importance()
    assert (results['feature'] == expected['feature']).all()
    assert results['importance'].values == approx(expected['importance'].values)
    

def test_output_get_feature_importance_when_regressor_is_linear_model():
    '''
    '''
    forecaster = ForecasterAutoregCustom(
                            regressor      = LinearRegression(),
                            fun_predictors = create_predictors,
                            window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(6)))
    expected = None
    results = forecaster.get_feature_importance()
    assert results is expected
    
